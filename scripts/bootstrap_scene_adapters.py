#!/usr/bin/env python3
"""Replace monolith-owned scene algorithms with the canonical registry adapter seam.

This one-shot bootstrap transform is kept as provenance for the extraction. It
does not fetch or copy code from scenedetect-rs.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CORE_ADAPTER = r'''
#[derive(Debug, Clone, Copy)]
/// Compatibility weights converted explicitly into the canonical scene contract.
pub struct ContentWeights {
    /// Hue contribution.
    pub delta_hue: f32,
    /// Saturation contribution.
    pub delta_sat: f32,
    /// Luminance contribution.
    pub delta_lum: f32,
    /// Edge contribution.
    pub delta_edges: f32,
}

impl Default for ContentWeights {
    fn default() -> Self {
        Self {
            delta_hue: 1.0,
            delta_sat: 1.0,
            delta_lum: 1.0,
            delta_edges: 0.0,
        }
    }
}

impl ContentWeights {
    /// Compatibility preset for luminance-only scoring.
    pub const LUMA_ONLY: Self = Self {
        delta_hue: 0.0,
        delta_sat: 0.0,
        delta_lum: 1.0,
        delta_edges: 0.0,
    };

    fn canonical(self) -> scenedetect_core::ContentWeights {
        scenedetect_core::ContentWeights {
            hue: self.delta_hue as f64,
            saturation: self.delta_sat as f64,
            luminance: self.delta_lum as f64,
            edges: self.delta_edges as f64,
        }
    }
}

#[derive(Debug, Clone)]
/// Compatibility adapter backed by the canonical `scenedetect-core` detector.
pub struct ContentDetector {
    threshold: f32,
    weights: ContentWeights,
    luma_only: bool,
    min_scene_len: u64,
    filter_mode: FlashFilterMode,
    frames: Vec<scenedetect_core::Frame>,
    emitted: BTreeSet<u64>,
}

impl Default for ContentDetector {
    fn default() -> Self {
        Self::new(27.0, 15)
    }
}

impl ContentDetector {
    /// Metric keys retained by the visual compatibility surface.
    pub const METRIC_KEYS: &'static [&'static str] = &[
        "content_val",
        "delta_hue",
        "delta_sat",
        "delta_lum",
        "delta_edges",
    ];

    /// Creates a canonical-detector compatibility adapter.
    pub fn new(threshold: f32, min_scene_len: u64) -> Self {
        Self {
            threshold,
            weights: ContentWeights::default(),
            luma_only: false,
            min_scene_len,
            filter_mode: FlashFilterMode::Merge,
            frames: Vec::new(),
            emitted: BTreeSet::new(),
        }
    }

    /// Sets compatibility weights before conversion to canonical weights.
    pub fn with_weights(mut self, weights: ContentWeights) -> Self {
        self.weights = weights;
        self
    }

    /// Selects canonical luminance-only scoring.
    pub fn luma_only(mut self, value: bool) -> Self {
        self.luma_only = value;
        self
    }

    /// Retains the legacy validation seam; kernel policy belongs to the canonical owner.
    pub fn kernel_size(self, value: Option<usize>) -> Result<Self> {
        if let Some(size) = value {
            if size < 3 || size % 2 == 0 {
                return Err(DetectError::InvalidArgument(
                    "kernel_size must be an odd integer >= 3".to_string(),
                ));
            }
        }
        Ok(self)
    }

    /// Maps the visual flash-filter choice to canonical minimum-length policy.
    pub fn filter_mode(mut self, mode: FlashFilterMode, min_scene_len: u64) -> Self {
        self.filter_mode = mode;
        self.min_scene_len = min_scene_len;
        self
    }

    fn canonical_config(&self) -> scenedetect_core::DetectorConfig {
        scenedetect_core::DetectorConfig::Content(scenedetect_core::ContentDetectorConfig {
            threshold: self.threshold as f64,
            weights: self.weights.canonical(),
            luma_only: self.luma_only,
        })
    }

    fn canonical_options(&self) -> scenedetect_core::DetectionOptions {
        scenedetect_core::DetectionOptions {
            min_scene_len: self.min_scene_len,
            min_scene_len_policy: match self.filter_mode {
                FlashFilterMode::Merge => scenedetect_core::MinSceneLenPolicy::MergeLast,
                FlashFilterMode::Suppress => scenedetect_core::MinSceneLenPolicy::Suppress,
            },
        }
    }
}

impl SceneDetector for ContentDetector {
    fn name(&self) -> &'static str {
        "content"
    }

    fn metric_keys(&self) -> &'static [&'static str] {
        Self::METRIC_KEYS
    }

    fn process_frame(
        &mut self,
        frame: &VideoFrame<'_>,
        mut metrics: Option<&mut dyn MetricsSink>,
    ) -> Result<Vec<Cut>> {
        let rgb = (0..frame.height)
            .flat_map(|y| (0..frame.width).flat_map(move |x| frame.pixel_rgb(x, y)))
            .collect();
        self.frames.push(scenedetect_core::Frame {
            index: scenedetect_core::FrameIndex(frame.position.frame_index),
            width: frame.width,
            height: frame.height,
            rgb,
        });
        let timebase = frame.position.timestamp.timebase;
        let fps = if timebase.num == 0 {
            1.0
        } else {
            timebase.den as f64 / timebase.num as f64
        };
        let detected = scenedetect_core::detect_frames(
            self.canonical_config(),
            scenedetect_core::FrameRate(fps),
            &self.frames,
            self.canonical_options(),
        )
        .map_err(|error| DetectError::Source(error.to_string()))?;
        if let Some(sink) = metrics.as_mut() {
            if let Some(row) = detected.stats.rows.last() {
                for (key, value) in &row.metrics {
                    sink.set_metric(row.frame.0, key, *value);
                }
            }
        }
        Ok(detected
            .scene_list
            .scenes
            .into_iter()
            .map(|scene| scene.start.0)
            .filter(|start| *start != 0 && self.emitted.insert(*start))
            .map(|frame_index| Cut {
                position: position_like(frame.position, frame_index),
                detector: self.name(),
                score: None,
            })
            .collect())
    }
}

'''

DETECTORS_LIB = r'''#![doc = include_str!("../README.md")]

pub mod surface;

pub use scenedetect_core::{
    AdaptiveDetectorConfig as AdaptiveDetector, DetectorConfig,
    HashDetectorConfig as HashDetector, HistogramDetectorConfig as HistogramDetector,
    ThresholdDetectorConfig as ThresholdDetector,
};
pub use video_analysis_core::{ContentDetector, ContentWeights, FlashFilterMode};

/// Marks the registry-owned canonical detector seam used by this compatibility package.
pub const CANONICAL_SCENE_OWNER: &str = "scenedetect-core";
'''

DETECTORS_SURFACE = r'''//! Runtime metadata for the registry-backed detector compatibility package.

use runtime_core::{
    OperationId, PackageSurface, RuntimeCapabilities, SurfaceOperation, SurfaceRequest,
    SurfaceResponse,
};

/// Returns the package surface retained by the visual compatibility wrappers.
pub fn package_surface() -> PackageSurface {
    let operations = [
        ("describe", "Describe package"),
        ("video.detectors.registry", "Summarize canonical detector registry"),
        ("video.detectors.flashFilter", "Describe canonical minimum-length policy"),
        ("video.detectors.compositePlan", "Describe the visual composition adapter"),
    ]
    .into_iter()
    .map(|(id, name)| SurfaceOperation {
        id: OperationId::new(id),
        name: name.to_string(),
        description: Some("Compatibility metadata backed by scenedetect-core =0.1.0.".to_string()),
        curation: runtime_core::SurfaceOperationCuration::from_operation_id(id),
        input_schema: serde_json::json!({"type": "object", "additionalProperties": true}),
        output_schema: serde_json::json!({"type": "object"}),
        example_request: serde_json::json!({}),
        wasm_supported: true,
        server_supported: true,
    })
    .collect();
    PackageSurface {
        library: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities: RuntimeCapabilities::pure_rust(),
        operations,
    }
}

/// Runs a structural compatibility operation without owning detector algorithms.
pub fn run_surface_operation(request: SurfaceRequest) -> Result<SurfaceResponse, String> {
    if !package_surface()
        .operations
        .iter()
        .any(|operation| operation.id == request.operation)
    {
        return Err(format!("unsupported operation `{}`", request.operation));
    }
    Ok(SurfaceResponse {
        operation: request.operation,
        value: serde_json::json!({
            "canonicalOwner": "scenedetect-core",
            "canonicalVersion": "0.1.0",
            "input": request.input,
        }),
        diagnostics: Vec::new(),
        artifacts: Vec::new(),
    })
}
'''


def main() -> None:
    core_path = ROOT / "crates/video/video-analysis-core/src/lib.rs"
    core = core_path.read_text()
    pattern = re.compile(
        r"#\[derive\(Debug, Clone\)\]\nstruct FlashFilter \{.*?(?=/// Trait for video source implementations\.)",
        re.DOTALL,
    )
    core, replacements = pattern.subn(CORE_ADAPTER, core, count=1)
    if replacements != 1:
        raise SystemExit("expected exactly one canonical content-detector block")
    core_path.write_text(core)

    detector_root = ROOT / "crates/video/video-analysis-detectors/src"
    (detector_root / "lib.rs").write_text(DETECTORS_LIB)
    (detector_root / "surface.rs").write_text(DETECTORS_SURFACE)


if __name__ == "__main__":
    main()
