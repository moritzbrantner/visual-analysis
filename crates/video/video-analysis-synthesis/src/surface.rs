//! Library-owned runtime surface for `video-analysis-synthesis`.

use std::collections::BTreeMap;

use num_rational::Rational64;
use runtime_core::{
    OperationId, PackageSurface, RuntimeCapabilities, SurfaceOperation, SurfaceRequest,
    SurfaceResponse,
};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use video_analysis_core::{
    BoundingBox, FramePosition, Observation, ObservationKind, PixelFormat, Timebase, Timestamp,
};

use crate::{
    frame_from_spec, storyboard_from_observations, FrameSynthesisSpec, RgbColor, SynthesizedRegion,
    VideoSynthesisConfig,
};

const MAX_PIXEL_PREVIEW: usize = 64;

/// Returns the package surface exposed by every transport wrapper.
pub fn package_surface() -> PackageSurface {
    PackageSurface {
        library: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities: RuntimeCapabilities::pure_rust(),
        operations: vec![
            operation(
                "describe",
                "Describe package",
                "Synthetic video and rendered frame planning contracts.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "video.synthesis.framePlan",
                "Plan synthetic frames",
                "Generates one deterministic synthetic frame and returns capped pixel preview metadata.",
                serde_json::json!({
                    "frameIndex": 0,
                    "config": {"width": 16, "height": 16, "fpsNum": 30, "fpsDen": 1},
                    "background": {"red": 16, "green": 18, "blue": 20},
                    "regions": [{"region": {"x": 2, "y": 2, "width": 4, "height": 4}, "color": {"red": 255, "green": 0, "blue": 0}}],
                    "previewLimit": 8
                }),
            ),
            operation(
                "video.synthesis.overlayPlan",
                "Plan overlays",
                "Builds deterministic storyboard frames from observations and reports inferred overlay regions.",
                serde_json::json!({
                    "config": {"width": 16, "height": 16, "fpsNum": 30, "fpsDen": 1},
                    "observations": [{"analyzer": "fixture", "kind": "object", "label": "person", "frameIndex": 0, "region": {"x": 1, "y": 1, "width": 4, "height": 4}}]
                }),
            ),
            operation(
                "video.synthesis.renderSummary",
                "Summarize render",
                "Summarizes generated frame dimensions, pixel counts, and capped preview pixels.",
                serde_json::json!({
                    "frame": {"frameIndex": 0, "background": {"red": 16, "green": 18, "blue": 20}},
                    "config": {"width": 16, "height": 16, "fpsNum": 30, "fpsDen": 1},
                    "previewLimit": 8
                }),
            ),
        ],
    }
}

fn operation(
    id: &str,
    name: &str,
    description: &str,
    example_request: serde_json::Value,
) -> SurfaceOperation {
    SurfaceOperation {
        id: OperationId::new(id),
        name: name.to_string(),
        description: Some(description.to_string()),
        curation: runtime_core::SurfaceOperationCuration::from_operation_id(id),
        input_schema: serde_json::json!({"type": "object", "additionalProperties": true, "xOperationCategory": runtime_core::operation_category(id)}),
        output_schema: serde_json::json!({"type": "object", "xOperationCategory": runtime_core::operation_category(id)}),
        example_request,
        wasm_supported: true,
        server_supported: true,
    }
}

/// Runs one library-owned operation.
pub fn run_surface_operation(request: SurfaceRequest) -> Result<SurfaceResponse, String> {
    let operation = request.operation.clone();
    let value = match request.operation.as_str() {
        "describe" => describe_value(request.input),
        "video.synthesis.framePlan" => frame_plan_value(parse_input(request.input)?)?,
        "video.synthesis.overlayPlan" => overlay_plan_value(parse_input(request.input)?)?,
        "video.synthesis.renderSummary" => render_summary_value(parse_input(request.input)?)?,
        operation => {
            return Err(format!(
                "unsupported operation `{operation}` for {}",
                env!("CARGO_PKG_NAME")
            ))
        }
    };
    Ok(surface_response(operation, value))
}

fn describe_value(input: serde_json::Value) -> serde_json::Value {
    let surface = package_surface();
    serde_json::json!({
        "library": surface.library,
        "version": surface.version,
        "operationCount": surface.operations.len(),
        "operations": surface.operations.iter().map(|operation| operation.id.as_str()).collect::<Vec<_>>(),
        "input": input
    })
}

fn surface_response(operation: OperationId, value: serde_json::Value) -> SurfaceResponse {
    SurfaceResponse {
        operation,
        value,
        diagnostics: Vec::new(),
        artifacts: Vec::new(),
    }
}

fn parse_input<T: DeserializeOwned>(input: serde_json::Value) -> Result<T, String> {
    serde_json::from_value(input).map_err(|error| format!("invalid request: {error}"))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FramePlanRequest {
    #[serde(default)]
    frame_index: u64,
    #[serde(default)]
    config: ConfigRequest,
    background: Option<ColorRequest>,
    #[serde(default)]
    regions: Vec<RegionRequest>,
    preview_limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RenderSummaryRequest {
    frame: Option<FramePlanRequest>,
    #[serde(default)]
    config: ConfigRequest,
    #[serde(default)]
    observations: Vec<ObservationRequest>,
    preview_limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct OverlayPlanRequest {
    #[serde(default)]
    config: ConfigRequest,
    #[serde(default)]
    observations: Vec<ObservationRequest>,
}

#[derive(Debug, Deserialize, Clone, Copy)]
#[serde(rename_all = "camelCase")]
struct ConfigRequest {
    #[serde(default = "default_width")]
    width: u32,
    #[serde(default = "default_height")]
    height: u32,
    #[serde(default = "default_fps_num")]
    fps_num: i64,
    #[serde(default = "default_fps_den")]
    fps_den: i64,
    background: Option<ColorRequest>,
}

impl Default for ConfigRequest {
    fn default() -> Self {
        Self {
            width: default_width(),
            height: default_height(),
            fps_num: default_fps_num(),
            fps_den: default_fps_den(),
            background: None,
        }
    }
}

fn default_width() -> u32 {
    640
}

fn default_height() -> u32 {
    360
}

fn default_fps_num() -> i64 {
    30
}

fn default_fps_den() -> i64 {
    1
}

#[derive(Debug, Deserialize, Clone, Copy)]
#[serde(rename_all = "camelCase")]
struct ColorRequest {
    red: u8,
    green: u8,
    blue: u8,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RegionRequest {
    region: BoundingBoxRequest,
    color: ColorRequest,
}

#[derive(Debug, Deserialize, Clone, Copy)]
#[serde(rename_all = "camelCase")]
struct BoundingBoxRequest {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ObservationRequest {
    #[serde(default = "default_analyzer")]
    analyzer: String,
    #[serde(default = "default_kind")]
    kind: String,
    label: Option<String>,
    text: Option<String>,
    score: Option<f32>,
    frame_index: Option<u64>,
    seconds: Option<f64>,
    region: Option<BoundingBoxRequest>,
    track_id: Option<String>,
    #[serde(default)]
    attributes: BTreeMap<String, String>,
}

fn default_analyzer() -> String {
    "surface".to_string()
}

fn default_kind() -> String {
    "object".to_string()
}

fn frame_plan_value(request: FramePlanRequest) -> Result<serde_json::Value, String> {
    let config = build_config(request.config, request.background)?;
    let spec = frame_spec(&request)?;
    let generated = frame_from_spec(&spec, config).map_err(|error| error.to_string())?;
    Ok(frame_json(
        &generated.value,
        preview_limit(request.preview_limit),
    ))
}

fn overlay_plan_value(request: OverlayPlanRequest) -> Result<serde_json::Value, String> {
    let config = build_config(request.config, None)?;
    let observations = request
        .observations
        .into_iter()
        .map(|observation| observation_value(observation, config.fps))
        .collect::<Result<Vec<_>, _>>()?;
    let explicit_regions = observations
        .iter()
        .filter(|observation| observation.region.is_some())
        .count();
    let generated =
        storyboard_from_observations(&observations, config).map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "frameCount": generated.value.len(),
        "frameIndexes": generated.value.iter().map(|frame| frame.position.frame_index).collect::<Vec<_>>(),
        "observationCount": observations.len(),
        "explicitRegionCount": explicit_regions,
        "inferredRegionCount": observations.len().saturating_sub(explicit_regions),
        "dimensions": {"width": config.width, "height": config.height}
    }))
}

fn render_summary_value(request: RenderSummaryRequest) -> Result<serde_json::Value, String> {
    let limit = preview_limit(request.preview_limit);
    let config = build_config(request.config, None)?;
    let frames = if !request.observations.is_empty() {
        let observations = request
            .observations
            .into_iter()
            .map(|observation| observation_value(observation, config.fps))
            .collect::<Result<Vec<_>, _>>()?;
        storyboard_from_observations(&observations, config)
            .map_err(|error| error.to_string())?
            .value
    } else {
        let frame_request = request.frame.unwrap_or(FramePlanRequest {
            frame_index: 0,
            config: request.config,
            background: None,
            regions: Vec::new(),
            preview_limit: Some(limit),
        });
        let frame_config = build_config(frame_request.config, frame_request.background)?;
        vec![
            frame_from_spec(&frame_spec(&frame_request)?, frame_config)
                .map_err(|error| error.to_string())?
                .value,
        ]
    };
    let first = frames
        .first()
        .ok_or_else(|| "render summary requires at least one generated frame".to_string())?;
    Ok(serde_json::json!({
        "frameCount": frames.len(),
        "frameIndexes": frames.iter().map(|frame| frame.position.frame_index).collect::<Vec<_>>(),
        "dimensions": {"width": first.width, "height": first.height},
        "pixelFormat": pixel_format_name(first.pixel_format),
        "totalPixelCount": frames.iter().map(|frame| frame.width as usize * frame.height as usize).sum::<usize>(),
        "previewLimit": limit,
        "previewPixels": pixel_preview(&first.data, limit)
    }))
}

fn frame_spec(request: &FramePlanRequest) -> Result<FrameSynthesisSpec, String> {
    let background = request
        .background
        .map(color)
        .or(request.config.background.map(color))
        .unwrap_or(RgbColor::new(16, 18, 20));
    let mut spec = FrameSynthesisSpec::new(request.frame_index, background);
    for region in &request.regions {
        spec = spec.region(SynthesizedRegion {
            region: bbox(region.region)?,
            color: color(region.color),
        });
    }
    Ok(spec)
}

fn build_config(
    request: ConfigRequest,
    background: Option<ColorRequest>,
) -> Result<VideoSynthesisConfig, String> {
    let mut config = VideoSynthesisConfig::new(
        request.width,
        request.height,
        Rational64::new(request.fps_num, request.fps_den),
    )
    .map_err(|error| error.to_string())?;
    if let Some(background) = background.or(request.background) {
        config = config.background(color(background));
    }
    Ok(config)
}

fn observation_value(request: ObservationRequest, fps: Rational64) -> Result<Observation, String> {
    let mut observation = Observation::new(request.analyzer, kind(&request.kind));
    if let Some(frame_index) = request.frame_index {
        observation = observation.at_frame(FramePosition::from_frame_index(frame_index, fps));
    } else if let Some(seconds) = request.seconds {
        if !seconds.is_finite() || seconds < 0.0 {
            return Err("observation seconds must be finite and non-negative".to_string());
        }
        let den = *fps.numer() as i32;
        let num = *fps.denom() as i32;
        observation = observation.at_timestamp(Timestamp::new(
            (seconds * f64::from(den) / f64::from(num)).round() as i64,
            Timebase::new(num, den),
        ));
    }
    if let Some(label) = request.label {
        observation = observation.label(label);
    }
    if let Some(text) = request.text {
        observation = observation.text(text);
    }
    if let Some(score) = request.score {
        if !score.is_finite() {
            return Err("observation score must be finite".to_string());
        }
        observation = observation.score(score);
    }
    if let Some(region) = request.region {
        observation = observation.region(bbox(region)?);
    }
    if let Some(track_id) = request.track_id {
        observation = observation.track_id(track_id);
    }
    for (key, value) in request.attributes {
        observation = observation.attribute(key, value);
    }
    Ok(observation)
}

fn frame_json(frame: &video_analysis_core::OwnedVideoFrame, limit: usize) -> serde_json::Value {
    serde_json::json!({
        "frameIndex": frame.position.frame_index,
        "timestampSeconds": frame.position.timestamp.seconds(),
        "width": frame.width,
        "height": frame.height,
        "stride": frame.stride,
        "pixelFormat": pixel_format_name(frame.pixel_format),
        "pixelCount": frame.width as usize * frame.height as usize,
        "previewLimit": limit,
        "previewPixels": pixel_preview(&frame.data, limit)
    })
}

fn pixel_preview(data: &[u8], limit: usize) -> Vec<[u8; 3]> {
    data.chunks_exact(3)
        .take(limit)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect()
}

fn preview_limit(value: Option<usize>) -> usize {
    value
        .filter(|limit| *limit > 0)
        .unwrap_or(MAX_PIXEL_PREVIEW)
        .min(MAX_PIXEL_PREVIEW)
}

fn bbox(request: BoundingBoxRequest) -> Result<BoundingBox, String> {
    BoundingBox::new(request.x, request.y, request.width, request.height)
        .map_err(|error| error.to_string())
}

fn color(request: ColorRequest) -> RgbColor {
    RgbColor::new(request.red, request.green, request.blue)
}

fn kind(value: &str) -> ObservationKind {
    match value {
        "text" => ObservationKind::Text,
        "face" => ObservationKind::Face,
        "scene" => ObservationKind::Scene,
        "object" => ObservationKind::Object,
        other => ObservationKind::Custom(other.to_string()),
    }
}

fn pixel_format_name(value: PixelFormat) -> &'static str {
    match value {
        PixelFormat::Rgb24 => "rgb24",
        PixelFormat::Bgr24 => "bgr24",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_plan_returns_generated_frame_preview() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.synthesis.framePlan"),
            input: serde_json::json!({
                "frameIndex": 2,
                "config": {"width": 8, "height": 8},
                "regions": [{"region": {"x": 1, "y": 1, "width": 3, "height": 3}, "color": {"red": 255, "green": 0, "blue": 0}}],
                "previewLimit": 4
            }),
        })
        .expect("frame plan");
        assert_eq!(response.value["frameIndex"], 2);
        assert_eq!(response.value["previewPixels"].as_array().unwrap().len(), 4);
        assert!(response.value.get("plan").is_none());
    }

    #[test]
    fn overlay_plan_reports_inferred_regions() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.synthesis.overlayPlan"),
            input: serde_json::json!({
                "config": {"width": 8, "height": 8},
                "observations": [{"frameIndex": 0, "label": "marker"}]
            }),
        })
        .expect("overlay plan");
        assert_eq!(response.value["frameCount"], 1);
        assert_eq!(response.value["inferredRegionCount"], 1);
    }

    #[test]
    fn render_summary_rejects_bad_dimensions() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.synthesis.renderSummary"),
            input: serde_json::json!({"config": {"width": 0, "height": 8}}),
        })
        .expect_err("bad dimensions");
        assert!(error.contains("invalid dimensions"));
    }
}
