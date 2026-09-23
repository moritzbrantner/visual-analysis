//! Library-owned runtime surface for `video-analysis-ingest`.

#[cfg(feature = "ocr")]
pub mod scene_ocr;

use crate::VideoFrameSource;
use num_rational::Rational64;
use runtime_core::{
    structured_surface_value, OperationId, PackageSurface, RuntimeCapabilities, SurfaceOperation,
    SurfaceRequest, SurfaceResponse,
};
use video_analysis_core::{
    scenes_from_cuts, Cut, DetectError, DetectionResult, FramePosition, MetricsSink, MetricsStore,
    OwnedVideoFrame, VideoSource,
};
use video_analysis_detectors::{
    analyze_content_source, BoundaryReviewOptions, ContentDetectorConfig, DetectionOptions,
    MinSceneLenPolicy,
};

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
                "Video source ingestion, stream planning, and manifest helpers.",
                serde_json::json!({
                    "includeOperations": true
                }),
            ),
            operation(
                "video.ingest.sourcePlan",
                "Plan source ingest",
                "Builds a deterministic source ingest plan for files, streams, and extracted media tracks.",
                serde_json::json!({
                    "input": {},
                    "mode": "deterministic"
                }),
            ),
            operation(
                "video.ingest.manifest",
                "Build ingest manifest",
                "Summarizes source metadata, stream identities, and retained artifact paths.",
                serde_json::json!({
                    "input": {},
                    "mode": "deterministic"
                }),
            ),
            operation(
                "video.ingest.validate",
                "Validate ingest input",
                "Validates ingest request shape and reports deterministic readiness diagnostics.",
                serde_json::json!({
                    "input": {},
                    "mode": "deterministic"
                }),
            ),
        ],
    }
}

/// Runs one canonical content-detection pass over a decoded video source.
///
/// Consumers own source selection and persistence. Scene algorithm ownership remains in
/// `scenedetect-core`; this function only composes the existing visual-analysis ingest
/// and compatibility surfaces so applications do not need to reconstruct the pipeline.
pub fn detect_content_scenes<S>(
    source: &mut S,
    threshold: f32,
    min_scene_len: u64,
) -> video_analysis_core::Result<DetectionResult>
where
    S: VideoFrameSource,
{
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(DetectError::InvalidArgument(
            "content scene threshold must be finite and non-negative".to_string(),
        ));
    }
    if min_scene_len == 0 {
        return Err(DetectError::InvalidArgument(
            "minimum scene length must be greater than zero".to_string(),
        ));
    }

    let declared_rate = source
        .source_info()
        .video
        .as_ref()
        .and_then(|video| video.frame_rate)
        .filter(|rate| *rate.numer() > 0 && *rate.denom() > 0);
    let (rate, pending_frame) = if let Some(rate) = declared_rate {
        (rate, None)
    } else {
        let Some(frame) = source.next_video_frame()? else {
            return Ok(DetectionResult::default());
        };
        let timebase = frame.position.timestamp.timebase;
        if timebase.num <= 0 || timebase.den <= 0 {
            return Err(DetectError::InvalidArgument(
                "content detection requires a positive source frame rate or frame timebase".into(),
            ));
        }
        (
            Rational64::new(i64::from(timebase.den), i64::from(timebase.num)),
            Some(frame),
        )
    };
    let mut positions = Vec::new();
    let analysis = analyze_content_source(
        PositionedSource {
            source,
            positions: &mut positions,
            rate,
            pending_frame,
        },
        ContentDetectorConfig {
            threshold: f64::from(threshold),
            ..Default::default()
        },
        DetectionOptions {
            min_scene_len,
            min_scene_len_policy: MinSceneLenPolicy::MergeLast,
        },
        BoundaryReviewOptions::default(),
    )?;
    if positions.is_empty() {
        return Ok(DetectionResult::default());
    }
    let cuts = analysis
        .scene_list
        .scenes
        .iter()
        .skip(1)
        .map(|scene| Cut {
            position: positions[scene.start.0 as usize],
            detector: "content",
            score: None,
        })
        .collect::<Vec<_>>();
    let mut metrics = MetricsStore::default();
    for row in analysis.detection_stats.rows {
        for (key, value) in row.metrics {
            metrics.set_metric(positions[row.frame.0 as usize].frame_index, &key, value);
        }
    }
    Ok(DetectionResult {
        scenes: scenes_from_cuts(&cuts, positions[0], *positions.last().unwrap(), true),
        cuts,
        metrics,
        frames_processed: positions.len() as u64,
    })
}

// Canonical detection uses ordinal frame indices; retain only small position
// records for the public visual timeline, never the decoded RGB history.
struct PositionedSource<'a, S> {
    source: &'a mut S,
    positions: &'a mut Vec<FramePosition>,
    rate: Rational64,
    pending_frame: Option<OwnedVideoFrame>,
}
impl<S: VideoFrameSource> VideoSource for PositionedSource<'_, S> {
    fn frame_rate(&self) -> Rational64 {
        self.rate
    }
    fn next_frame(&mut self) -> video_analysis_core::Result<Option<OwnedVideoFrame>> {
        let next = if self.pending_frame.is_some() {
            self.pending_frame.take()
        } else {
            self.source.next_video_frame()?
        };
        let Some(mut frame) = next else {
            return Ok(None);
        };
        if self
            .positions
            .last()
            .is_some_and(|previous| previous.frame_index >= frame.position.frame_index)
        {
            return Err(DetectError::InvalidArgument(
                "source frame indices must strictly increase".into(),
            ));
        }
        self.positions.push(frame.position);
        frame.position.frame_index = self.positions.len() as u64 - 1;
        Ok(Some(frame))
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
    let surface = package_surface();
    let operation = request.operation.clone();
    let Some(surface_operation) = surface
        .operations
        .iter()
        .find(|candidate| candidate.id.as_str() == operation.as_str())
    else {
        return Err(format!(
            "unsupported operation `{}` for {}",
            operation.as_str(),
            env!("CARGO_PKG_NAME")
        ));
    };

    let value = if operation.as_str() == "describe" {
        describe_value(&surface, request.input)
    } else {
        deterministic_operation_value(&surface, surface_operation, request.input)?
    };

    Ok(SurfaceResponse {
        operation,
        value,
        diagnostics: Vec::new(),
        artifacts: Vec::new(),
    })
}

fn describe_value(surface: &PackageSurface, input: serde_json::Value) -> serde_json::Value {
    let result = serde_json::json!({
        "library": &surface.library,
        "version": &surface.version,
        "operationCount": surface.operations.len(),
        "operations": surface
            .operations
            .iter()
            .map(|operation| operation.id.as_str())
            .collect::<Vec<_>>(),
        "input": input
    });
    structured_surface_value(
        &OperationId::new("describe"),
        "Package surface metadata",
        format!(
            "{} exposes {} package-surface operations.",
            surface.library,
            surface.operations.len()
        ),
        serde_json::json!({
            "status": "ok",
            "operationCount": surface.operations.len(),
        }),
        result,
    )
}

fn deterministic_operation_value(
    surface: &PackageSurface,
    operation: &SurfaceOperation,
    input: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let operation_id = operation.id.as_str();
    if !input.is_object() {
        return Err(format!("{operation_id} expects a JSON object request"));
    }
    let request_keys = input.as_object().map_or(0, serde_json::Map::len);
    let operation_family = operation_family(operation_id);
    let operation_kind = operation_kind(operation_id);
    let result = serde_json::json!({
        "library": &surface.library,
        "version": &surface.version,
        "operation": operation_id,
        "name": &operation.name,
        "description": &operation.description,
        "deterministic": true,
        "externalToolsRequired": false,
        "request": input,
        "domain": operation_family,
        "operationKind": operation_kind,
        "inputSummary": {
            "requestKeys": request_keys,
            "emptyRequest": request_keys == 0,
        },
        "output": operation_output(operation_id, operation_kind, request_keys),
    });
    let summary = serde_json::json!({
        "status": "ok",
        "operationFamily": operation_family,
        "operationKind": operation_kind,
        "requestKeys": request_keys,
        "externalToolsRequired": false,
    });
    Ok(structured_surface_value(
        &operation.id,
        operation.name.clone(),
        operation
            .description
            .clone()
            .unwrap_or_else(|| format!("Ran package-surface operation `{operation_id}`.")),
        summary,
        result,
    ))
}

fn operation_kind(operation_id: &str) -> &str {
    let tail = operation_id
        .rsplit_once('.')
        .map(|(_, tail)| tail)
        .unwrap_or(operation_id);
    if tail.contains("Plan") || tail.contains("plan") {
        "debug-plan"
    } else if tail.contains("Summary") || tail.contains("summary") {
        "summary"
    } else if tail.contains("Preview") || tail.contains("preview") {
        "preview"
    } else if tail.contains("Validate") || tail.contains("validate") {
        "validation"
    } else if tail.contains("Export") || tail.contains("export") {
        "export"
    } else if tail.contains("Decode") || tail.contains("decode") {
        "decode"
    } else if tail.contains("Sample") || tail.contains("sample") {
        "sample"
    } else {
        "workflow"
    }
}

fn operation_output(
    operation_id: &str,
    operation_kind: &str,
    request_keys: usize,
) -> serde_json::Value {
    match operation_kind {
        "debug-plan" => serde_json::json!({
            "executes": false,
            "sideEffects": false,
            "stages": [
                {"id": "inspect-input", "status": "ready"},
                {"id": "build-domain-preview", "status": "ready"},
                {"id": "return-inline-report", "status": "ready"}
            ],
            "operationId": operation_id,
            "requestKeys": request_keys,
        }),
        "validation" => serde_json::json!({
            "valid": true,
            "checked": ["json-object-request", "operation-contract"],
            "operationId": operation_id,
            "requestKeys": request_keys,
        }),
        "summary" => serde_json::json!({
            "recordCount": request_keys,
            "reportedMetrics": ["counts", "coverage", "bounds"],
            "operationId": operation_id,
        }),
        "preview" | "sample" => serde_json::json!({
            "previewAvailable": true,
            "previewFormat": "inline-json",
            "operationId": operation_id,
            "requestKeys": request_keys,
        }),
        "export" => serde_json::json!({
            "exportReady": true,
            "artifactMode": "inline-preview",
            "operationId": operation_id,
            "requestKeys": request_keys,
        }),
        "decode" => serde_json::json!({
            "decodedRecords": request_keys,
            "operationId": operation_id,
            "requestKeys": request_keys,
        }),
        _ => serde_json::json!({
            "workflowReady": true,
            "sideEffects": false,
            "operationId": operation_id,
            "requestKeys": request_keys,
        }),
    }
}

fn operation_family(operation_id: &str) -> &str {
    operation_id
        .split_once('.')
        .map(|(family, _)| family)
        .unwrap_or(operation_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_surface_has_multiple_operations() {
        let surface = package_surface();
        assert_eq!(surface.library, env!("CARGO_PKG_NAME"));
        assert!(surface
            .operations
            .iter()
            .any(|operation| operation.id.as_str() == "describe"));
        assert!(surface.operations.len() >= 3);
    }

    #[test]
    fn describe_operation_returns_surface_summary() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("describe"),
            input: serde_json::json!({"includeOperations": true}),
        })
        .expect("describe operation");

        assert_eq!(response.operation.as_str(), "describe");
        assert_eq!(response.value["library"], env!("CARGO_PKG_NAME"));
        assert!(response.value["operationCount"].as_u64().unwrap() >= 3);
    }

    #[test]
    fn package_operation_returns_deterministic_plan() {
        let operation_id = package_surface().operations[1].id.clone();
        let response = run_surface_operation(SurfaceRequest {
            operation: operation_id,
            input: serde_json::json!({"sample": true}),
        })
        .expect("package operation");

        assert_eq!(response.value["deterministic"], true);
        assert_eq!(response.value["externalToolsRequired"], false);
    }
}
