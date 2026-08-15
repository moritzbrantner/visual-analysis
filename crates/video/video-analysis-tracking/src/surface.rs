//! Library-owned runtime surface for `video-analysis-tracking`.

use std::collections::{BTreeMap, BTreeSet};

use runtime_core::{
    OperationId, PackageSurface, RuntimeCapabilities, SurfaceOperation, SurfaceRequest,
    SurfaceResponse,
};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use video_analysis_core::{BoundingBox, FramePosition, ObservationKind, Timebase, Timestamp};

use crate::{bbox_iou, IouTracker, TrackedDetection, TrackingOptions};

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
                "Tracking contracts for observations, tracks, paths, and assignment diagnostics.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "video.tracking.trackSummary",
                "Summarize tracks",
                "Runs IoU tracking over frame detections and summarizes resulting tracks.",
                serde_json::json!({
                    "options": {"minIou": 0.3, "maxMissedFrames": 15},
                    "frames": [
                        {"frameIndex": 0, "detections": [{"region": {"x": 0, "y": 0, "width": 8, "height": 8}, "label": "person"}]},
                        {"frameIndex": 1, "detections": [{"region": {"x": 1, "y": 0, "width": 8, "height": 8}, "label": "person"}]}
                    ]
                }),
            ),
            operation(
                "video.tracking.assignmentPlan",
                "Plan assignments",
                "Builds a deterministic IoU association plan between previous and next detections.",
                serde_json::json!({
                    "minIou": 0.3,
                    "previous": [{"region": {"x": 0, "y": 0, "width": 8, "height": 8}}],
                    "next": [{"region": {"x": 1, "y": 0, "width": 8, "height": 8}}]
                }),
            ),
            operation(
                "video.tracking.smoothPath",
                "Smooth track path",
                "Applies deterministic moving-average smoothing to ordered track points.",
                serde_json::json!({
                    "window": 3,
                    "points": [
                        {"frameIndex": 0, "x": 0.0, "y": 0.0},
                        {"frameIndex": 1, "x": 3.0, "y": 0.0}
                    ]
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
        "video.tracking.trackSummary" => track_summary_value(parse_input(request.input)?)?,
        "video.tracking.assignmentPlan" => assignment_plan_value(parse_input(request.input)?)?,
        "video.tracking.smoothPath" => smooth_path_value(parse_input(request.input)?)?,
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
struct TrackSummaryRequest {
    #[serde(default)]
    options: TrackingOptionsRequest,
    #[serde(default)]
    frames: Vec<FrameDetectionsRequest>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TrackingOptionsRequest {
    #[serde(default = "default_min_iou")]
    min_iou: f32,
    #[serde(default = "default_max_missed_frames")]
    max_missed_frames: u64,
    min_score: Option<f32>,
}

impl Default for TrackingOptionsRequest {
    fn default() -> Self {
        Self {
            min_iou: default_min_iou(),
            max_missed_frames: default_max_missed_frames(),
            min_score: None,
        }
    }
}

fn default_min_iou() -> f32 {
    0.3
}

fn default_max_missed_frames() -> u64 {
    15
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FrameDetectionsRequest {
    frame_index: u64,
    #[serde(default = "default_fps_num")]
    fps_num: i32,
    #[serde(default = "default_fps_den")]
    fps_den: i32,
    #[serde(default)]
    detections: Vec<DetectionRequest>,
}

fn default_fps_num() -> i32 {
    30
}

fn default_fps_den() -> i32 {
    1
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct DetectionRequest {
    region: BoundingBoxRequest,
    kind: Option<String>,
    label: Option<String>,
    score: Option<f32>,
    track_hint: Option<String>,
    #[serde(default)]
    attributes: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize, Clone, Copy)]
#[serde(rename_all = "camelCase")]
struct BoundingBoxRequest {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

fn track_summary_value(request: TrackSummaryRequest) -> Result<serde_json::Value, String> {
    let mut tracker = IouTracker::new(TrackingOptions {
        min_iou: request.options.min_iou,
        max_missed_frames: request.options.max_missed_frames,
        min_score: request.options.min_score,
    })
    .map_err(|error| error.to_string())?;
    let mut frame_summaries = Vec::new();

    for frame in request.frames {
        let position = frame_position(frame.frame_index, frame.fps_num, frame.fps_den)?;
        let detections = frame
            .detections
            .into_iter()
            .map(tracked_detection)
            .collect::<Result<Vec<_>, _>>()?;
        let visible = tracker
            .update(position, detections)
            .map_err(|error| error.to_string())?;
        frame_summaries.push(serde_json::json!({
            "frameIndex": frame.frame_index,
            "visibleTrackIds": visible.iter().map(|track| track.id.as_str()).collect::<Vec<_>>(),
            "visibleCount": visible.len()
        }));
    }

    let tracks = tracker
        .tracks()
        .map(|track| {
            serde_json::json!({
                "id": track.id,
                "kind": kind_name(&track.kind),
                "label": track.label,
                "score": track.score,
                "firstFrame": track.first_position.frame_index,
                "lastFrame": track.last_position.frame_index,
                "ageFrames": track.age_frames,
                "missedFrames": track.missed_frames,
                "region": bbox_json(track.region),
                "attributes": track.attributes
            })
        })
        .collect::<Vec<_>>();

    Ok(serde_json::json!({
        "frameCount": frame_summaries.len(),
        "activeTrackCount": tracks.len(),
        "frames": frame_summaries,
        "tracks": tracks,
        "options": {
            "minIou": request.options.min_iou,
            "maxMissedFrames": request.options.max_missed_frames,
            "minScore": request.options.min_score
        }
    }))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AssignmentPlanRequest {
    #[serde(default = "default_min_iou")]
    min_iou: f32,
    #[serde(default)]
    previous: Vec<DetectionRequest>,
    #[serde(default)]
    next: Vec<DetectionRequest>,
}

fn assignment_plan_value(request: AssignmentPlanRequest) -> Result<serde_json::Value, String> {
    if !request.min_iou.is_finite() || !(0.0..=1.0).contains(&request.min_iou) {
        return Err("minIou must be finite and between 0.0 and 1.0".to_string());
    }
    let previous = request
        .previous
        .into_iter()
        .map(|detection| bbox(detection.region))
        .collect::<Result<Vec<_>, _>>()?;
    let next = request
        .next
        .into_iter()
        .map(|detection| bbox(detection.region))
        .collect::<Result<Vec<_>, _>>()?;
    let mut matrix = Vec::new();
    let mut pairs = Vec::new();
    let mut used_previous = BTreeSet::new();
    let mut used_next = BTreeSet::new();

    for (next_index, next_box) in next.iter().enumerate() {
        let mut best = None::<(usize, f32)>;
        let mut row = Vec::new();
        for (previous_index, previous_box) in previous.iter().enumerate() {
            let iou = bbox_iou(*previous_box, *next_box);
            row.push(iou);
            if !used_previous.contains(&previous_index)
                && iou >= request.min_iou
                && best.map(|(_, best_iou)| iou > best_iou).unwrap_or(true)
            {
                best = Some((previous_index, iou));
            }
        }
        if let Some((previous_index, iou)) = best {
            used_previous.insert(previous_index);
            used_next.insert(next_index);
            pairs.push(serde_json::json!({
                "previousIndex": previous_index,
                "nextIndex": next_index,
                "iou": iou
            }));
        }
        matrix.push(row);
    }

    let unmatched_previous = (0..previous.len())
        .filter(|index| !used_previous.contains(index))
        .collect::<Vec<_>>();
    let unmatched_next = (0..next.len())
        .filter(|index| !used_next.contains(index))
        .collect::<Vec<_>>();

    Ok(serde_json::json!({
        "minIou": request.min_iou,
        "iouMatrix": matrix,
        "acceptedPairs": pairs,
        "unmatchedPrevious": unmatched_previous,
        "unmatchedNext": unmatched_next
    }))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SmoothPathRequest {
    #[serde(default = "default_window")]
    window: usize,
    #[serde(default)]
    points: Vec<PathPointRequest>,
}

fn default_window() -> usize {
    3
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PathPointRequest {
    frame_index: u64,
    x: f64,
    y: f64,
}

fn smooth_path_value(request: SmoothPathRequest) -> Result<serde_json::Value, String> {
    if request.window == 0 {
        return Err("window must be greater than zero".to_string());
    }
    if request
        .points
        .iter()
        .any(|point| !point.x.is_finite() || !point.y.is_finite())
    {
        return Err("path points must be finite".to_string());
    }
    let radius = request.window / 2;
    let mut smoothed = Vec::new();
    for index in 0..request.points.len() {
        let start = index.saturating_sub(radius);
        let end = (index + radius + 1).min(request.points.len());
        let count = (end - start) as f64;
        let x = request.points[start..end]
            .iter()
            .map(|point| point.x)
            .sum::<f64>()
            / count;
        let y = request.points[start..end]
            .iter()
            .map(|point| point.y)
            .sum::<f64>()
            / count;
        smoothed.push(serde_json::json!({
            "frameIndex": request.points[index].frame_index,
            "x": x,
            "y": y
        }));
    }

    Ok(serde_json::json!({
        "window": request.window,
        "originalCount": request.points.len(),
        "smoothedCount": smoothed.len(),
        "points": smoothed
    }))
}

fn tracked_detection(request: DetectionRequest) -> Result<TrackedDetection, String> {
    let mut detection = TrackedDetection::new(bbox(request.region)?)
        .kind(kind_from_request(request.kind.as_deref()));
    if let Some(label) = request.label {
        detection = detection.label(label);
    }
    if let Some(score) = request.score {
        if !score.is_finite() {
            return Err("detection score must be finite".to_string());
        }
        detection = detection.score(score);
    }
    if let Some(track_hint) = request.track_hint {
        detection = detection.track_hint(track_hint);
    }
    for (key, value) in request.attributes {
        detection = detection.attribute(key, value);
    }
    Ok(detection)
}

fn bbox(request: BoundingBoxRequest) -> Result<BoundingBox, String> {
    BoundingBox::new(request.x, request.y, request.width, request.height)
        .map_err(|error| error.to_string())
}

fn bbox_json(value: BoundingBox) -> serde_json::Value {
    serde_json::json!({
        "x": value.x,
        "y": value.y,
        "width": value.width,
        "height": value.height
    })
}

fn frame_position(frame_index: u64, fps_num: i32, fps_den: i32) -> Result<FramePosition, String> {
    if fps_num <= 0 || fps_den <= 0 {
        return Err("fps numerator and denominator must be positive".to_string());
    }
    Ok(FramePosition {
        frame_index,
        timestamp: Timestamp::new(frame_index as i64, Timebase::new(fps_den, fps_num)),
    })
}

fn kind_from_request(kind: Option<&str>) -> ObservationKind {
    match kind.unwrap_or("object") {
        "text" => ObservationKind::Text,
        "face" => ObservationKind::Face,
        "scene" => ObservationKind::Scene,
        "object" => ObservationKind::Object,
        other => ObservationKind::Custom(other.to_string()),
    }
}

fn kind_name(kind: &ObservationKind) -> String {
    match kind {
        ObservationKind::Text => "text".to_string(),
        ObservationKind::Face => "face".to_string(),
        ObservationKind::Object => "object".to_string(),
        ObservationKind::Scene => "scene".to_string(),
        ObservationKind::Custom(value) => value.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn track_summary_runs_iou_tracker() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.tracking.trackSummary"),
            input: serde_json::json!({
                "frames": [
                    {"frameIndex": 0, "detections": [{"region": {"x": 0, "y": 0, "width": 8, "height": 8}, "label": "person"}]},
                    {"frameIndex": 1, "detections": [{"region": {"x": 1, "y": 0, "width": 8, "height": 8}, "label": "person"}]}
                ]
            }),
        })
        .expect("track summary");
        assert_eq!(response.value["activeTrackCount"], 1);
        assert!(response.value.get("plan").is_none());
    }

    #[test]
    fn assignment_plan_reports_unmatched() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.tracking.assignmentPlan"),
            input: serde_json::json!({
                "minIou": 0.5,
                "previous": [{"region": {"x": 0, "y": 0, "width": 4, "height": 4}}],
                "next": [{"region": {"x": 20, "y": 20, "width": 4, "height": 4}}]
            }),
        })
        .expect("assignment plan");
        assert_eq!(response.value["acceptedPairs"].as_array().unwrap().len(), 0);
        assert_eq!(
            response.value["unmatchedPrevious"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn smooth_path_rejects_bad_window() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.tracking.smoothPath"),
            input: serde_json::json!({"window": 0, "points": []}),
        })
        .expect_err("bad window");
        assert!(error.contains("window must be greater than zero"));
    }
}
