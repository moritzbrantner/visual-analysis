//! Library-owned runtime surface for `video-analysis-segmentation`.

use image_analysis_segmentation::{
    BinaryMask, ImageSegmentationPrompt, PointLabel, SegmentationPoint,
};
use runtime_core::{
    describe_surface_response, parse_surface_input, structured_operation_response,
    surface_operation, OperationId, PackageSurface, RuntimeCapabilities, SurfaceError,
    SurfaceRequest, SurfaceResponse,
};
use serde::Deserialize;
use video_analysis_core::BoundingBox;

use crate::{
    plan_mask_tracks, plan_video_prompt, summarize_video_masks, MaskTrack, VideoMaskObservation,
    VideoMaskStats, VideoMaskSummaryRequest, VideoPromptPlan, VideoPromptPlanRequest,
    VideoSegmentationPrompt, VideoTrackPlan, VideoTrackPlanRequest,
};

/// Returns the package surface exposed by every transport wrapper.
pub fn package_surface() -> PackageSurface {
    PackageSurface {
        library: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities: RuntimeCapabilities::pure_rust(),
        operations: vec![
            surface_operation(
                "describe",
                "Describe package",
                "Video mask summaries, SAM-style prompt normalization, and IoU mask tracking.",
                serde_json::json!({"includeOperations": true}),
            ),
            surface_operation(
                "video.segmentation.maskSummary",
                "Summarize masks",
                "Computes mask area, coverage, bounds, frame counts, and confidence statistics.",
                serde_json::json!({
                    "masks": [
                        {
                            "frameIndex": 0,
                            "objectId": "person-1",
                            "label": "person",
                            "confidence": 0.92,
                            "width": 4,
                            "height": 4,
                            "rect": {"x": 1, "y": 1, "width": 2, "height": 2}
                        }
                    ],
                    "minMaskPixels": 1
                }),
            ),
            surface_operation(
                "video.segmentation.promptPlan",
                "Normalize prompt",
                "Normalizes boxes, points, labels, frame anchors, object IDs, and propagation controls for SAM-style video segmentation.",
                serde_json::json!({
                    "frameIndex": 12,
                    "objectId": "person-1",
                    "propagate": true,
                    "points": [
                        {"x": 24, "y": 32, "label": "foreground"},
                        {"x": 6, "y": 8, "label": "background"}
                    ],
                    "boxes": [{"x": 10, "y": 12, "width": 64, "height": 48}],
                    "multimaskOutput": true,
                    "minMaskPixels": 8
                }),
            ),
            surface_operation(
                "video.segmentation.trackPlan",
                "Associate masks",
                "Associates masks across frames with deterministic IoU matching and returns track observations.",
                serde_json::json!({
                    "iouThreshold": 0.5,
                    "masks": [
                        {"frameIndex": 0, "objectId": "person-1", "width": 8, "height": 8, "rect": {"x": 0, "y": 0, "width": 4, "height": 4}},
                        {"frameIndex": 1, "width": 8, "height": 8, "rect": {"x": 1, "y": 0, "width": 4, "height": 4}}
                    ]
                }),
            ),
        ],
    }
}

/// Runs one library-owned operation.
pub fn run_surface_operation(request: SurfaceRequest) -> Result<SurfaceResponse, String> {
    let surface = package_surface();
    let operation = request.operation.clone();
    let value = match request.operation.as_str() {
        "describe" => return Ok(describe_surface_response(&surface, request)),
        "video.segmentation.maskSummary" => mask_summary_value(
            operation.as_str(),
            parse_surface_input(Some(operation.as_str()), request.input)?,
        )?,
        "video.segmentation.promptPlan" => prompt_plan_value(
            operation.as_str(),
            parse_surface_input(Some(operation.as_str()), request.input)?,
        )?,
        "video.segmentation.trackPlan" => track_plan_value(
            operation.as_str(),
            parse_surface_input(Some(operation.as_str()), request.input)?,
        )?,
        operation => {
            return Err(
                SurfaceError::unsupported_operation(operation, env!("CARGO_PKG_NAME"))
                    .to_error_string(),
            );
        }
    };
    Ok(structured_operation_response(&surface, operation, value))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MaskSummaryInput {
    #[serde(default)]
    masks: Vec<MaskInput>,
    #[serde(default)]
    frame_index: u64,
    #[serde(default)]
    object_id: Option<String>,
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    confidence: Option<f32>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    #[serde(default)]
    data: Option<Vec<u8>>,
    #[serde(default)]
    rect: Option<BoxInput>,
    #[serde(default = "default_min_mask_pixels")]
    min_mask_pixels: usize,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TrackPlanInput {
    masks: Vec<MaskInput>,
    #[serde(default = "default_iou_threshold")]
    iou_threshold: f32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MaskInput {
    frame_index: u64,
    #[serde(default)]
    object_id: Option<String>,
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    confidence: Option<f32>,
    width: u32,
    height: u32,
    #[serde(default)]
    data: Option<Vec<u8>>,
    #[serde(default)]
    rect: Option<BoxInput>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PromptPlanInput {
    #[serde(default)]
    frame_index: u64,
    #[serde(default)]
    prompt: Option<PromptInput>,
    #[serde(default)]
    object_id: Option<String>,
    #[serde(default = "default_true")]
    propagate: bool,
    #[serde(default)]
    points: Vec<PointInput>,
    #[serde(default)]
    boxes: Vec<BoxInput>,
    #[serde(default)]
    automatic_mask_generation: bool,
    #[serde(default)]
    multimask_output: bool,
    #[serde(default = "default_min_mask_pixels")]
    min_mask_pixels: usize,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PromptInput {
    #[serde(default)]
    object_id: Option<String>,
    #[serde(default = "default_true")]
    propagate: bool,
    #[serde(default)]
    points: Vec<PointInput>,
    #[serde(default)]
    boxes: Vec<BoxInput>,
    #[serde(default)]
    automatic_mask_generation: bool,
    #[serde(default)]
    multimask_output: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PointInput {
    x: u32,
    y: u32,
    #[serde(default = "default_point_label")]
    label: String,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BoxInput {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

fn mask_summary_value(
    operation: &str,
    input: MaskSummaryInput,
) -> Result<serde_json::Value, String> {
    let masks = if input.masks.is_empty() {
        let width = input
            .width
            .ok_or_else(|| invalid_request(operation, "width is required for a single mask"))?;
        let height = input
            .height
            .ok_or_else(|| invalid_request(operation, "height is required for a single mask"))?;
        vec![MaskInput {
            frame_index: input.frame_index,
            object_id: input.object_id,
            label: input.label,
            confidence: input.confidence,
            width,
            height,
            data: input.data,
            rect: input.rect,
        }]
    } else {
        input.masks
    };
    let observations = masks
        .into_iter()
        .map(|mask| mask_observation(operation, mask))
        .collect::<Result<Vec<_>, _>>()?;
    let summary = summarize_video_masks(
        &VideoMaskSummaryRequest::new(observations).min_mask_pixels(input.min_mask_pixels),
    )
    .map_err(|error| invalid_request(operation, error.to_string()))?;

    Ok(serde_json::json!({
        "maskCount": summary.mask_count,
        "frameCount": summary.frame_count,
        "activePixels": summary.active_pixels,
        "totalPixels": summary.total_pixels,
        "coverage": summary.coverage,
        "confidence": summary.confidence.map(|confidence| serde_json::json!({
            "count": confidence.count,
            "min": confidence.min,
            "max": confidence.max,
            "mean": confidence.mean
        })),
        "masks": summary.masks.iter().map(mask_stats_value).collect::<Vec<_>>()
    }))
}

fn prompt_plan_value(operation: &str, input: PromptPlanInput) -> Result<serde_json::Value, String> {
    let prompt_input = input.prompt.unwrap_or(PromptInput {
        object_id: input.object_id,
        propagate: input.propagate,
        points: input.points,
        boxes: input.boxes,
        automatic_mask_generation: input.automatic_mask_generation,
        multimask_output: input.multimask_output,
    });
    let mut prompt = ImageSegmentationPrompt::new();
    prompt.automatic_mask_generation = prompt_input.automatic_mask_generation;
    prompt.multimask_output = prompt_input.multimask_output;
    for point in prompt_input.points {
        prompt = prompt.point(parse_point(operation, point)?);
    }
    for region in prompt_input.boxes {
        prompt = prompt.bounding_box(region.bounding_box(operation)?);
    }
    let mut video_prompt = VideoSegmentationPrompt::new(prompt).propagate(prompt_input.propagate);
    if let Some(object_id) = prompt_input.object_id {
        video_prompt = video_prompt.object_id(object_id);
    }
    let plan = plan_video_prompt(
        &VideoPromptPlanRequest::new(input.frame_index, video_prompt)
            .min_mask_pixels(input.min_mask_pixels),
    )
    .map_err(|error| invalid_request(operation, error.to_string()))?;

    Ok(prompt_plan_json(&plan))
}

fn track_plan_value(operation: &str, input: TrackPlanInput) -> Result<serde_json::Value, String> {
    let observations = input
        .masks
        .into_iter()
        .map(|mask| mask_observation(operation, mask))
        .collect::<Result<Vec<_>, _>>()?;
    let plan = plan_mask_tracks(
        &VideoTrackPlanRequest::new(observations).iou_threshold(input.iou_threshold),
    )
    .map_err(|error| invalid_request(operation, error.to_string()))?;

    Ok(track_plan_json(&plan))
}

fn mask_observation(operation: &str, input: MaskInput) -> Result<VideoMaskObservation, String> {
    let mask = if let Some(rect) = input.rect {
        BinaryMask::filled_rect(input.width, input.height, rect.bounding_box(operation)?)
    } else {
        BinaryMask::new(
            input.width,
            input.height,
            input
                .data
                .ok_or_else(|| invalid_request(operation, "mask data or rect is required"))?,
        )
    }
    .map_err(|error| invalid_request(operation, error.to_string()))?;
    let mut observation = VideoMaskObservation::new(input.frame_index, mask);
    if let Some(object_id) = input.object_id {
        observation = observation.object_id(object_id);
    }
    if let Some(label) = input.label {
        observation = observation.label(label);
    }
    if let Some(confidence) = input.confidence {
        observation = observation.confidence(confidence);
    }
    Ok(observation)
}

fn parse_point(operation: &str, input: PointInput) -> Result<SegmentationPoint, String> {
    match input.label.as_str() {
        "foreground" | "fg" | "positive" | "include" => {
            Ok(SegmentationPoint::foreground(input.x, input.y))
        }
        "background" | "bg" | "negative" | "exclude" => {
            Ok(SegmentationPoint::background(input.x, input.y))
        }
        other => Err(SurfaceError::unsupported_value(
            Some(OperationId::new(operation)),
            "label",
            other,
            &[
                "foreground",
                "background",
                "fg",
                "bg",
                "positive",
                "negative",
            ],
        )
        .to_error_string()),
    }
}

fn prompt_plan_json(plan: &VideoPromptPlan) -> serde_json::Value {
    serde_json::json!({
        "frameIndex": plan.frame_index,
        "objectId": plan.object_id,
        "propagate": plan.propagate,
        "automaticMaskGeneration": plan.automatic_mask_generation,
        "multimaskOutput": plan.multimask_output,
        "minMaskPixels": plan.min_mask_pixels,
        "promptModes": plan.prompt_modes,
        "pointCount": plan.points.len(),
        "foregroundPointCount": plan.points.iter().filter(|point| point.label == PointLabel::Foreground).count(),
        "backgroundPointCount": plan.points.iter().filter(|point| point.label == PointLabel::Background).count(),
        "boxCount": plan.boxes.len(),
        "points": plan.points.iter().map(|point| serde_json::json!({
            "x": point.x,
            "y": point.y,
            "label": point_label(point.label),
            "samLabel": point.sam_label
        })).collect::<Vec<_>>(),
        "boxes": plan.boxes.iter().map(|region| serde_json::json!({
            "x": region.region.x,
            "y": region.region.y,
            "width": region.region.width,
            "height": region.region.height,
            "xMin": region.x_min,
            "yMin": region.y_min,
            "xMax": region.x_max,
            "yMax": region.y_max
        })).collect::<Vec<_>>()
    })
}

fn track_plan_json(plan: &VideoTrackPlan) -> serde_json::Value {
    serde_json::json!({
        "iouThreshold": plan.iou_threshold,
        "maskCount": plan.mask_count,
        "trackCount": plan.track_count,
        "associations": plan.associations.iter().map(|association| serde_json::json!({
            "trackId": association.track_id,
            "maskIndex": association.mask_index,
            "frameIndex": association.frame_index,
            "iou": association.iou,
            "newTrack": association.new_track
        })).collect::<Vec<_>>(),
        "tracks": plan.tracks.iter().map(track_json).collect::<Vec<_>>()
    })
}

fn track_json(track: &MaskTrack) -> serde_json::Value {
    serde_json::json!({
        "trackId": track.track_id,
        "objectId": track.object_id,
        "observationCount": track.observations.len(),
        "observations": track.observations.iter().map(|observation| serde_json::json!({
            "frameIndex": observation.frame_index,
            "maskIndex": observation.mask_index,
            "activePixels": observation.active_pixels,
            "boundingBox": bounding_box_value(observation.bounding_box),
            "iouWithPrevious": observation.iou_with_previous
        })).collect::<Vec<_>>()
    })
}

fn mask_stats_value(stats: &VideoMaskStats) -> serde_json::Value {
    serde_json::json!({
        "maskIndex": stats.mask_index,
        "frameIndex": stats.frame_index,
        "objectId": stats.object_id,
        "label": stats.label,
        "confidence": stats.confidence,
        "width": stats.width,
        "height": stats.height,
        "activePixels": stats.active_pixels,
        "totalPixels": stats.total_pixels,
        "coverage": stats.coverage,
        "boundingBox": bounding_box_value(stats.bounding_box)
    })
}

fn bounding_box_value(region: BoundingBox) -> serde_json::Value {
    serde_json::json!({
        "x": region.x,
        "y": region.y,
        "width": region.width,
        "height": region.height
    })
}

impl BoxInput {
    fn bounding_box(self, operation: &str) -> Result<BoundingBox, String> {
        BoundingBox::new(self.x, self.y, self.width, self.height)
            .map_err(|error| invalid_request(operation, error.to_string()))
    }
}

fn point_label(label: PointLabel) -> &'static str {
    match label {
        PointLabel::Foreground => "foreground",
        PointLabel::Background => "background",
    }
}

fn invalid_request(operation: &str, message: impl Into<String>) -> String {
    SurfaceError::invalid_request(Some(OperationId::new(operation)), message).to_error_string()
}

fn default_min_mask_pixels() -> usize {
    1
}

fn default_iou_threshold() -> f32 {
    0.5
}

fn default_true() -> bool {
    true
}

fn default_point_label() -> String {
    "foreground".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_surface_lists_segmentation_operations() {
        let ids = package_surface()
            .operations
            .into_iter()
            .map(|operation| operation.id.0)
            .collect::<Vec<_>>();

        assert!(ids.contains(&"video.segmentation.maskSummary".to_string()));
        assert!(ids.contains(&"video.segmentation.promptPlan".to_string()));
        assert!(ids.contains(&"video.segmentation.trackPlan".to_string()));
    }

    #[test]
    fn mask_summary_returns_real_mask_metrics() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.segmentation.maskSummary"),
            input: serde_json::json!({
                "masks": [{
                    "frameIndex": 0,
                    "confidence": 0.8,
                    "width": 4,
                    "height": 4,
                    "rect": {"x": 1, "y": 1, "width": 2, "height": 2}
                }]
            }),
        })
        .expect("mask summary");

        assert_eq!(response.value["maskCount"], 1);
        assert_eq!(response.value["activePixels"], 4);
        assert_eq!(response.value["masks"][0]["boundingBox"]["x"], 1);
        assert!((response.value["confidence"]["mean"].as_f64().unwrap() - 0.8).abs() < 1.0e-6);
        assert!(response.value.get("operationKind").is_none());
    }

    #[test]
    fn prompt_plan_normalizes_point_labels() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.segmentation.promptPlan"),
            input: serde_json::json!({
                "frameIndex": 2,
                "objectId": "subject",
                "points": [
                    {"x": 1, "y": 2, "label": "foreground"},
                    {"x": 3, "y": 4, "label": "negative"}
                ],
                "boxes": [{"x": 0, "y": 0, "width": 4, "height": 5}]
            }),
        })
        .expect("prompt plan");

        assert_eq!(response.value["frameIndex"], 2);
        assert_eq!(response.value["objectId"], "subject");
        assert_eq!(response.value["pointCount"], 2);
        assert_eq!(response.value["points"][0]["samLabel"], 1);
        assert_eq!(response.value["points"][1]["samLabel"], 0);
        assert_eq!(response.value["boxes"][0]["xMax"], 4);
        assert!(response.value.get("operationKind").is_none());
    }

    #[test]
    fn track_plan_associates_adjacent_frame_masks() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.segmentation.trackPlan"),
            input: serde_json::json!({
                "iouThreshold": 0.5,
                "masks": [
                    {"frameIndex": 0, "objectId": "subject", "width": 8, "height": 8, "rect": {"x": 0, "y": 0, "width": 4, "height": 4}},
                    {"frameIndex": 1, "width": 8, "height": 8, "rect": {"x": 1, "y": 0, "width": 4, "height": 4}}
                ]
            }),
        })
        .expect("track plan");

        assert_eq!(response.value["trackCount"], 1);
        assert_eq!(response.value["tracks"][0]["observationCount"], 2);
        assert_eq!(response.value["associations"][1]["newTrack"], false);
        assert!(response.value["associations"][1]["iou"].as_f64().unwrap() > 0.5);
        assert!(response.value.get("operationKind").is_none());
    }
}
