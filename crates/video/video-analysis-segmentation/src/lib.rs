#![doc = include_str!("../README.md")]

pub mod surface;
use std::collections::{BTreeMap, BTreeSet};

use image_analysis_segmentation::{BinaryMask, ImageSegment, ImageSegmentationPrompt, PointLabel};
use model_runtime::{HuggingFaceModelSpec, ModelTask};
use video_analysis_core::{BoundingBox, DetectError, FramePosition, Result, VideoFrame};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Variants describing SAM video preset.
pub enum SamVideoPreset {
    #[default]
    /// The sam2 1 hiera large variant.
    Sam2_1HieraLarge,
}

impl SamVideoPreset {
    /// Constant for all.
    pub const ALL: &'static [Self] = &[Self::Sam2_1HieraLarge];

    /// Borrows this value as a str.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sam2_1HieraLarge => "sam2.1-hiera-large",
        }
    }

    /// Returns repo identifier.
    pub fn repo_id(self) -> &'static str {
        match self {
            Self::Sam2_1HieraLarge => "facebook/sam2.1-hiera-large",
        }
    }

    /// Returns model spec.
    pub fn model_spec(self) -> HuggingFaceModelSpec {
        HuggingFaceModelSpec::new(
            self.repo_id(),
            ModelTask::Custom("video_segmentation".to_string()),
        )
        .name(self.as_str())
        .file("config.json")
        .file("preprocessor_config.json")
        .file("processor_config.json")
        .file("video_preprocessor_config.json")
        .file("sam2.1_hiera_l.yaml")
        .first_available_file(["model.safetensors", "sam2.1_hiera_large.pt"])
    }
}

/// Returns default sam2 model spec.
pub fn default_sam2_model_spec() -> HuggingFaceModelSpec {
    SamVideoPreset::default().model_spec()
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Data type for video segmentation prompt.
pub struct VideoSegmentationPrompt {
    /// The prompt value.
    pub prompt: ImageSegmentationPrompt,
    /// The object identifier value.
    pub object_id: Option<String>,
    /// The propagate value.
    pub propagate: bool,
}

impl VideoSegmentationPrompt {
    /// Creates a new value.
    pub fn new(prompt: ImageSegmentationPrompt) -> Self {
        Self {
            prompt,
            object_id: None,
            propagate: true,
        }
    }

    /// Returns object identifier.
    pub fn object_id(mut self, value: impl Into<String>) -> Self {
        self.object_id = Some(value.into());
        self
    }

    /// Returns propagate.
    pub fn propagate(mut self, value: bool) -> Self {
        self.propagate = value;
        self
    }
}

impl Default for VideoSegmentationPrompt {
    fn default() -> Self {
        Self::new(ImageSegmentationPrompt::automatic_mask_generation())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
/// Data type for video segmentation request.
pub struct VideoSegmentationRequest {
    /// The prompt value.
    pub prompt: VideoSegmentationPrompt,
    /// The min mask pixels value.
    pub min_mask_pixels: usize,
}

impl VideoSegmentationRequest {
    /// Returns min mask pixels.
    pub fn min_mask_pixels(mut self, value: usize) -> Self {
        self.min_mask_pixels = value.max(1);
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Input mask with video frame metadata for deterministic segmentation summaries and tracking.
pub struct VideoMaskObservation {
    /// Zero-based frame index associated with this mask.
    pub frame_index: u64,
    /// Optional object identifier supplied by a prompt or upstream tracker.
    pub object_id: Option<String>,
    /// Optional semantic label.
    pub label: Option<String>,
    /// Optional confidence score.
    pub confidence: Option<f32>,
    /// Binary mask payload.
    pub mask: BinaryMask,
}

impl VideoMaskObservation {
    /// Creates a new mask observation for a frame.
    pub fn new(frame_index: u64, mask: BinaryMask) -> Self {
        Self {
            frame_index,
            object_id: None,
            label: None,
            confidence: None,
            mask,
        }
    }

    /// Returns object identifier.
    pub fn object_id(mut self, value: impl Into<String>) -> Self {
        self.object_id = Some(value.into());
        self
    }

    /// Returns label.
    pub fn label(mut self, value: impl Into<String>) -> Self {
        self.label = Some(value.into());
        self
    }

    /// Returns confidence.
    pub fn confidence(mut self, value: f32) -> Self {
        self.confidence = Some(value);
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Request for summarizing a set of video masks.
pub struct VideoMaskSummaryRequest {
    /// Mask observations to summarize.
    pub masks: Vec<VideoMaskObservation>,
    /// Minimum active pixels required for each mask.
    pub min_mask_pixels: usize,
}

impl VideoMaskSummaryRequest {
    /// Creates a new request and defaults to requiring at least one active pixel.
    pub fn new(masks: impl Into<Vec<VideoMaskObservation>>) -> Self {
        Self {
            masks: masks.into(),
            min_mask_pixels: 1,
        }
    }

    /// Returns min mask pixels.
    pub fn min_mask_pixels(mut self, value: usize) -> Self {
        self.min_mask_pixels = value.max(1);
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Per-mask summary derived from a binary video mask.
pub struct VideoMaskStats {
    /// Index of the input mask.
    pub mask_index: usize,
    /// Frame index associated with the mask.
    pub frame_index: u64,
    /// Optional object identifier.
    pub object_id: Option<String>,
    /// Optional semantic label.
    pub label: Option<String>,
    /// Optional confidence score.
    pub confidence: Option<f32>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Active mask area in pixels.
    pub active_pixels: usize,
    /// Total mask area in pixels.
    pub total_pixels: usize,
    /// Active-pixel coverage in the range `0.0..=1.0`.
    pub coverage: f32,
    /// Tight bounds around active pixels.
    pub bounding_box: BoundingBox,
}

#[derive(Debug, Clone, PartialEq)]
/// Aggregate confidence summary for masks that include scores.
pub struct ConfidenceSummary {
    /// Number of masks with a confidence value.
    pub count: usize,
    /// Minimum confidence.
    pub min: f32,
    /// Maximum confidence.
    pub max: f32,
    /// Mean confidence.
    pub mean: f32,
}

#[derive(Debug, Clone, PartialEq)]
/// Summary of mask areas, coverage, bounds, and confidence values.
pub struct VideoMaskSummary {
    /// Number of accepted masks.
    pub mask_count: usize,
    /// Number of distinct frames represented by accepted masks.
    pub frame_count: usize,
    /// Sum of active pixels across masks.
    pub active_pixels: usize,
    /// Sum of total pixels across masks.
    pub total_pixels: usize,
    /// Aggregate active-pixel coverage.
    pub coverage: f32,
    /// Aggregate confidence statistics when any mask has confidence.
    pub confidence: Option<ConfidenceSummary>,
    /// Per-mask summaries.
    pub masks: Vec<VideoMaskStats>,
}

/// Summarizes video masks with area, coverage, bounds, and confidence statistics.
pub fn summarize_video_masks(request: &VideoMaskSummaryRequest) -> Result<VideoMaskSummary> {
    if request.masks.is_empty() {
        return Err(invalid_argument("mask summary requires at least one mask"));
    }

    let mut stats = Vec::with_capacity(request.masks.len());
    let mut frames = BTreeSet::new();
    let mut active_pixels = 0_usize;
    let mut total_pixels = 0_usize;
    let mut confidences = Vec::new();

    for (mask_index, observation) in request.masks.iter().enumerate() {
        if let Some(confidence) = observation.confidence {
            if !confidence.is_finite() {
                return Err(invalid_argument("mask confidence must be finite"));
            }
        }
        let active = observation.mask.active_pixels();
        if active < request.min_mask_pixels {
            return Err(invalid_argument(format!(
                "mask {mask_index} has {active} active pixels, below min_mask_pixels {}",
                request.min_mask_pixels
            )));
        }
        let bounding_box = observation.mask.bounding_box().ok_or_else(|| {
            invalid_argument(format!("mask {mask_index} must contain active pixels"))
        })?;
        let total = observation.mask.width as usize * observation.mask.height as usize;
        active_pixels += active;
        total_pixels += total;
        frames.insert(observation.frame_index);
        if let Some(confidence) = observation.confidence {
            confidences.push(confidence);
        }
        stats.push(VideoMaskStats {
            mask_index,
            frame_index: observation.frame_index,
            object_id: observation.object_id.clone(),
            label: observation.label.clone(),
            confidence: observation.confidence,
            width: observation.mask.width,
            height: observation.mask.height,
            active_pixels: active,
            total_pixels: total,
            coverage: active as f32 / total as f32,
            bounding_box,
        });
    }

    let confidence = if confidences.is_empty() {
        None
    } else {
        let min = confidences.iter().copied().fold(f32::INFINITY, f32::min);
        let max = confidences
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mean = confidences.iter().sum::<f32>() / confidences.len() as f32;
        Some(ConfidenceSummary {
            count: confidences.len(),
            min,
            max,
            mean,
        })
    };

    Ok(VideoMaskSummary {
        mask_count: stats.len(),
        frame_count: frames.len(),
        active_pixels,
        total_pixels,
        coverage: active_pixels as f32 / total_pixels as f32,
        confidence,
        masks: stats,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Request for normalizing a SAM-style video segmentation prompt.
pub struct VideoPromptPlanRequest {
    /// Frame to which the prompt is anchored.
    pub frame_index: u64,
    /// Prompt payload.
    pub prompt: VideoSegmentationPrompt,
    /// Minimum accepted mask area in downstream segmentation.
    pub min_mask_pixels: usize,
}

impl VideoPromptPlanRequest {
    /// Creates a new prompt plan request for a frame.
    pub fn new(frame_index: u64, prompt: VideoSegmentationPrompt) -> Self {
        Self {
            frame_index,
            prompt,
            min_mask_pixels: 1,
        }
    }

    /// Returns min mask pixels.
    pub fn min_mask_pixels(mut self, value: usize) -> Self {
        self.min_mask_pixels = value.max(1);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Normalized point prompt with SAM label encoding.
pub struct NormalizedPromptPoint {
    /// X coordinate in pixels.
    pub x: u32,
    /// Y coordinate in pixels.
    pub y: u32,
    /// Semantic point label.
    pub label: PointLabel,
    /// SAM point label value: foreground is `1`, background is `0`.
    pub sam_label: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Normalized box prompt with inclusive-exclusive edges.
pub struct NormalizedPromptBox {
    /// Original bounding box.
    pub region: BoundingBox,
    /// Left edge in pixels.
    pub x_min: u32,
    /// Top edge in pixels.
    pub y_min: u32,
    /// Right edge in pixels.
    pub x_max: u32,
    /// Bottom edge in pixels.
    pub y_max: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// SAM-style prompt normalization plan.
pub struct VideoPromptPlan {
    /// Frame to which the prompt is anchored.
    pub frame_index: u64,
    /// Optional object identifier.
    pub object_id: Option<String>,
    /// Whether masks should propagate to later frames.
    pub propagate: bool,
    /// Automatic mask generation flag.
    pub automatic_mask_generation: bool,
    /// Multimask output flag.
    pub multimask_output: bool,
    /// Minimum accepted mask area.
    pub min_mask_pixels: usize,
    /// Prompt modes represented by the request.
    pub prompt_modes: Vec<String>,
    /// Normalized points.
    pub points: Vec<NormalizedPromptPoint>,
    /// Normalized boxes.
    pub boxes: Vec<NormalizedPromptBox>,
}

/// Normalizes boxes, points, labels, and propagation controls into a SAM-style prompt plan.
pub fn plan_video_prompt(request: &VideoPromptPlanRequest) -> Result<VideoPromptPlan> {
    let prompt = &request.prompt.prompt;
    if !prompt.automatic_mask_generation && prompt.points.is_empty() && prompt.boxes.is_empty() {
        return Err(invalid_argument(
            "prompt must include points, boxes, or automatic_mask_generation",
        ));
    }
    let mut modes = Vec::new();
    if prompt.automatic_mask_generation {
        modes.push("automatic".to_string());
    }
    if !prompt.points.is_empty() {
        modes.push("points".to_string());
    }
    if !prompt.boxes.is_empty() {
        modes.push("boxes".to_string());
    }

    let points = prompt
        .points
        .iter()
        .map(|point| NormalizedPromptPoint {
            x: point.x,
            y: point.y,
            label: point.label,
            sam_label: match point.label {
                PointLabel::Foreground => 1,
                PointLabel::Background => 0,
            },
        })
        .collect::<Vec<_>>();
    let boxes = prompt
        .boxes
        .iter()
        .map(|region| {
            let x_max = region
                .x
                .checked_add(region.width)
                .ok_or_else(|| invalid_argument("prompt box x + width overflows"))?;
            let y_max = region
                .y
                .checked_add(region.height)
                .ok_or_else(|| invalid_argument("prompt box y + height overflows"))?;
            Ok(NormalizedPromptBox {
                region: *region,
                x_min: region.x,
                y_min: region.y,
                x_max,
                y_max,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(VideoPromptPlan {
        frame_index: request.frame_index,
        object_id: request.prompt.object_id.clone(),
        propagate: request.prompt.propagate,
        automatic_mask_generation: prompt.automatic_mask_generation,
        multimask_output: prompt.multimask_output,
        min_mask_pixels: request.min_mask_pixels.max(1),
        prompt_modes: modes,
        points,
        boxes,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Options for deterministic IoU-based mask association.
pub struct VideoTrackPlanOptions {
    /// Minimum IoU required to associate a mask with an existing track.
    pub iou_threshold: f32,
}

impl Default for VideoTrackPlanOptions {
    fn default() -> Self {
        Self { iou_threshold: 0.5 }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Request for IoU-based mask association across frames.
pub struct VideoTrackPlanRequest {
    /// Mask observations to associate.
    pub masks: Vec<VideoMaskObservation>,
    /// Tracking options.
    pub options: VideoTrackPlanOptions,
}

impl VideoTrackPlanRequest {
    /// Creates a new track-plan request.
    pub fn new(masks: impl Into<Vec<VideoMaskObservation>>) -> Self {
        Self {
            masks: masks.into(),
            options: VideoTrackPlanOptions::default(),
        }
    }

    /// Returns iou threshold.
    pub fn iou_threshold(mut self, value: f32) -> Self {
        self.options.iou_threshold = value;
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
/// A single mask observation assigned to a track.
pub struct TrackObservation {
    /// Frame index.
    pub frame_index: u64,
    /// Index of the input mask.
    pub mask_index: usize,
    /// Active mask area in pixels.
    pub active_pixels: usize,
    /// Tight bounds around active pixels.
    pub bounding_box: BoundingBox,
    /// IoU against the previous track mask; `None` for the first observation.
    pub iou_with_previous: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
/// Deterministic mask track.
pub struct MaskTrack {
    /// Stable track identifier.
    pub track_id: String,
    /// Optional object identifier inherited from observations.
    pub object_id: Option<String>,
    /// Ordered observations in this track.
    pub observations: Vec<TrackObservation>,
}

#[derive(Debug, Clone, PartialEq)]
/// Association decision made for one input mask.
pub struct MaskAssociation {
    /// Assigned track identifier.
    pub track_id: String,
    /// Input mask index.
    pub mask_index: usize,
    /// Frame index.
    pub frame_index: u64,
    /// IoU used for an existing-track association.
    pub iou: Option<f32>,
    /// Whether this mask started a new track.
    pub new_track: bool,
}

#[derive(Debug, Clone, PartialEq)]
/// IoU tracking plan response.
pub struct VideoTrackPlan {
    /// IoU threshold used by association.
    pub iou_threshold: f32,
    /// Number of input masks.
    pub mask_count: usize,
    /// Number of output tracks.
    pub track_count: usize,
    /// Per-mask association decisions.
    pub associations: Vec<MaskAssociation>,
    /// Output tracks.
    pub tracks: Vec<MaskTrack>,
}

/// Builds deterministic tracks by associating masks to previous-frame tracks by IoU.
pub fn plan_mask_tracks(request: &VideoTrackPlanRequest) -> Result<VideoTrackPlan> {
    let threshold = request.options.iou_threshold;
    if !(0.0..=1.0).contains(&threshold) || !threshold.is_finite() {
        return Err(invalid_argument(
            "iou_threshold must be finite and within 0..=1",
        ));
    }
    if request.masks.is_empty() {
        return Err(invalid_argument("track plan requires at least one mask"));
    }

    let mut indexed = request
        .masks
        .iter()
        .enumerate()
        .map(|(index, mask)| {
            let bounding_box = mask.mask.bounding_box().ok_or_else(|| {
                invalid_argument(format!("mask {index} must contain active pixels"))
            })?;
            Ok(IndexedMask {
                input_index: index,
                frame_index: mask.frame_index,
                object_id: mask.object_id.clone(),
                active_pixels: mask.mask.active_pixels(),
                bounding_box,
                mask: &mask.mask,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    indexed.sort_by_key(|mask| (mask.frame_index, mask.input_index));

    let mut tracks = Vec::<InternalTrack>::new();
    let mut associations = Vec::with_capacity(indexed.len());
    let mut used_tracks_by_frame = BTreeMap::<u64, BTreeSet<usize>>::new();

    for mask in indexed {
        let used_tracks = used_tracks_by_frame.entry(mask.frame_index).or_default();
        let mut best_track = None;
        let mut best_iou = 0.0_f32;
        for (track_index, track) in tracks.iter().enumerate() {
            if used_tracks.contains(&track_index) {
                continue;
            }
            if track.last_frame_index >= mask.frame_index {
                continue;
            }
            let iou = mask_iou(track.last_mask, mask.mask);
            if iou > best_iou {
                best_iou = iou;
                best_track = Some(track_index);
            }
        }

        if let Some(track_index) = best_track.filter(|_| best_iou >= threshold) {
            let track = &mut tracks[track_index];
            let track_id = track.track_id.clone();
            track.last_frame_index = mask.frame_index;
            track.last_mask = mask.mask;
            if track.object_id.is_none() {
                track.object_id = mask.object_id.clone();
            }
            track.observations.push(TrackObservation {
                frame_index: mask.frame_index,
                mask_index: mask.input_index,
                active_pixels: mask.active_pixels,
                bounding_box: mask.bounding_box,
                iou_with_previous: Some(best_iou),
            });
            used_tracks.insert(track_index);
            associations.push(MaskAssociation {
                track_id,
                mask_index: mask.input_index,
                frame_index: mask.frame_index,
                iou: Some(best_iou),
                new_track: false,
            });
        } else {
            let track_id = mask
                .object_id
                .clone()
                .unwrap_or_else(|| format!("track-{}", tracks.len() + 1));
            let track_index = tracks.len();
            tracks.push(InternalTrack {
                track_id: track_id.clone(),
                object_id: mask.object_id.clone(),
                last_frame_index: mask.frame_index,
                last_mask: mask.mask,
                observations: vec![TrackObservation {
                    frame_index: mask.frame_index,
                    mask_index: mask.input_index,
                    active_pixels: mask.active_pixels,
                    bounding_box: mask.bounding_box,
                    iou_with_previous: None,
                }],
            });
            used_tracks.insert(track_index);
            associations.push(MaskAssociation {
                track_id,
                mask_index: mask.input_index,
                frame_index: mask.frame_index,
                iou: None,
                new_track: true,
            });
        }
    }

    associations.sort_by_key(|association| association.mask_index);
    let tracks = tracks
        .into_iter()
        .map(|track| MaskTrack {
            track_id: track.track_id,
            object_id: track.object_id,
            observations: track.observations,
        })
        .collect::<Vec<_>>();

    Ok(VideoTrackPlan {
        iou_threshold: threshold,
        mask_count: request.masks.len(),
        track_count: tracks.len(),
        associations,
        tracks,
    })
}

struct IndexedMask<'a> {
    input_index: usize,
    frame_index: u64,
    object_id: Option<String>,
    active_pixels: usize,
    bounding_box: BoundingBox,
    mask: &'a BinaryMask,
}

struct InternalTrack<'a> {
    track_id: String,
    object_id: Option<String>,
    last_frame_index: u64,
    last_mask: &'a BinaryMask,
    observations: Vec<TrackObservation>,
}

/// Computes binary-mask IoU, falling back to bounding-box IoU for different dimensions.
pub fn mask_iou(left: &BinaryMask, right: &BinaryMask) -> f32 {
    if left.width == right.width && left.height == right.height {
        let mut intersection = 0_usize;
        let mut union = 0_usize;
        for (left, right) in left.data.iter().zip(&right.data) {
            let left_active = *left != 0;
            let right_active = *right != 0;
            if left_active && right_active {
                intersection += 1;
            }
            if left_active || right_active {
                union += 1;
            }
        }
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    } else {
        match (left.bounding_box(), right.bounding_box()) {
            (Some(left), Some(right)) => bounding_box_iou(left, right),
            _ => 0.0,
        }
    }
}

fn bounding_box_iou(left: BoundingBox, right: BoundingBox) -> f32 {
    let left_x2 = left.x.saturating_add(left.width);
    let left_y2 = left.y.saturating_add(left.height);
    let right_x2 = right.x.saturating_add(right.width);
    let right_y2 = right.y.saturating_add(right.height);
    let intersection_width = left_x2.min(right_x2).saturating_sub(left.x.max(right.x));
    let intersection_height = left_y2.min(right_y2).saturating_sub(left.y.max(right.y));
    let intersection = intersection_width as usize * intersection_height as usize;
    let left_area = left.width as usize * left.height as usize;
    let right_area = right.width as usize * right.height as usize;
    let union = left_area + right_area - intersection;
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for tracked segment.
pub struct TrackedSegment {
    /// The object identifier value.
    pub object_id: Option<String>,
    /// The segment value.
    pub segment: ImageSegment,
}

impl TrackedSegment {
    /// Creates a new value.
    pub fn new(segment: ImageSegment) -> Self {
        Self {
            object_id: None,
            segment,
        }
    }

    /// Returns object identifier.
    pub fn object_id(mut self, value: impl Into<String>) -> Self {
        self.object_id = Some(value.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for frame segmentation.
pub struct FrameSegmentation {
    /// The position value.
    pub position: FramePosition,
    /// The segments value.
    pub segments: Vec<TrackedSegment>,
}

/// Trait for video segmentation backend implementations.
pub trait VideoSegmentationBackend {
    /// Returns model spec.
    fn model_spec(&self) -> HuggingFaceModelSpec {
        default_sam2_model_spec()
    }

    /// Returns segment frame.
    fn segment_frame(
        &mut self,
        frame: &VideoFrame<'_>,
        request: &VideoSegmentationRequest,
    ) -> Result<Vec<TrackedSegment>>;
}

fn invalid_argument(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(message.into())
}

/// Data type for video segmenter.
pub struct VideoSegmenter<B> {
    backend: B,
    request: VideoSegmentationRequest,
}

impl<B> VideoSegmenter<B> {
    /// Creates a new value.
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            request: VideoSegmentationRequest::default(),
        }
    }

    /// Returns request.
    pub fn request(mut self, value: VideoSegmentationRequest) -> Self {
        self.request = value;
        self
    }

    /// Returns backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Returns backend mut.
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }
}

impl<B: VideoSegmentationBackend> VideoSegmenter<B> {
    /// Returns model spec.
    pub fn model_spec(&self) -> HuggingFaceModelSpec {
        self.backend.model_spec()
    }

    /// Returns process frame.
    pub fn process_frame(&mut self, frame: &VideoFrame<'_>) -> Result<FrameSegmentation> {
        let segments = self.backend.segment_frame(frame, &self.request)?;
        Ok(FrameSegmentation {
            position: frame.position,
            segments,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image_analysis_segmentation::{BinaryMask, ImageSegment, SegmentationPoint};
    use video_analysis_core::{BoundingBox, PixelFormat, Timebase, Timestamp};

    struct StubVideoBackend;

    impl VideoSegmentationBackend for StubVideoBackend {
        fn segment_frame(
            &mut self,
            frame: &VideoFrame<'_>,
            request: &VideoSegmentationRequest,
        ) -> Result<Vec<TrackedSegment>> {
            let id = request
                .prompt
                .object_id
                .clone()
                .unwrap_or_else(|| format!("frame-{}", frame.position.frame_index));
            Ok(vec![TrackedSegment::new(
                ImageSegment::new(
                    BinaryMask::filled_rect(
                        frame.width,
                        frame.height,
                        BoundingBox::new(1, 1, 2, 2).unwrap(),
                    )
                    .unwrap(),
                )
                .unwrap(),
            )
            .object_id(id)])
        }
    }

    fn frame<'a>(data: &'a [u8]) -> VideoFrame<'a> {
        VideoFrame::packed(
            FramePosition {
                frame_index: 7,
                timestamp: Timestamp::new(7, Timebase::new(1, 1)),
            },
            4,
            4,
            PixelFormat::Rgb24,
            data,
            4 * 3,
        )
        .unwrap()
    }

    #[test]
    fn default_video_spec_uses_sam2_1() {
        assert_eq!(
            default_sam2_model_spec().repo_id_value(),
            Some("facebook/sam2.1-hiera-large")
        );
    }

    #[test]
    fn video_segmenter_preserves_frame_position() {
        let bytes = vec![0_u8; 4 * 4 * 3];
        let mut segmenter =
            VideoSegmenter::new(StubVideoBackend).request(VideoSegmentationRequest {
                prompt: VideoSegmentationPrompt::default().object_id("car"),
                min_mask_pixels: 1,
            });
        let frame = frame(&bytes);
        let output = segmenter.process_frame(&frame).unwrap();
        assert_eq!(output.position.frame_index, 7);
        assert_eq!(output.segments[0].object_id.as_deref(), Some("car"));
    }

    #[test]
    fn mask_summary_reports_area_coverage_bounds_and_confidence() {
        let mask = BinaryMask::filled_rect(4, 4, BoundingBox::new(1, 1, 2, 2).unwrap()).unwrap();
        let summary =
            summarize_video_masks(&VideoMaskSummaryRequest::new([VideoMaskObservation::new(
                3, mask,
            )
            .object_id("car")
            .label("vehicle")
            .confidence(0.75)]))
            .unwrap();

        assert_eq!(summary.mask_count, 1);
        assert_eq!(summary.frame_count, 1);
        assert_eq!(summary.active_pixels, 4);
        assert_eq!(summary.total_pixels, 16);
        assert_eq!(
            summary.masks[0].bounding_box,
            BoundingBox::new(1, 1, 2, 2).unwrap()
        );
        assert!((summary.coverage - 0.25).abs() < f32::EPSILON);
        assert_eq!(summary.confidence.unwrap().mean, 0.75);
    }

    #[test]
    fn prompt_plan_normalizes_sam_points_boxes_and_labels() {
        let prompt = ImageSegmentationPrompt::new()
            .point(SegmentationPoint::foreground(1, 2))
            .point(SegmentationPoint::background(3, 4))
            .bounding_box(BoundingBox::new(0, 1, 5, 6).unwrap())
            .multimask_output(true);
        let plan = plan_video_prompt(
            &VideoPromptPlanRequest::new(
                9,
                VideoSegmentationPrompt::new(prompt)
                    .object_id("person")
                    .propagate(false),
            )
            .min_mask_pixels(4),
        )
        .unwrap();

        assert_eq!(plan.frame_index, 9);
        assert_eq!(plan.object_id.as_deref(), Some("person"));
        assert!(!plan.propagate);
        assert_eq!(plan.points[0].sam_label, 1);
        assert_eq!(plan.points[1].sam_label, 0);
        assert_eq!(plan.boxes[0].x_max, 5);
        assert_eq!(plan.boxes[0].y_max, 7);
        assert_eq!(plan.min_mask_pixels, 4);
        assert_eq!(plan.prompt_modes, vec!["points", "boxes"]);
    }

    #[test]
    fn track_plan_associates_masks_by_iou_across_frames() {
        let masks = vec![
            VideoMaskObservation::new(
                0,
                BinaryMask::filled_rect(8, 8, BoundingBox::new(0, 0, 4, 4).unwrap()).unwrap(),
            )
            .object_id("subject"),
            VideoMaskObservation::new(
                1,
                BinaryMask::filled_rect(8, 8, BoundingBox::new(1, 0, 4, 4).unwrap()).unwrap(),
            ),
            VideoMaskObservation::new(
                1,
                BinaryMask::filled_rect(8, 8, BoundingBox::new(6, 6, 2, 2).unwrap()).unwrap(),
            ),
        ];
        let plan = plan_mask_tracks(&VideoTrackPlanRequest::new(masks).iou_threshold(0.5)).unwrap();

        assert_eq!(plan.mask_count, 3);
        assert_eq!(plan.track_count, 2);
        assert!(!plan.associations[1].new_track);
        assert!(plan.associations[1].iou.unwrap() > 0.5);
        assert_eq!(plan.tracks[0].observations.len(), 2);
        assert_eq!(plan.tracks[0].object_id.as_deref(), Some("subject"));
    }
}
