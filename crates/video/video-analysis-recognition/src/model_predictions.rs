use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use video_analysis_core::{AnalysisEvent, BoundingBox, Observation, ObservationKind};

use model_runtime::ModelTask;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
/// Data type for raw prediction.
pub struct RawPrediction {
    /// The kind value.
    pub kind: Option<String>,
    /// Label assigned to this value.
    pub label: Option<String>,
    /// Text content for this value.
    pub text: Option<String>,
    /// Score assigned to this value.
    pub score: Option<f32>,
    /// The region value.
    pub region: Option<RawBoundingBox>,
    #[serde(default)]
    /// The attributes value.
    pub attributes: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for raw keypoint2d.
pub struct RawKeypoint2d {
    /// Human-readable name for this value.
    pub name: String,
    /// The x value.
    pub x: f32,
    /// The y value.
    pub y: f32,
    /// Score assigned to this value.
    pub score: Option<f32>,
    /// The visible value.
    pub visible: Option<bool>,
}

impl RawKeypoint2d {
    /// Creates a new value.
    pub fn new(name: impl Into<String>, x: f32, y: f32) -> Self {
        Self {
            name: name.into(),
            x,
            y,
            score: None,
            visible: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for raw keypoint3d.
pub struct RawKeypoint3d {
    /// Human-readable name for this value.
    pub name: String,
    /// The x value.
    pub x: f32,
    /// The y value.
    pub y: f32,
    /// The z value.
    pub z: f32,
    /// Score assigned to this value.
    pub score: Option<f32>,
    /// The visible value.
    pub visible: Option<bool>,
}

impl RawKeypoint3d {
    /// Creates a new value.
    pub fn new(name: impl Into<String>, x: f32, y: f32, z: f32) -> Self {
        Self {
            name: name.into(),
            x,
            y,
            z,
            score: None,
            visible: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
/// Data type for raw pose2d prediction.
pub struct RawPose2dPrediction {
    /// Identifier for this value.
    pub id: Option<String>,
    /// Label assigned to this value.
    pub label: Option<String>,
    /// Score assigned to this value.
    pub score: Option<f32>,
    /// The region value.
    pub region: Option<RawBoundingBox>,
    #[serde(default)]
    /// The keypoints value.
    pub keypoints: Vec<RawKeypoint2d>,
    #[serde(default)]
    /// The attributes value.
    pub attributes: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
/// Data type for raw pose3d prediction.
pub struct RawPose3dPrediction {
    /// Identifier for this value.
    pub id: Option<String>,
    /// Label assigned to this value.
    pub label: Option<String>,
    /// Score assigned to this value.
    pub score: Option<f32>,
    #[serde(default)]
    /// The keypoints value.
    pub keypoints: Vec<RawKeypoint3d>,
    #[serde(default)]
    /// The attributes value.
    pub attributes: BTreeMap<String, String>,
}

impl RawPrediction {
    /// Returns object.
    pub fn object(label: impl Into<String>, score: f32, region: RawBoundingBox) -> Self {
        Self {
            kind: Some("object".to_string()),
            label: Some(label.into()),
            score: Some(score),
            region: Some(region),
            ..Self::default()
        }
    }

    /// Returns label.
    pub fn label(label: impl Into<String>, score: f32) -> Self {
        Self {
            label: Some(label.into()),
            score: Some(score),
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
/// Data type for raw bounding box.
pub struct RawBoundingBox {
    /// The x value.
    pub x: Option<f32>,
    /// The y value.
    pub y: Option<f32>,
    /// Width in pixels.
    pub width: Option<f32>,
    /// Height in pixels.
    pub height: Option<f32>,
    /// The xmin value.
    pub xmin: Option<f32>,
    /// The ymin value.
    pub ymin: Option<f32>,
    /// The xmax value.
    pub xmax: Option<f32>,
    /// The ymax value.
    pub ymax: Option<f32>,
    #[serde(default)]
    /// The normalized value.
    pub normalized: bool,
}

impl RawBoundingBox {
    /// Returns xywh.
    pub fn xywh(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x: Some(x),
            y: Some(y),
            width: Some(width),
            height: Some(height),
            ..Self::default()
        }
    }

    /// Returns xyxy.
    pub fn xyxy(xmin: f32, ymin: f32, xmax: f32, ymax: f32) -> Self {
        Self {
            xmin: Some(xmin),
            ymin: Some(ymin),
            xmax: Some(xmax),
            ymax: Some(ymax),
            ..Self::default()
        }
    }

    fn to_bounding_box(self, dimensions: Option<(u32, u32)>, clamp: bool) -> Option<BoundingBox> {
        let (mut x0, mut y0, mut x1, mut y1) =
            if let (Some(x), Some(y), Some(width), Some(height)) =
                (self.x, self.y, self.width, self.height)
            {
                (x, y, x + width, y + height)
            } else if let (Some(xmin), Some(ymin), Some(xmax), Some(ymax)) =
                (self.xmin, self.ymin, self.xmax, self.ymax)
            {
                (xmin, ymin, xmax, ymax)
            } else {
                return None;
            };

        if self.normalized
            || [x0, y0, x1, y1]
                .into_iter()
                .all(|value| (0.0..=1.0).contains(&value))
        {
            let (width, height) = dimensions?;
            let width = width as f32;
            let height = height as f32;
            x0 *= width;
            x1 *= width;
            y0 *= height;
            y1 *= height;
        }

        if clamp {
            if let Some((width, height)) = dimensions {
                x0 = x0.clamp(0.0, width as f32);
                x1 = x1.clamp(0.0, width as f32);
                y0 = y0.clamp(0.0, height as f32);
                y1 = y1.clamp(0.0, height as f32);
            }
        }

        if x1 <= x0 || y1 <= y0 {
            return None;
        }
        let epsilon = 0.0001;
        let x = (x0 + epsilon).floor().max(0.0) as u32;
        let y = (y0 + epsilon).floor().max(0.0) as u32;
        let width = ((x1 - epsilon).ceil() as u32).saturating_sub(x);
        let height = ((y1 - epsilon).ceil() as u32).saturating_sub(y);
        BoundingBox::new(x, y, width, height).ok()
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for normalized prediction.
pub struct NormalizedPrediction {
    /// The kind value.
    pub kind: ObservationKind,
    /// Label assigned to this value.
    pub label: Option<String>,
    /// Text content for this value.
    pub text: Option<String>,
    /// Score assigned to this value.
    pub score: Option<f32>,
    /// The region value.
    pub region: Option<BoundingBox>,
    /// The attributes value.
    pub attributes: BTreeMap<String, String>,
}

impl NormalizedPrediction {
    /// Converts this value to observation.
    pub fn to_observation(&self, analyzer: impl Into<String>) -> Observation {
        let mut observation = Observation::new(analyzer, self.kind.clone());
        if let Some(label) = &self.label {
            observation = observation.label(label.clone());
        }
        if let Some(text) = &self.text {
            observation = observation.text(text.clone());
        }
        if let Some(score) = self.score {
            observation = observation.score(score);
        }
        if let Some(region) = self.region {
            observation = observation.region(region);
        }
        for (key, value) in &self.attributes {
            observation = observation.attribute(key.clone(), value.clone());
        }
        observation
    }

    /// Converts this value to event.
    pub fn to_event(&self, analyzer: impl Into<String>) -> AnalysisEvent {
        let label = self
            .label
            .clone()
            .or_else(|| self.text.clone())
            .unwrap_or_else(|| "prediction".to_string());
        let mut event = AnalysisEvent::new(analyzer, label);
        if let Some(score) = self.score {
            event = event.score(score);
        }
        event
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Data type for prediction repair options.
pub struct PredictionRepairOptions {
    /// The min score value.
    pub min_score: Option<f32>,
    /// The clamp regions value.
    pub clamp_regions: bool,
    /// The nms IoU threshold value.
    pub nms_iou_threshold: Option<f32>,
    /// The fill missing labels value.
    pub fill_missing_labels: bool,
}

impl Default for PredictionRepairOptions {
    fn default() -> Self {
        Self {
            min_score: None,
            clamp_regions: true,
            nms_iou_threshold: Some(0.5),
            fill_missing_labels: true,
        }
    }
}

/// Returns normalize predictions.
pub fn normalize_predictions(
    raw: Vec<RawPrediction>,
    task: &ModelTask,
    dimensions: Option<(u32, u32)>,
    repair: PredictionRepairOptions,
) -> Vec<NormalizedPrediction> {
    let mut predictions = raw
        .into_iter()
        .filter(|prediction| {
            repair
                .min_score
                .zip(prediction.score)
                .map(|(minimum, score)| score >= minimum)
                .unwrap_or(true)
        })
        .map(|prediction| normalize_prediction(prediction, task, dimensions, repair))
        .collect::<Vec<_>>();

    if let Some(threshold) = repair.nms_iou_threshold {
        predictions = non_max_suppression(predictions, threshold);
    }
    predictions
}

fn normalize_prediction(
    prediction: RawPrediction,
    task: &ModelTask,
    dimensions: Option<(u32, u32)>,
    repair: PredictionRepairOptions,
) -> NormalizedPrediction {
    let kind = prediction
        .kind
        .as_deref()
        .map(kind_from_str)
        .unwrap_or_else(|| default_kind(task));
    let label = prediction.label.or_else(|| {
        repair
            .fill_missing_labels
            .then(|| task.default_label().to_string())
    });
    let region = prediction
        .region
        .and_then(|region| region.to_bounding_box(dimensions, repair.clamp_regions));

    NormalizedPrediction {
        kind,
        label,
        text: prediction.text,
        score: prediction.score,
        region,
        attributes: prediction.attributes,
    }
}

fn default_kind(task: &ModelTask) -> ObservationKind {
    match task {
        ModelTask::ObjectDetection => ObservationKind::Object,
        ModelTask::PoseEstimation2d | ModelTask::PoseLifting3d => {
            ObservationKind::Custom("posture".to_string())
        }
        ModelTask::ImageClassification => ObservationKind::Scene,
        ModelTask::FaceDetection => ObservationKind::Face,
        ModelTask::Ocr
        | ModelTask::TextClassification
        | ModelTask::TokenClassification
        | ModelTask::ZeroShotClassification => ObservationKind::Text,
        ModelTask::AudioClassification | ModelTask::AudioEventDetection => {
            ObservationKind::Custom("audio".to_string())
        }
        ModelTask::ImageSegmentation => ObservationKind::Custom("image_segmentation".to_string()),
        ModelTask::ImageEmbedding => ObservationKind::Custom("image_embedding".to_string()),
        ModelTask::FaceEmbedding => ObservationKind::Custom("face_embedding".to_string()),
        ModelTask::AudioEmbedding => ObservationKind::Custom("audio_embedding".to_string()),
        ModelTask::SpeechRecognition => ObservationKind::Custom("speech_recognition".to_string()),
        ModelTask::SpeakerDiarization => ObservationKind::Custom("speaker".to_string()),
        ModelTask::SourceSeparation => ObservationKind::Custom("source_separation".to_string()),
        ModelTask::AudioGeneration => ObservationKind::Custom("audio_generation".to_string()),
        ModelTask::SpeakerConditionedTts => {
            ObservationKind::Custom("speaker_conditioned_tts".to_string())
        }
        ModelTask::TextEmbedding => ObservationKind::Custom("embedding".to_string()),
        ModelTask::Summarization => ObservationKind::Custom("summary".to_string()),
        ModelTask::Reranking => ObservationKind::Custom("reranking".to_string()),
        ModelTask::QuestionAnswering => ObservationKind::Custom("question_answering".to_string()),
        ModelTask::MultimodalEmbedding => {
            ObservationKind::Custom("multimodal_embedding".to_string())
        }
        ModelTask::Custom(kind) => ObservationKind::Custom(kind.clone()),
    }
}

fn kind_from_str(kind: &str) -> ObservationKind {
    match kind {
        "object" | "detection" | "object_detection" => ObservationKind::Object,
        "text" | "ocr" | "token" | "text_classification" => ObservationKind::Text,
        "face" => ObservationKind::Face,
        "scene" | "image_classification" => ObservationKind::Scene,
        other => ObservationKind::Custom(other.to_string()),
    }
}

fn non_max_suppression(
    mut predictions: Vec<NormalizedPrediction>,
    threshold: f32,
) -> Vec<NormalizedPrediction> {
    predictions.sort_by(|left, right| {
        right
            .score
            .unwrap_or(0.0)
            .total_cmp(&left.score.unwrap_or(0.0))
    });
    let mut kept: Vec<NormalizedPrediction> = Vec::new();
    'candidate: for prediction in predictions {
        if let Some(region) = prediction.region {
            for existing in &kept {
                if existing.region.is_some()
                    && existing.kind == prediction.kind
                    && existing.label == prediction.label
                    && bbox_iou(existing.region.unwrap(), region) > threshold
                {
                    continue 'candidate;
                }
            }
        }
        kept.push(prediction);
    }
    kept
}

fn bbox_iou(left: BoundingBox, right: BoundingBox) -> f32 {
    let left_x1 = left.x + left.width;
    let left_y1 = left.y + left.height;
    let right_x1 = right.x + right.width;
    let right_y1 = right.y + right.height;

    let ix0 = left.x.max(right.x);
    let iy0 = left.y.max(right.y);
    let ix1 = left_x1.min(right_x1);
    let iy1 = left_y1.min(right_y1);
    if ix1 <= ix0 || iy1 <= iy0 {
        return 0.0;
    }
    let intersection = (ix1 - ix0) as f32 * (iy1 - iy0) as f32;
    let left_area = left.width as f32 * left.height as f32;
    let right_area = right.width as f32 * right.height as f32;
    intersection / (left_area + right_area - intersection)
}
