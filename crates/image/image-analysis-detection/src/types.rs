//! Shared detection results and their conversion to the visual contract.

use std::collections::BTreeMap;

use image_analysis_core::ImageView;
use model_runtime::{HuggingFaceModelSpec, ModelTask};
use video_analysis_core::{BoundingBox, DetectError, FramePosition, Result};
use vision_core::{
    VisualDetection, VisualDetectionKind, VisualKeypoint, VisualPoint, VisualRegion,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Variants describing face detection preset.
pub enum FaceDetectionPreset {
    #[default]
    /// The OpenCV YuNet ONNX variant.
    OpenCvYuNet,
}

impl FaceDetectionPreset {
    /// Returns model spec.
    pub fn model_spec(self) -> HuggingFaceModelSpec {
        match self {
            Self::OpenCvYuNet => {
                HuggingFaceModelSpec::new("opencv/face_detection_yunet", ModelTask::FaceDetection)
                    .name("opencv-yunet-onnx")
                    .first_available_file([
                        "face_detection_yunet_2023mar.onnx",
                        "face_detection_yunet_2023mar_int8.onnx",
                        "face_detection_yunet_2023mar_int8bq.onnx",
                    ])
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for image detection.
pub struct ImageDetection {
    /// Label assigned to this value.
    pub label: String,
    /// Score assigned to this value.
    pub score: Option<f32>,
    /// The region value.
    pub region: BoundingBox,
    /// The attributes value.
    pub attributes: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
/// Normalized face bounding box.
pub struct FaceBox {
    /// Normalized x coordinate.
    pub x: f32,
    /// Normalized y coordinate.
    pub y: f32,
    /// Normalized width.
    pub width: f32,
    /// Normalized height.
    pub height: f32,
}

impl FaceBox {
    /// Creates a new value.
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Result<Self> {
        let bbox = Self {
            x,
            y,
            width,
            height,
        };
        bbox.validate()?;
        Ok(bbox)
    }

    /// Validates this value.
    pub fn validate(self) -> Result<()> {
        if [self.x, self.y, self.width, self.height]
            .iter()
            .any(|value| !value.is_finite())
            || self.width <= 0.0
            || self.height <= 0.0
        {
            return Err(DetectError::InvalidArgument(
                "face boxes must be finite and have non-zero dimensions".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
/// Normalized face landmark points.
pub struct FaceLandmarks {
    /// The points value.
    pub points: Vec<[f32; 2]>,
}

impl FaceLandmarks {
    /// Creates a new value.
    pub fn new(points: Vec<[f32; 2]>) -> Result<Self> {
        if points.iter().flatten().any(|value| !value.is_finite()) {
            return Err(DetectError::InvalidArgument(
                "face landmarks must be finite".to_string(),
            ));
        }
        Ok(Self { points })
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for face detection.
pub struct FaceDetection {
    /// The bounding box value.
    pub bbox: FaceBox,
    /// The confidence value.
    pub confidence: f32,
    /// The landmarks value.
    pub landmarks: Option<FaceLandmarks>,
    /// Adapter-specific attributes.
    pub attributes: BTreeMap<String, String>,
}

impl FaceDetection {
    /// Creates a new value.
    pub fn new(bbox: FaceBox, confidence: f32) -> Result<Self> {
        bbox.validate()?;
        if !confidence.is_finite() {
            return Err(DetectError::InvalidArgument(
                "face detection confidence must be finite".to_string(),
            ));
        }
        Ok(Self {
            bbox,
            confidence,
            landmarks: None,
            attributes: BTreeMap::new(),
        })
    }

    /// Returns landmarks.
    pub fn landmarks(mut self, landmarks: FaceLandmarks) -> Self {
        self.landmarks = Some(landmarks);
        self
    }

    /// Returns attribute.
    pub fn attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// Converts this specialized face detection into a shared visual detection.
    pub fn to_visual_detection_for_size(&self, width: u32, height: u32) -> Result<VisualDetection> {
        let region = normalized_face_region(self.bbox, width, height)?;
        let mut visual = VisualDetection::face(region)
            .map_err(vision_error)?
            .score(self.confidence)
            .map_err(vision_error)?;
        visual.attributes = self.attributes.clone();

        if let Some(landmarks) = &self.landmarks {
            for (index, point) in landmarks.points.iter().enumerate() {
                let x = point[0] * width as f32;
                let y = point[1] * height as f32;
                let keypoint = VisualKeypoint::new(VisualPoint::new(x, y)?)
                    .map_err(vision_error)?
                    .name(format!("landmark_{index}"))
                    .map_err(vision_error)?;
                visual = visual.keypoint(keypoint).map_err(vision_error)?;
            }
        }

        Ok(visual)
    }

    /// Converts this specialized face detection using dimensions from an image view.
    pub fn to_visual_detection(&self, image: &ImageView<'_>) -> Result<VisualDetection> {
        self.to_visual_detection_for_size(image.width, image.height)
    }
}

/// Trait for face detector backend implementations.
pub trait FaceDetectorBackend {
    /// Returns detect faces.
    fn detect_faces(&mut self, image: &ImageView<'_>) -> Result<Vec<FaceDetection>>;
}

impl ImageDetection {
    /// Converts this image detection into a shared visual detection.
    pub fn to_visual_detection(&self) -> Result<VisualDetection> {
        let mut visual = VisualDetection::new(
            visual_kind_from_label(&self.label),
            VisualRegion::from(self.region),
        )
        .map_err(vision_error)?
        .label(self.label.clone())
        .map_err(vision_error)?;
        if let Some(score) = self.score {
            visual = visual.score(score).map_err(vision_error)?;
        }
        visual.attributes = self.attributes.clone();
        Ok(visual)
    }

    /// Returns attribute.
    pub fn attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }
}

impl TryFrom<ImageDetection> for VisualDetection {
    type Error = DetectError;

    fn try_from(value: ImageDetection) -> std::result::Result<Self, Self::Error> {
        value.to_visual_detection()
    }
}

impl TryFrom<&ImageDetection> for VisualDetection {
    type Error = DetectError;

    fn try_from(value: &ImageDetection) -> std::result::Result<Self, Self::Error> {
        value.to_visual_detection()
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for frame detections.
pub struct FrameDetections {
    /// The position value.
    pub position: FramePosition,
    /// The detections value.
    pub detections: Vec<ImageDetection>,
}

fn visual_kind_from_label(label: &str) -> VisualDetectionKind {
    match label.trim().to_ascii_lowercase().as_str() {
        "face" => VisualDetectionKind::Face,
        "person" => VisualDetectionKind::Person,
        "text" | "text_region" | "text-region" | "textregion" => VisualDetectionKind::TextRegion,
        "object" => VisualDetectionKind::Object,
        "" => VisualDetectionKind::Object,
        value => VisualDetectionKind::Custom(value.to_string()),
    }
}

fn normalized_face_region(bbox: FaceBox, width: u32, height: u32) -> Result<VisualRegion> {
    if width == 0 || height == 0 {
        return Err(DetectError::InvalidDimensions { width, height });
    }
    let image_width = width as f32;
    let image_height = height as f32;
    let x = (bbox.x * image_width).floor().clamp(0.0, image_width) as u32;
    let y = (bbox.y * image_height).floor().clamp(0.0, image_height) as u32;
    let max_x = ((bbox.x + bbox.width) * image_width)
        .ceil()
        .clamp(0.0, image_width) as u32;
    let max_y = ((bbox.y + bbox.height) * image_height)
        .ceil()
        .clamp(0.0, image_height) as u32;
    VisualRegion::new(x, y, max_x.saturating_sub(x), max_y.saturating_sub(y))
        .map_err(DetectError::from)
}

fn vision_error(error: vision_core::VisionError) -> DetectError {
    DetectError::InvalidArgument(error.to_string())
}
