#![doc = include_str!("../README.md")]

pub mod surface;

use std::collections::BTreeMap;

use math_geometry_2d::{GeometryError, Point2f, RectU32};
use serde::{Deserialize, Serialize};

pub use math_geometry_2d::{Point2f as VisualPoint, RectU32 as VisualRegion};

/// Result type for visual contract validation.
pub type Result<T> = std::result::Result<T, VisionError>;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Error type for visual contract validation.
pub enum VisionError {
    /// Invalid argument.
    InvalidArgument(String),
    /// Invalid geometry.
    Geometry(GeometryError),
}

impl std::fmt::Display for VisionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArgument(message) => f.write_str(message),
            Self::Geometry(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for VisionError {}

impl From<GeometryError> for VisionError {
    fn from(value: GeometryError) -> Self {
        Self::Geometry(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// Visual-only detection kind.
pub enum VisualDetectionKind {
    /// A detected face.
    Face,
    /// A detected person.
    Person,
    /// A generic detected object.
    Object,
    /// A localized text region.
    TextRegion,
    /// A caller or model-specific visual kind.
    Custom(String),
}

impl VisualDetectionKind {
    /// Validates this value.
    pub fn validate(&self) -> Result<()> {
        if let Self::Custom(value) = self {
            validate_non_empty(value, "custom visual detection kind")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// A point attached to a visual detection.
pub struct VisualKeypoint {
    /// Point location in image coordinates.
    pub point: Point2f,
    /// Optional semantic point name.
    pub name: Option<String>,
    /// Optional point confidence score.
    pub score: Option<f32>,
}

impl VisualKeypoint {
    /// Creates a new value.
    pub fn new(point: Point2f) -> Result<Self> {
        point.validate()?;
        Ok(Self {
            point,
            name: None,
            score: None,
        })
    }

    /// Sets the semantic point name.
    pub fn name(mut self, value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        validate_non_empty(&value, "visual keypoint name")?;
        self.name = Some(value);
        Ok(self)
    }

    /// Sets the point confidence score.
    pub fn score(mut self, value: f32) -> Result<Self> {
        validate_score(value, "visual keypoint score")?;
        self.score = Some(value);
        Ok(self)
    }

    /// Validates this value.
    pub fn validate(&self) -> Result<()> {
        self.point.validate()?;
        if let Some(name) = &self.name {
            validate_non_empty(name, "visual keypoint name")?;
        }
        if let Some(score) = self.score {
            validate_score(score, "visual keypoint score")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// A localized visual finding.
pub struct VisualDetection {
    /// Optional stable detection identifier.
    pub id: Option<String>,
    /// Visual detection kind.
    pub kind: VisualDetectionKind,
    /// Optional detector label.
    pub label: Option<String>,
    /// Optional detection confidence score.
    pub score: Option<f32>,
    /// Detection region in absolute pixel coordinates.
    pub region: RectU32,
    /// Optional localized keypoints.
    pub keypoints: Vec<VisualKeypoint>,
    /// Caller or adapter-specific metadata.
    pub attributes: BTreeMap<String, String>,
}

impl VisualDetection {
    /// Creates a new value.
    pub fn new(kind: VisualDetectionKind, region: RectU32) -> Result<Self> {
        kind.validate()?;
        region.validate()?;
        Ok(Self {
            id: None,
            kind,
            label: None,
            score: None,
            region,
            keypoints: Vec::new(),
            attributes: BTreeMap::new(),
        })
    }

    /// Creates a face detection.
    pub fn face(region: RectU32) -> Result<Self> {
        Self::new(VisualDetectionKind::Face, region)
    }

    /// Creates an object detection.
    pub fn object(region: RectU32) -> Result<Self> {
        Self::new(VisualDetectionKind::Object, region)
    }

    /// Sets a stable detection identifier.
    pub fn id(mut self, value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        validate_non_empty(&value, "visual detection id")?;
        self.id = Some(value);
        Ok(self)
    }

    /// Sets the detector label.
    pub fn label(mut self, value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        validate_non_empty(&value, "visual detection label")?;
        self.label = Some(value);
        Ok(self)
    }

    /// Sets the confidence score.
    pub fn score(mut self, value: f32) -> Result<Self> {
        validate_score(value, "visual detection score")?;
        self.score = Some(value);
        Ok(self)
    }

    /// Adds one keypoint.
    pub fn keypoint(mut self, value: VisualKeypoint) -> Result<Self> {
        value.validate()?;
        self.keypoints.push(value);
        Ok(self)
    }

    /// Adds adapter-specific metadata.
    pub fn attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Result<Self> {
        let key = key.into();
        validate_non_empty(&key, "visual detection attribute key")?;
        self.attributes.insert(key, value.into());
        Ok(self)
    }

    /// Validates this value.
    pub fn validate(&self) -> Result<()> {
        if let Some(id) = &self.id {
            validate_non_empty(id, "visual detection id")?;
        }
        self.kind.validate()?;
        if let Some(label) = &self.label {
            validate_non_empty(label, "visual detection label")?;
        }
        if let Some(score) = self.score {
            validate_score(score, "visual detection score")?;
        }
        self.region.validate()?;
        for keypoint in &self.keypoints {
            keypoint.validate()?;
        }
        for key in self.attributes.keys() {
            validate_non_empty(key, "visual detection attribute key")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// A dense vector representation of an image, region, or detection.
pub struct VisualEmbedding {
    /// Optional stable embedding identifier.
    pub id: Option<String>,
    /// Optional source detection identifier.
    pub source_detection_id: Option<String>,
    /// Optional visual kind represented by the embedding.
    pub kind: Option<VisualDetectionKind>,
    /// Optional source region in absolute pixel coordinates.
    pub region: Option<RectU32>,
    /// Dense embedding values.
    pub vector: Vec<f32>,
    /// Caller or adapter-specific metadata.
    pub attributes: BTreeMap<String, String>,
}

impl VisualEmbedding {
    /// Creates a new value.
    pub fn new(vector: impl Into<Vec<f32>>) -> Result<Self> {
        let value = Self {
            id: None,
            source_detection_id: None,
            kind: None,
            region: None,
            vector: vector.into(),
            attributes: BTreeMap::new(),
        };
        value.validate()?;
        Ok(value)
    }

    /// Sets a stable embedding identifier.
    pub fn id(mut self, value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        validate_non_empty(&value, "visual embedding id")?;
        self.id = Some(value);
        Ok(self)
    }

    /// Sets the source detection identifier.
    pub fn source_detection_id(mut self, value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        validate_non_empty(&value, "visual embedding source detection id")?;
        self.source_detection_id = Some(value);
        Ok(self)
    }

    /// Sets the visual kind.
    pub fn kind(mut self, value: VisualDetectionKind) -> Result<Self> {
        value.validate()?;
        self.kind = Some(value);
        Ok(self)
    }

    /// Sets the source region.
    pub fn region(mut self, value: RectU32) -> Result<Self> {
        value.validate()?;
        self.region = Some(value);
        Ok(self)
    }

    /// Adds adapter-specific metadata.
    pub fn attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Result<Self> {
        let key = key.into();
        validate_non_empty(&key, "visual embedding attribute key")?;
        self.attributes.insert(key, value.into());
        Ok(self)
    }

    /// Validates this value.
    pub fn validate(&self) -> Result<()> {
        if let Some(id) = &self.id {
            validate_non_empty(id, "visual embedding id")?;
        }
        if let Some(id) = &self.source_detection_id {
            validate_non_empty(id, "visual embedding source detection id")?;
        }
        if let Some(kind) = &self.kind {
            kind.validate()?;
        }
        if let Some(region) = self.region {
            region.validate()?;
        }
        validate_vector(&self.vector, "visual embedding vector")?;
        for key in self.attributes.keys() {
            validate_non_empty(key, "visual embedding attribute key")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// A scored hypothesis linking a visual source to a reference identity.
pub struct IdentityMatch {
    /// Optional source detection identifier.
    pub source_detection_id: Option<String>,
    /// Optional source embedding identifier.
    pub source_embedding_id: Option<String>,
    /// Matched reference identifier.
    pub reference_id: String,
    /// Optional matched reference label.
    pub label: Option<String>,
    /// Match score.
    pub score: f32,
    /// Optional visual kind.
    pub kind: Option<VisualDetectionKind>,
    /// Caller or adapter-specific metadata.
    pub attributes: BTreeMap<String, String>,
}

impl IdentityMatch {
    /// Creates a new value.
    pub fn new(reference_id: impl Into<String>, score: f32) -> Result<Self> {
        let value = Self {
            source_detection_id: None,
            source_embedding_id: None,
            reference_id: reference_id.into(),
            label: None,
            score,
            kind: None,
            attributes: BTreeMap::new(),
        };
        value.validate()?;
        Ok(value)
    }

    /// Sets the source detection identifier.
    pub fn source_detection_id(mut self, value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        validate_non_empty(&value, "identity match source detection id")?;
        self.source_detection_id = Some(value);
        Ok(self)
    }

    /// Sets the source embedding identifier.
    pub fn source_embedding_id(mut self, value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        validate_non_empty(&value, "identity match source embedding id")?;
        self.source_embedding_id = Some(value);
        Ok(self)
    }

    /// Sets the matched reference label.
    pub fn label(mut self, value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        validate_non_empty(&value, "identity match label")?;
        self.label = Some(value);
        Ok(self)
    }

    /// Sets the visual kind.
    pub fn kind(mut self, value: VisualDetectionKind) -> Result<Self> {
        value.validate()?;
        self.kind = Some(value);
        Ok(self)
    }

    /// Adds adapter-specific metadata.
    pub fn attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Result<Self> {
        let key = key.into();
        validate_non_empty(&key, "identity match attribute key")?;
        self.attributes.insert(key, value.into());
        Ok(self)
    }

    /// Validates this value.
    pub fn validate(&self) -> Result<()> {
        if let Some(id) = &self.source_detection_id {
            validate_non_empty(id, "identity match source detection id")?;
        }
        if let Some(id) = &self.source_embedding_id {
            validate_non_empty(id, "identity match source embedding id")?;
        }
        validate_non_empty(&self.reference_id, "identity match reference id")?;
        if let Some(label) = &self.label {
            validate_non_empty(label, "identity match label")?;
        }
        validate_score(self.score, "identity match score")?;
        if let Some(kind) = &self.kind {
            kind.validate()?;
        }
        for key in self.attributes.keys() {
            validate_non_empty(key, "identity match attribute key")?;
        }
        Ok(())
    }
}

fn validate_non_empty(value: &str, field: &str) -> Result<()> {
    if value.trim().is_empty() {
        Err(VisionError::InvalidArgument(format!(
            "{field} must not be empty"
        )))
    } else {
        Ok(())
    }
}

fn validate_score(value: f32, field: &str) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(VisionError::InvalidArgument(format!(
            "{field} must be finite"
        )))
    }
}

fn validate_vector(values: &[f32], field: &str) -> Result<()> {
    if values.is_empty() {
        return Err(VisionError::InvalidArgument(format!(
            "{field} must not be empty"
        )));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(VisionError::InvalidArgument(format!(
            "{field} values must be finite"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_detection_embedding_and_match() -> Result<()> {
        let region = RectU32::new(1, 2, 10, 12)?;
        let keypoint = VisualKeypoint::new(Point2f::new(4.0, 6.0)?)?
            .name("left_eye")?
            .score(0.95)?;
        let detection = VisualDetection::face(region)?
            .id("face-1")?
            .score(0.9)?
            .keypoint(keypoint)?;
        detection.validate()?;

        let embedding = VisualEmbedding::new([1.0, 0.0])?
            .id("embedding-1")?
            .source_detection_id("face-1")?
            .kind(VisualDetectionKind::Face)?
            .region(region)?;
        embedding.validate()?;

        let identity_match = IdentityMatch::new("alice", 0.99)?
            .source_detection_id("face-1")?
            .source_embedding_id("embedding-1")?
            .label("Alice")?
            .kind(VisualDetectionKind::Face)?;
        identity_match.validate()?;
        Ok(())
    }

    #[test]
    fn rejects_invalid_values() {
        assert!(VisualDetectionKind::Custom(" ".to_string())
            .validate()
            .is_err());
        assert!(VisualEmbedding::new(Vec::<f32>::new()).is_err());
        assert!(VisualEmbedding::new([f32::NAN]).is_err());
        assert!(IdentityMatch::new("alice", f32::INFINITY).is_err());
        assert!(VisualDetection::face(RectU32::new(0, 0, 1, 1).unwrap())
            .unwrap()
            .id(" ")
            .is_err());
    }

    #[test]
    fn serializes_camel_case_contracts() -> Result<()> {
        let detection =
            VisualDetection::new(VisualDetectionKind::TextRegion, RectU32::new(0, 0, 5, 6)?)?
                .id("text-1")?;
        let value = serde_json::to_value(detection).unwrap();
        assert_eq!(value["id"], "text-1");
        assert_eq!(value["kind"], "textRegion");
        assert!(value.get("keypoints").is_some());
        Ok(())
    }
}
