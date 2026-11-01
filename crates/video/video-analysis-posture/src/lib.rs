#![doc = include_str!("../README.md")]

pub mod geometry;
pub mod surface;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use image_analysis_core::ImageView;
use image_analysis_processing::{
    image_model_preprocessing_from_config, preprocess_image_for_model,
    validate_image_model_preprocessing, ImageModelPreprocessing, ImageModelTensor,
};
use math_geometry_2d::{NormalizedPoint2, Point2f, Size2u};
use serde::{Deserialize, Serialize};
pub use geometry::{LineSegment3, Point3, Vector3};
use video_analysis_core::{
    BoundingBox, DetectError, Observation, ObservationKind, Result, VideoAnalyzer, VideoFrame,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for keypoint.
pub struct Keypoint {
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

impl Keypoint {
    /// Creates a new value.
    pub fn new(name: impl Into<String>, x: f32, y: f32) -> Result<Self> {
        let keypoint = Self {
            name: name.into(),
            x,
            y,
            score: None,
            visible: None,
        };
        keypoint.validate()?;
        Ok(keypoint)
    }

    /// Returns score.
    pub fn score(mut self, score: f32) -> Result<Self> {
        self.score = Some(score);
        self.validate()?;
        Ok(self)
    }

    /// Returns visible.
    pub fn visible(mut self, visible: bool) -> Self {
        self.visible = Some(visible);
        self
    }

    /// Validates this value.
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(invalid_argument("keypoint name must not be empty"));
        }
        if !self.x.is_finite() || !self.y.is_finite() {
            return Err(invalid_argument("keypoint coordinates must be finite"));
        }
        if let Some(score) = self.score {
            if !score.is_finite() {
                return Err(invalid_argument("keypoint score must be finite"));
            }
        }
        Ok(())
    }

    /// Converts this value to point2f.
    pub fn to_point2f(&self) -> Result<Point2f> {
        Ok(Point2f::new(self.x, self.y)?)
    }

    /// Builds this value from point2f.
    pub fn from_point2f(name: impl Into<String>, point: Point2f) -> Result<Self> {
        Self::new(name, point.x, point.y)
    }

    /// Converts this value to normalized point2.
    pub fn to_normalized_point2(&self, image_size: Size2u) -> Result<NormalizedPoint2> {
        Ok(self.to_point2f()?.to_normalized(image_size)?)
    }

    /// Builds this value from normalized point2.
    pub fn from_normalized_point2(
        name: impl Into<String>,
        point: NormalizedPoint2,
        image_size: Size2u,
    ) -> Result<Self> {
        let point = point.to_pixel_point_f32(image_size);
        Self::new(name, point.x, point.y)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for keypoint3d.
pub struct Keypoint3d {
    /// Human-readable name for this value.
    pub name: String,
    /// The position value.
    pub position: Point3,
    /// Score assigned to this value.
    pub score: Option<f32>,
    /// The visible value.
    pub visible: Option<bool>,
}

impl Keypoint3d {
    /// Creates a new value.
    pub fn new(name: impl Into<String>, position: Point3) -> Result<Self> {
        let keypoint = Self {
            name: name.into(),
            position,
            score: None,
            visible: None,
        };
        keypoint.validate()?;
        Ok(keypoint)
    }

    /// Returns score.
    pub fn score(mut self, score: f32) -> Result<Self> {
        self.score = Some(score);
        self.validate()?;
        Ok(self)
    }

    /// Returns visible.
    pub fn visible(mut self, visible: bool) -> Self {
        self.visible = Some(visible);
        self
    }

    /// Validates this value.
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(invalid_argument("3D keypoint name must not be empty"));
        }
        if !self.position.is_finite() {
            return Err(invalid_argument("3D keypoint coordinates must be finite"));
        }
        if let Some(score) = self.score {
            if !score.is_finite() {
                return Err(invalid_argument("3D keypoint score must be finite"));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// Variants describing keypoint space.
pub enum KeypointSpace {
    /// The pixel variant.
    Pixel,
    /// The normalized variant.
    Normalized,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Data type for skeleton edge.
pub struct SkeletonEdge {
    /// The from value.
    pub from: String,
    /// The to value.
    pub to: String,
    /// Label assigned to this value.
    pub label: Option<String>,
}

impl SkeletonEdge {
    /// Creates a new value.
    pub fn new(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            label: None,
        }
    }

    /// Returns label.
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Data type for skeleton.
pub struct Skeleton {
    /// The keypoints value.
    pub keypoints: Vec<String>,
    /// The edges value.
    pub edges: Vec<SkeletonEdge>,
}

impl Skeleton {
    /// Creates a new value.
    pub fn new(keypoints: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            keypoints: keypoints.into_iter().map(Into::into).collect(),
            edges: Vec::new(),
        }
    }

    /// Returns edge.
    pub fn edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.edges.push(SkeletonEdge::new(from, to));
        self
    }

    /// Returns edge label.
    pub fn edge_label(
        mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        label: impl Into<String>,
    ) -> Self {
        self.edges.push(SkeletonEdge::new(from, to).label(label));
        self
    }

    /// Returns coco 17.
    pub fn coco_17() -> Self {
        Self::new([
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ])
        .edge("left_shoulder", "right_shoulder")
        .edge("left_shoulder", "left_elbow")
        .edge("left_elbow", "left_wrist")
        .edge("right_shoulder", "right_elbow")
        .edge("right_elbow", "right_wrist")
        .edge("left_shoulder", "left_hip")
        .edge("right_shoulder", "right_hip")
        .edge("left_hip", "right_hip")
        .edge("left_hip", "left_knee")
        .edge("left_knee", "left_ankle")
        .edge("right_hip", "right_knee")
        .edge("right_knee", "right_ankle")
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for pose estimate.
pub struct PoseEstimate {
    /// Identifier for this value.
    pub id: Option<String>,
    /// Label assigned to this value.
    pub label: Option<String>,
    /// Score assigned to this value.
    pub score: Option<f32>,
    /// The region value.
    pub region: Option<BoundingBox>,
    /// The keypoints value.
    pub keypoints: Vec<Keypoint>,
    /// The attributes value.
    pub attributes: BTreeMap<String, String>,
}

impl PoseEstimate {
    /// Creates a new value.
    pub fn new(keypoints: impl IntoIterator<Item = Keypoint>) -> Result<Self> {
        let estimate = Self {
            id: None,
            label: None,
            score: None,
            region: None,
            keypoints: keypoints.into_iter().collect(),
            attributes: BTreeMap::new(),
        };
        estimate.validate()?;
        Ok(estimate)
    }

    /// Returns identifier.
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Returns label.
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Returns score.
    pub fn score(mut self, score: f32) -> Result<Self> {
        self.score = Some(score);
        self.validate()?;
        Ok(self)
    }

    /// Returns region.
    pub fn region(mut self, region: BoundingBox) -> Self {
        self.region = Some(region);
        self
    }

    /// Returns attribute.
    pub fn attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// Returns keypoint.
    pub fn keypoint(&self, name: &str) -> Option<&Keypoint> {
        self.keypoints.iter().find(|keypoint| keypoint.name == name)
    }

    /// Validates this value.
    pub fn validate(&self) -> Result<()> {
        if self.keypoints.is_empty() {
            return Err(invalid_argument(
                "pose estimate must contain at least one keypoint",
            ));
        }
        if let Some(score) = self.score {
            if !score.is_finite() {
                return Err(invalid_argument("pose score must be finite"));
            }
        }
        for keypoint in &self.keypoints {
            keypoint.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for pose3d estimate.
pub struct Pose3dEstimate {
    /// Identifier for this value.
    pub id: Option<String>,
    /// Label assigned to this value.
    pub label: Option<String>,
    /// Score assigned to this value.
    pub score: Option<f32>,
    /// The keypoints value.
    pub keypoints: Vec<Keypoint3d>,
    /// The attributes value.
    pub attributes: BTreeMap<String, String>,
}

impl Pose3dEstimate {
    /// Creates a new value.
    pub fn new(keypoints: impl IntoIterator<Item = Keypoint3d>) -> Result<Self> {
        let estimate = Self {
            id: None,
            label: None,
            score: None,
            keypoints: keypoints.into_iter().collect(),
            attributes: BTreeMap::new(),
        };
        estimate.validate()?;
        Ok(estimate)
    }

    /// Returns identifier.
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Returns label.
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Returns score.
    pub fn score(mut self, score: f32) -> Result<Self> {
        self.score = Some(score);
        self.validate()?;
        Ok(self)
    }

    /// Returns attribute.
    pub fn attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// Returns keypoint.
    pub fn keypoint(&self, name: &str) -> Option<&Keypoint3d> {
        self.keypoints.iter().find(|keypoint| keypoint.name == name)
    }

    /// Validates this value.
    pub fn validate(&self) -> Result<()> {
        if self.keypoints.is_empty() {
            return Err(invalid_argument(
                "3D pose estimate must contain at least one keypoint",
            ));
        }
        if let Some(score) = self.score {
            if !score.is_finite() {
                return Err(invalid_argument("3D pose score must be finite"));
            }
        }
        for keypoint in &self.keypoints {
            keypoint.validate()?;
        }
        Ok(())
    }

    /// Converts this value to stick figure.
    pub fn to_stick_figure(&self, skeleton: Skeleton) -> Result<StickFigure3d> {
        StickFigure3d::new(skeleton, self.keypoints.clone())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for stick figure3d.
pub struct StickFigure3d {
    /// The skeleton value.
    pub skeleton: Skeleton,
    /// The keypoints value.
    pub keypoints: Vec<Keypoint3d>,
}

impl StickFigure3d {
    /// Creates a new value.
    pub fn new(skeleton: Skeleton, keypoints: impl Into<Vec<Keypoint3d>>) -> Result<Self> {
        let figure = Self {
            skeleton,
            keypoints: keypoints.into(),
        };
        figure.validate()?;
        Ok(figure)
    }

    /// Validates this value.
    pub fn validate(&self) -> Result<()> {
        for keypoint in &self.keypoints {
            keypoint.validate()?;
        }
        Ok(())
    }

    /// Returns keypoint.
    pub fn keypoint(&self, name: &str) -> Option<&Keypoint3d> {
        self.keypoints.iter().find(|keypoint| keypoint.name == name)
    }

    /// Returns segments.
    pub fn segments(&self) -> Result<Vec<LineSegment3>> {
        self.validate()?;
        let mut segments = Vec::new();
        for edge in &self.skeleton.edges {
            let Some(from) = self.keypoint(&edge.from) else {
                continue;
            };
            let Some(to) = self.keypoint(&edge.to) else {
                continue;
            };
            segments.push(LineSegment3::new(from.position, to.position)?);
        }
        Ok(segments)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Data type for pose sequence.
pub struct PoseSequence<T> {
    /// The frames value.
    pub frames: Vec<T>,
}

impl<T> PoseSequence<T> {
    /// Creates a new value.
    pub fn new(frames: impl Into<Vec<T>>) -> Self {
        Self {
            frames: frames.into(),
        }
    }

    /// Returns len.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Returns whether is empty.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Returns frames.
    pub fn frames(&self) -> &[T] {
        &self.frames
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Data type for posture options.
pub struct PostureOptions {
    /// The keypoint space value.
    pub keypoint_space: KeypointSpace,
    /// The min pose score value.
    pub min_pose_score: Option<f32>,
    /// The min keypoint score value.
    pub min_keypoint_score: Option<f32>,
    /// The infer region value.
    pub infer_region: bool,
}

impl Default for PostureOptions {
    fn default() -> Self {
        Self {
            keypoint_space: KeypointSpace::Pixel,
            min_pose_score: None,
            min_keypoint_score: None,
            infer_region: true,
        }
    }
}

impl PostureOptions {
    /// Validates this value.
    pub fn validate(self) -> Result<()> {
        if let Some(score) = self.min_pose_score {
            if !score.is_finite() {
                return Err(invalid_argument("minimum pose score must be finite"));
            }
        }
        if let Some(score) = self.min_keypoint_score {
            if !score.is_finite() {
                return Err(invalid_argument("minimum keypoint score must be finite"));
            }
        }
        Ok(())
    }
}

/// Trait for posture backend implementations.
pub trait PostureBackend {
    /// Returns estimate frame.
    fn estimate_frame(&mut self, frame: &VideoFrame<'_>) -> Result<Vec<PoseEstimate>>;
}

/// Trait for posture lift backend implementations.
pub trait PostureLiftBackend {
    /// Returns lift sequence.
    fn lift_sequence(
        &mut self,
        sequence: &PoseSequence<PoseEstimate>,
    ) -> Result<PoseSequence<Pose3dEstimate>>;
}

/// Data type for posture analyzer.
pub struct PostureAnalyzer<B> {
    name: String,
    backend: B,
    options: PostureOptions,
}

impl<B> PostureAnalyzer<B> {
    /// Creates a new value.
    pub fn new(name: impl Into<String>, backend: B) -> Self {
        Self {
            name: name.into(),
            backend,
            options: PostureOptions::default(),
        }
    }

    /// Returns options.
    pub fn options(mut self, options: PostureOptions) -> Result<Self> {
        options.validate()?;
        self.options = options;
        Ok(self)
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

impl<B: PostureBackend> VideoAnalyzer for PostureAnalyzer<B> {
    fn name(&self) -> &str {
        &self.name
    }

    fn process_frame(&mut self, frame: &VideoFrame<'_>) -> Result<Vec<Observation>> {
        self.options.validate()?;
        let estimates = self.backend.estimate_frame(frame)?;
        let mut observations = Vec::new();
        for estimate in estimates {
            estimate.validate()?;
            if self
                .options
                .min_pose_score
                .zip(estimate.score)
                .map(|(minimum, score)| score < minimum)
                .unwrap_or(false)
            {
                continue;
            }
            let keypoints = filtered_keypoints(&estimate, self.options.min_keypoint_score);
            if keypoints.is_empty() {
                continue;
            }
            observations.push(observation_for_pose(
                self.name(),
                frame,
                estimate,
                keypoints,
                self.options,
            ));
        }
        Ok(observations)
    }
}

/// Returns joint angle degrees.
pub fn joint_angle_degrees(a: &Keypoint, b: &Keypoint, c: &Keypoint) -> Result<f32> {
    a.validate()?;
    b.validate()?;
    c.validate()?;
    let ab = (a.x - b.x, a.y - b.y);
    let cb = (c.x - b.x, c.y - b.y);
    angle_from_vectors(Vector3::new(ab.0, ab.1, 0.0), Vector3::new(cb.0, cb.1, 0.0))
}

/// Returns joint angle 3d degrees.
pub fn joint_angle_3d_degrees(a: &Keypoint3d, b: &Keypoint3d, c: &Keypoint3d) -> Result<f32> {
    a.validate()?;
    b.validate()?;
    c.validate()?;
    angle_from_vectors(a.position - b.position, c.position - b.position)
}

/// Returns bone lengths.
pub fn bone_lengths(pose: &Pose3dEstimate, skeleton: &Skeleton) -> Result<BTreeMap<String, f32>> {
    pose.validate()?;
    let mut lengths = BTreeMap::new();
    for edge in &skeleton.edges {
        let Some(from) = pose.keypoint(&edge.from) else {
            continue;
        };
        let Some(to) = pose.keypoint(&edge.to) else {
            continue;
        };
        lengths.insert(
            edge.label
                .clone()
                .unwrap_or_else(|| format!("{}-{}", edge.from, edge.to)),
            from.position.distance(to.position),
        );
    }
    Ok(lengths)
}

/// Returns normalize pose3d.
pub fn normalize_pose3d(pose: &Pose3dEstimate, root_name: &str) -> Result<Pose3dEstimate> {
    pose.validate()?;
    let root = pose
        .keypoint(root_name)
        .ok_or_else(|| invalid_argument(format!("pose has no root keypoint `{root_name}`")))?;
    let max_radius = pose
        .keypoints
        .iter()
        .map(|keypoint| keypoint.position.distance(root.position))
        .fold(0.0_f32, f32::max)
        .max(1.0);
    let keypoints = pose
        .keypoints
        .iter()
        .map(|keypoint| {
            let relative = keypoint.position - root.position;
            let mut normalized = keypoint.clone();
            normalized.position = Point3::new(
                relative.x / max_radius,
                relative.y / max_radius,
                relative.z / max_radius,
            );
            normalized
        })
        .collect::<Vec<_>>();
    let mut normalized = Pose3dEstimate::new(keypoints)?;
    normalized.id = pose.id.clone();
    normalized.label = pose.label.clone();
    normalized.score = pose.score;
    normalized.attributes = pose.attributes.clone();
    Ok(normalized)
}

/// Returns interpolate missing joints.
pub fn interpolate_missing_joints(
    pose: &Pose3dEstimate,
    skeleton: &Skeleton,
) -> Result<Pose3dEstimate> {
    pose.validate()?;
    let mut keypoints = pose.keypoints.clone();
    for name in &skeleton.keypoints {
        if keypoints.iter().any(|keypoint| &keypoint.name == name) {
            continue;
        }
        let neighbors = skeleton
            .edges
            .iter()
            .filter_map(|edge| {
                if &edge.from == name {
                    pose.keypoint(&edge.to)
                } else if &edge.to == name {
                    pose.keypoint(&edge.from)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        if neighbors.is_empty() {
            continue;
        }
        let mean = neighbors.iter().fold(Vector3::ZERO, |sum, neighbor| {
            sum + Vector3::new(
                neighbor.position.x,
                neighbor.position.y,
                neighbor.position.z,
            )
        }) / neighbors.len() as f32;
        keypoints.push(Keypoint3d::new(
            name.clone(),
            Point3::new(mean.x, mean.y, mean.z),
        )?);
    }
    let mut interpolated = Pose3dEstimate::new(keypoints)?;
    interpolated.id = pose.id.clone();
    interpolated.label = pose.label.clone();
    interpolated.score = pose.score;
    interpolated.attributes = pose.attributes.clone();
    Ok(interpolated)
}

/// Returns smooth pose sequence.
pub fn smooth_pose_sequence(
    sequence: &PoseSequence<Pose3dEstimate>,
    window_radius: usize,
) -> Result<PoseSequence<Pose3dEstimate>> {
    if sequence.is_empty() {
        return Ok(PoseSequence::new(Vec::<Pose3dEstimate>::new()));
    }
    for pose in &sequence.frames {
        pose.validate()?;
    }
    let mut smoothed = Vec::with_capacity(sequence.frames.len());
    for frame_index in 0..sequence.frames.len() {
        let start = frame_index.saturating_sub(window_radius);
        let end = (frame_index + window_radius + 1).min(sequence.frames.len());
        let window = &sequence.frames[start..end];
        let current = &sequence.frames[frame_index];
        let mut keypoints = Vec::new();
        for keypoint in &current.keypoints {
            let matches = window
                .iter()
                .filter_map(|pose| pose.keypoint(&keypoint.name))
                .collect::<Vec<_>>();
            let mean = matches.iter().fold(Vector3::ZERO, |sum, item| {
                sum + Vector3::new(item.position.x, item.position.y, item.position.z)
            }) / matches.len() as f32;
            let mut smoothed_keypoint = keypoint.clone();
            smoothed_keypoint.position = Point3::new(mean.x, mean.y, mean.z);
            keypoints.push(smoothed_keypoint);
        }
        let mut pose = Pose3dEstimate::new(keypoints)?;
        pose.id = current.id.clone();
        pose.label = current.label.clone();
        pose.score = current.score;
        pose.attributes = current.attributes.clone();
        smoothed.push(pose);
    }
    Ok(PoseSequence::new(smoothed))
}

fn angle_from_vectors(ab: Vector3, cb: Vector3) -> Result<f32> {
    let ab_len = ab.length();
    let cb_len = cb.length();
    if ab_len <= f32::EPSILON || cb_len <= f32::EPSILON {
        return Err(invalid_argument(
            "joint angle requires non-overlapping keypoints",
        ));
    }
    let cosine = (ab.dot(cb) / (ab_len * cb_len)).clamp(-1.0, 1.0);
    Ok(cosine.acos().to_degrees())
}

fn filtered_keypoints(estimate: &PoseEstimate, min_score: Option<f32>) -> Vec<Keypoint> {
    estimate
        .keypoints
        .iter()
        .filter(|keypoint| {
            min_score
                .zip(keypoint.score)
                .map(|(minimum, score)| score >= minimum)
                .unwrap_or(true)
        })
        .cloned()
        .collect()
}

fn observation_for_pose(
    analyzer: &str,
    frame: &VideoFrame<'_>,
    estimate: PoseEstimate,
    keypoints: Vec<Keypoint>,
    options: PostureOptions,
) -> Observation {
    let mut observation =
        Observation::new(analyzer, ObservationKind::Custom("posture".to_string()))
            .at_frame(frame.position)
            .label(estimate.label.unwrap_or_else(|| "pose".to_string()))
            .attribute("keypoint_count", keypoints.len().to_string())
            .attribute("keypoints", encode_keypoints(&keypoints));
    if let Some(score) = estimate.score {
        observation = observation.score(score);
    }
    if let Some(id) = estimate.id {
        observation = observation.track_id(id);
    }
    let region = estimate.region.or_else(|| {
        options
            .infer_region
            .then(|| {
                infer_region(frame, &keypoints, options.keypoint_space)
                    .ok()
                    .flatten()
            })
            .flatten()
    });
    if let Some(region) = region {
        observation = observation.region(region);
    }
    for (key, value) in estimate.attributes {
        observation = observation.attribute(key, value);
    }
    observation
}

fn infer_region(
    frame: &VideoFrame<'_>,
    keypoints: &[Keypoint],
    keypoint_space: KeypointSpace,
) -> Result<Option<BoundingBox>> {
    let visible = keypoints
        .iter()
        .filter(|keypoint| keypoint.visible.unwrap_or(true))
        .collect::<Vec<_>>();
    if visible.is_empty() {
        return Ok(None);
    }

    let scale_x = if keypoint_space == KeypointSpace::Normalized {
        frame.width as f32
    } else {
        1.0
    };
    let scale_y = if keypoint_space == KeypointSpace::Normalized {
        frame.height as f32
    } else {
        1.0
    };

    let min_x = visible
        .iter()
        .map(|keypoint| keypoint.x * scale_x)
        .fold(f32::INFINITY, f32::min)
        .floor()
        .max(0.0) as u32;
    let min_y = visible
        .iter()
        .map(|keypoint| keypoint.y * scale_y)
        .fold(f32::INFINITY, f32::min)
        .floor()
        .max(0.0) as u32;
    let mut max_x = visible
        .iter()
        .map(|keypoint| keypoint.x * scale_x)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .min(frame.width as f32) as u32;
    let mut max_y = visible
        .iter()
        .map(|keypoint| keypoint.y * scale_y)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .min(frame.height as f32) as u32;
    if max_x <= min_x {
        max_x = (min_x + 1).min(frame.width);
    }
    if max_y <= min_y {
        max_y = (min_y + 1).min(frame.height);
    }
    if max_x <= min_x || max_y <= min_y {
        return Ok(None);
    }
    BoundingBox::new(min_x, min_y, max_x - min_x, max_y - min_y).map(Some)
}

fn encode_keypoints(keypoints: &[Keypoint]) -> String {
    keypoints
        .iter()
        .map(|keypoint| {
            let score = keypoint
                .score
                .map(|score| score.to_string())
                .unwrap_or_default();
            let visible = keypoint
                .visible
                .map(|visible| visible.to_string())
                .unwrap_or_default();
            format!(
                "{}:{:.3}:{:.3}:{score}:{visible}",
                keypoint.name, keypoint.x, keypoint.y
            )
        })
        .collect::<Vec<_>>()
        .join("|")
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for ONNX pose bundle info.
pub struct OnnxPoseBundleInfo {
    /// The config path value.
    pub config_path: PathBuf,
    /// The preprocessor config path value.
    pub preprocessor_config_path: Option<PathBuf>,
    /// The model path value.
    pub model_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for ONNX pose2d options.
pub struct OnnxPose2dOptions {
    /// The preprocessing value.
    pub preprocessing: ImageModelPreprocessing,
    /// The skeleton value.
    pub skeleton: Skeleton,
    /// The output space value.
    pub output_space: KeypointSpace,
    /// The min pose score value.
    pub min_pose_score: f32,
    /// The min keypoint score value.
    pub min_keypoint_score: f32,
}

impl Default for OnnxPose2dOptions {
    fn default() -> Self {
        Self {
            preprocessing: ImageModelPreprocessing::default(),
            skeleton: Skeleton::coco_17(),
            output_space: KeypointSpace::Normalized,
            min_pose_score: 0.0,
            min_keypoint_score: 0.0,
        }
    }
}

/// Trait for ONNX pose2d runner implementations.
pub trait OnnxPose2dRunner {
    /// Runs pose 2d.
    fn run_pose_2d(&mut self, input: &ImageModelTensor) -> Result<Vec<PoseEstimate>>;
}

#[derive(Debug, Clone, Default)]
/// Data type for unavailable pose2d runner.
pub struct UnavailablePose2dRunner;

impl OnnxPose2dRunner for UnavailablePose2dRunner {
    fn run_pose_2d(&mut self, _input: &ImageModelTensor) -> Result<Vec<PoseEstimate>> {
        Err(DetectError::Source(
            "native ONNX 2D pose execution is unavailable; construct with a runner or enable an executor"
                .to_string(),
        ))
    }
}

#[cfg(feature = "onnx")]
impl OnnxPose2dRunner for runtime_onnx::OnnxSession {
    fn run_pose_2d(&mut self, input: &ImageModelTensor) -> Result<Vec<PoseEstimate>> {
        use runtime_onnx::{OnnxRunner, OnnxTensor, OnnxTensorValue};

        let metadata = self.metadata().map_err(runtime_onnx_error)?;
        let input_name = runtime_onnx::input_name(&metadata, 0)
            .map_err(runtime_onnx_error)?
            .to_string();
        let outputs = self
            .run(vec![runtime_onnx::OnnxNamedTensor {
                name: input_name,
                tensor: OnnxTensorValue::F32(
                    OnnxTensor::new(
                        vec![
                            1,
                            input.channels,
                            input.height as usize,
                            input.width as usize,
                        ],
                        input.values.clone(),
                    )
                    .map_err(runtime_onnx_error)?,
                ),
            }])
            .map_err(runtime_onnx_error)?;
        let pose_tensor = pose_2d_output_tensor(&outputs)?;
        let scores = pose_scores_output(&outputs).map(|tensor| tensor.values.as_slice());
        match decode_pose_2d_tensor(pose_tensor, None, scores) {
            Ok(poses) => Ok(poses),
            Err(generic_error) => decode_yolov8_pose_tensor(
                pose_tensor,
                &Skeleton::coco_17(),
                input.width,
                input.height,
            )
            .map_err(|_| generic_error),
        }
    }
}

#[derive(Debug, Clone)]
/// Data type for ONNX pose2d estimator.
pub struct OnnxPose2dEstimator<R = UnavailablePose2dRunner> {
    options: OnnxPose2dOptions,
    runner: R,
}

#[cfg(not(feature = "onnx"))]
impl OnnxPose2dEstimator<UnavailablePose2dRunner> {
    /// Builds this value from bundle.
    pub fn from_bundle(bundle: model_runtime::ModelBundle) -> Result<Self> {
        Ok(Self {
            options: pose_2d_options_from_bundle(&bundle)?,
            runner: UnavailablePose2dRunner,
        })
    }
}

#[cfg(feature = "onnx")]
impl OnnxPose2dEstimator<runtime_onnx::OnnxSession> {
    /// Builds this value from bundle.
    pub fn from_bundle(bundle: model_runtime::ModelBundle) -> Result<Self> {
        let info = validate_onnx_pose_bundle(&bundle, model_runtime::ModelTask::PoseEstimation2d)?;
        Ok(Self {
            options: pose_2d_options_from_bundle(&bundle)?,
            runner: runtime_onnx::OnnxSession::from_file(info.model_path)
                .map_err(runtime_onnx_error)?,
        })
    }
}

impl<R: OnnxPose2dRunner> OnnxPose2dEstimator<R> {
    /// Builds this value from runner.
    pub fn from_runner(bundle: model_runtime::ModelBundle, runner: R) -> Result<Self> {
        Ok(Self {
            options: pose_2d_options_from_bundle(&bundle)?,
            runner,
        })
    }

    /// Returns this value with options.
    pub fn with_options(options: OnnxPose2dOptions, runner: R) -> Result<Self> {
        validate_image_model_preprocessing(&options.preprocessing)?;
        validate_threshold(options.min_pose_score)?;
        validate_threshold(options.min_keypoint_score)?;
        Ok(Self { options, runner })
    }

    /// Returns options.
    pub fn options(&self) -> &OnnxPose2dOptions {
        &self.options
    }

    /// Predicts poses from a video frame.
    pub fn predict_frame(&mut self, frame: &VideoFrame<'_>) -> Result<Vec<PoseEstimate>> {
        let input = preprocess_frame_for_model(frame, &self.options.preprocessing)?;
        let mut poses = self.runner.run_pose_2d(&input)?;
        validate_and_apply_pose_skeleton(&mut poses, &self.options.skeleton)?;
        poses.retain(|pose| pose.score.unwrap_or(1.0) >= self.options.min_pose_score);
        for pose in &mut poses {
            pose.keypoints.retain(|keypoint| {
                keypoint.score.unwrap_or(1.0) >= self.options.min_keypoint_score
            });
        }
        Ok(poses)
    }
}

impl<R: OnnxPose2dRunner> PostureBackend for OnnxPose2dEstimator<R> {
    fn estimate_frame(&mut self, frame: &VideoFrame<'_>) -> Result<Vec<PoseEstimate>> {
        self.predict_frame(frame)
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for ONNX pose lifter options.
pub struct OnnxPoseLifterOptions {
    /// The min pose score value.
    pub min_pose_score: f32,
}

impl Default for OnnxPoseLifterOptions {
    fn default() -> Self {
        Self {
            min_pose_score: 0.0,
        }
    }
}

#[derive(Debug, Clone, Default)]
/// Data type for unavailable pose lift runner.
pub struct UnavailablePoseLiftRunner;

/// Trait for ONNX pose lift runner implementations.
pub trait OnnxPoseLiftRunner {
    /// Runs pose lift.
    fn run_pose_lift(
        &mut self,
        sequence: &PoseSequence<PoseEstimate>,
    ) -> Result<PoseSequence<Pose3dEstimate>>;
}

impl OnnxPoseLiftRunner for UnavailablePoseLiftRunner {
    fn run_pose_lift(
        &mut self,
        _sequence: &PoseSequence<PoseEstimate>,
    ) -> Result<PoseSequence<Pose3dEstimate>> {
        Err(DetectError::Source(
            "native ONNX pose lifting execution is unavailable; construct with a runner or enable an executor"
                .to_string(),
        ))
    }
}

#[cfg(feature = "onnx")]
impl OnnxPoseLiftRunner for runtime_onnx::OnnxSession {
    fn run_pose_lift(
        &mut self,
        sequence: &PoseSequence<PoseEstimate>,
    ) -> Result<PoseSequence<Pose3dEstimate>> {
        use runtime_onnx::{OnnxRunner, OnnxTensorValue};

        let input = pose_lift_input_tensor(sequence)?;
        let metadata = self.metadata().map_err(runtime_onnx_error)?;
        let input_name = runtime_onnx::input_name(&metadata, 0)
            .map_err(runtime_onnx_error)?
            .to_string();
        let outputs = self
            .run(vec![runtime_onnx::OnnxNamedTensor {
                name: input_name,
                tensor: OnnxTensorValue::F32(input),
            }])
            .map_err(runtime_onnx_error)?;
        let output = pose_lift_output_tensor(&outputs)?;
        decode_pose_lift_tensor(output, sequence)
    }
}

#[derive(Debug, Clone)]
/// Data type for ONNX pose lifter.
pub struct OnnxPoseLifter<R = UnavailablePoseLiftRunner> {
    options: OnnxPoseLifterOptions,
    runner: R,
}

#[cfg(not(feature = "onnx"))]
impl OnnxPoseLifter<UnavailablePoseLiftRunner> {
    /// Builds this value from bundle.
    pub fn from_bundle(bundle: model_runtime::ModelBundle) -> Result<Self> {
        validate_onnx_pose_bundle(&bundle, model_runtime::ModelTask::PoseLifting3d)?;
        Ok(Self {
            options: OnnxPoseLifterOptions::default(),
            runner: UnavailablePoseLiftRunner,
        })
    }
}

#[cfg(feature = "onnx")]
impl OnnxPoseLifter<runtime_onnx::OnnxSession> {
    /// Builds this value from bundle.
    pub fn from_bundle(bundle: model_runtime::ModelBundle) -> Result<Self> {
        let info = validate_onnx_pose_bundle(&bundle, model_runtime::ModelTask::PoseLifting3d)?;
        Ok(Self {
            options: OnnxPoseLifterOptions::default(),
            runner: runtime_onnx::OnnxSession::from_file(info.model_path)
                .map_err(runtime_onnx_error)?,
        })
    }
}

impl<R: OnnxPoseLiftRunner> OnnxPoseLifter<R> {
    /// Builds this value from runner.
    pub fn from_runner(bundle: model_runtime::ModelBundle, runner: R) -> Result<Self> {
        validate_onnx_pose_bundle(&bundle, model_runtime::ModelTask::PoseLifting3d)?;
        Ok(Self {
            options: OnnxPoseLifterOptions::default(),
            runner,
        })
    }

    /// Returns this value with options.
    pub fn with_options(options: OnnxPoseLifterOptions, runner: R) -> Result<Self> {
        validate_threshold(options.min_pose_score)?;
        Ok(Self { options, runner })
    }
}

impl<R: OnnxPoseLiftRunner> PostureLiftBackend for OnnxPoseLifter<R> {
    fn lift_sequence(
        &mut self,
        sequence: &PoseSequence<PoseEstimate>,
    ) -> Result<PoseSequence<Pose3dEstimate>> {
        let mut lifted = self.runner.run_pose_lift(sequence)?;
        lifted
            .frames
            .retain(|pose| pose.score.unwrap_or(1.0) >= self.options.min_pose_score);
        Ok(lifted)
    }
}

#[cfg(any(feature = "onnx", test))]
fn decode_pose_2d_tensor(
    tensor: &runtime_onnx::OnnxF32Tensor,
    skeleton: Option<&Skeleton>,
    scores: Option<&[f32]>,
) -> Result<Vec<PoseEstimate>> {
    let (pose_count, joint_count) = match tensor.shape.as_slice() {
        [poses, joints, 3] => (*poses, *joints),
        [1, poses, joints, 3] => (*poses, *joints),
        [batch, _, _, 3] => {
            return Err(invalid_argument(format!(
                "ONNX pose 2D output batch size must be 1, got {batch}"
            )))
        }
        shape => {
            return Err(invalid_argument(format!(
                "unsupported ONNX pose 2D output shape `{shape:?}`"
            )))
        }
    };
    if let Some(skeleton) = skeleton {
        if skeleton.keypoints.len() != joint_count {
            return Err(invalid_argument(format!(
                "ONNX pose 2D output joint count {joint_count} does not match skeleton joint count {}",
                skeleton.keypoints.len()
            )));
        }
    }
    let expected = pose_count
        .checked_mul(joint_count)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| invalid_argument("ONNX pose 2D output shape is too large"))?;
    if tensor.values.len() != expected {
        return Err(invalid_argument(
            "ONNX pose 2D output values do not match shape",
        ));
    }
    if let Some(scores) = scores {
        if scores.len() < pose_count {
            return Err(invalid_argument(
                "ONNX pose 2D scores output is shorter than pose count",
            ));
        }
    }

    let mut poses = Vec::with_capacity(pose_count);
    for pose_index in 0..pose_count {
        let mut keypoints = Vec::with_capacity(joint_count);
        for joint_index in 0..joint_count {
            let start = (pose_index * joint_count + joint_index) * 3;
            let name = skeleton
                .and_then(|skeleton| skeleton.keypoints.get(joint_index))
                .cloned()
                .unwrap_or_else(|| format!("keypoint_{joint_index}"));
            keypoints.push(
                Keypoint::new(name, tensor.values[start], tensor.values[start + 1])?
                    .score(tensor.values[start + 2])?,
            );
        }
        let mut pose = PoseEstimate::new(keypoints)?;
        if let Some(scores) = scores {
            pose = pose.score(scores[pose_index])?;
        }
        poses.push(pose);
    }
    Ok(poses)
}

#[cfg(any(feature = "onnx", test))]
#[derive(Debug, Clone)]
struct YoloV8PoseDetection {
    pose: PoseEstimate,
    score: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

#[cfg(any(feature = "onnx", test))]
fn decode_yolov8_pose_tensor(
    tensor: &runtime_onnx::OnnxF32Tensor,
    skeleton: &Skeleton,
    input_width: u32,
    input_height: u32,
) -> Result<Vec<PoseEstimate>> {
    const DETECTION_LEN: usize = 56;
    const KEYPOINT_COUNT: usize = 17;
    const DETECTION_THRESHOLD: f32 = 0.30;
    const NMS_IOU_THRESHOLD: f32 = 0.50;

    if input_width == 0 || input_height == 0 {
        return Err(invalid_argument(
            "YOLOv8 pose input dimensions must be positive",
        ));
    }
    let detections = transpose_yolov8_pose_output(tensor)?;
    let mut decoded = Vec::new();
    for detection in detections {
        if detection.len() != DETECTION_LEN {
            return Err(invalid_argument(format!(
                "YOLOv8 pose detection length must be {DETECTION_LEN}, got {}",
                detection.len()
            )));
        }
        if !detection.iter().all(|value| value.is_finite()) {
            continue;
        }
        let score = detection[4];
        if score < DETECTION_THRESHOLD {
            continue;
        }
        let xc = detection[0];
        let yc = detection[1];
        let w = detection[2].max(0.0);
        let h = detection[3].max(0.0);
        let x1 = xc - w / 2.0;
        let y1 = yc - h / 2.0;
        let x2 = xc + w / 2.0;
        let y2 = yc + h / 2.0;
        if ![x1, y1, x2, y2].iter().all(|value| value.is_finite()) || x2 <= x1 || y2 <= y1 {
            continue;
        }

        let mut keypoints = Vec::with_capacity(KEYPOINT_COUNT);
        for keypoint_index in 0..KEYPOINT_COUNT {
            let start = 5 + keypoint_index * 3;
            let name = skeleton
                .keypoints
                .get(keypoint_index)
                .cloned()
                .unwrap_or_else(|| format!("keypoint_{keypoint_index}"));
            keypoints.push(
                Keypoint::new(
                    name,
                    (detection[start] / input_width as f32).clamp(0.0, 1.0),
                    (detection[start + 1] / input_height as f32).clamp(0.0, 1.0),
                )?
                .score(detection[start + 2])?,
            );
        }

        decoded.push(YoloV8PoseDetection {
            pose: PoseEstimate::new(keypoints)?.score(score)?,
            score,
            x1,
            y1,
            x2,
            y2,
        });
    }

    let mut retained = non_max_suppression(decoded, NMS_IOU_THRESHOLD);
    retained.sort_by(|left, right| right.score.total_cmp(&left.score));
    Ok(retained
        .into_iter()
        .map(|detection| detection.pose)
        .collect())
}

#[cfg(any(feature = "onnx", test))]
fn transpose_yolov8_pose_output(tensor: &runtime_onnx::OnnxF32Tensor) -> Result<Vec<Vec<f32>>> {
    const DETECTION_LEN: usize = 56;
    let (detection_count, channel_first) = match tensor.shape.as_slice() {
        [1, DETECTION_LEN, detections] => (*detections, true),
        [1, detections, DETECTION_LEN] => (*detections, false),
        shape => {
            return Err(invalid_argument(format!(
                "unsupported YOLOv8 pose output shape `{shape:?}`"
            )))
        }
    };
    let expected = detection_count
        .checked_mul(DETECTION_LEN)
        .ok_or_else(|| invalid_argument("YOLOv8 pose output shape is too large"))?;
    if tensor.values.len() != expected {
        return Err(invalid_argument(
            "YOLOv8 pose output values do not match shape",
        ));
    }

    if !channel_first {
        return Ok(tensor
            .values
            .chunks(DETECTION_LEN)
            .map(|chunk| chunk.to_vec())
            .collect());
    }

    let mut detections = vec![vec![0.0; DETECTION_LEN]; detection_count];
    for channel in 0..DETECTION_LEN {
        let start = channel * detection_count;
        let end = start + detection_count;
        for (row, value) in detections.iter_mut().zip(&tensor.values[start..end]) {
            row[channel] = *value;
        }
    }
    Ok(detections)
}

#[cfg(any(feature = "onnx", test))]
fn yolov8_pose_iou(left: &YoloV8PoseDetection, right: &YoloV8PoseDetection) -> f32 {
    let x_overlap = (left.x2.min(right.x2) - left.x1.max(right.x1)).max(0.0);
    let y_overlap = (left.y2.min(right.y2) - left.y1.max(right.y1)).max(0.0);
    let overlap = x_overlap * y_overlap;
    let left_area = (left.x2 - left.x1).max(0.0) * (left.y2 - left.y1).max(0.0);
    let right_area = (right.x2 - right.x1).max(0.0) * (right.y2 - right.y1).max(0.0);
    let union = left_area + right_area - overlap;
    if union <= 0.0 {
        0.0
    } else {
        overlap / union
    }
}

#[cfg(any(feature = "onnx", test))]
fn non_max_suppression(
    mut detections: Vec<YoloV8PoseDetection>,
    iou_threshold: f32,
) -> Vec<YoloV8PoseDetection> {
    detections.sort_by(|left, right| right.score.total_cmp(&left.score));
    let mut retained = Vec::new();
    'candidate: for detection in detections {
        for existing in &retained {
            if yolov8_pose_iou(&detection, existing) > iou_threshold {
                continue 'candidate;
            }
        }
        retained.push(detection);
    }
    retained
}

fn validate_and_apply_pose_skeleton(poses: &mut [PoseEstimate], skeleton: &Skeleton) -> Result<()> {
    for pose in poses {
        if pose.keypoints.len() != skeleton.keypoints.len() {
            return Err(invalid_argument(format!(
                "ONNX pose output joint count {} does not match skeleton joint count {}",
                pose.keypoints.len(),
                skeleton.keypoints.len()
            )));
        }
        for (keypoint, name) in pose.keypoints.iter_mut().zip(&skeleton.keypoints) {
            keypoint.name = name.clone();
        }
    }
    Ok(())
}

#[cfg(feature = "onnx")]
fn pose_lift_input_tensor(
    sequence: &PoseSequence<PoseEstimate>,
) -> Result<runtime_onnx::OnnxF32Tensor> {
    let first = sequence
        .frames
        .first()
        .ok_or_else(|| invalid_argument("ONNX pose lifter requires a non-empty sequence"))?;
    let joint_count = first.keypoints.len();
    if joint_count == 0 {
        return Err(invalid_argument(
            "ONNX pose lifter requires at least one keypoint",
        ));
    }
    let mut values = Vec::with_capacity(sequence.frames.len() * joint_count * 3);
    for pose in &sequence.frames {
        if pose.keypoints.len() != joint_count {
            return Err(invalid_argument(
                "ONNX pose lifter input poses must have the same joint count",
            ));
        }
        for keypoint in &pose.keypoints {
            values.push(keypoint.x);
            values.push(keypoint.y);
            values.push(keypoint.score.unwrap_or(1.0));
        }
    }
    runtime_onnx::OnnxF32Tensor::new(vec![sequence.frames.len(), joint_count, 3], values)
        .map_err(runtime_onnx_error)
}

#[cfg(any(feature = "onnx", test))]
fn decode_pose_lift_tensor(
    tensor: &runtime_onnx::OnnxF32Tensor,
    source: &PoseSequence<PoseEstimate>,
) -> Result<PoseSequence<Pose3dEstimate>> {
    let (frame_count, joint_count) = match tensor.shape.as_slice() {
        [frames, joints, 3] => (*frames, *joints),
        [1, frames, joints, 3] => (*frames, *joints),
        [batch, _, _, 3] => {
            return Err(invalid_argument(format!(
                "ONNX pose lift output batch size must be 1, got {batch}"
            )))
        }
        shape => {
            return Err(invalid_argument(format!(
                "unsupported ONNX pose lift output shape `{shape:?}`"
            )))
        }
    };
    if frame_count != source.frames.len() {
        return Err(invalid_argument(format!(
            "ONNX pose lift output frame count {frame_count} does not match input frame count {}",
            source.frames.len()
        )));
    }
    let expected = frame_count
        .checked_mul(joint_count)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| invalid_argument("ONNX pose lift output shape is too large"))?;
    if tensor.values.len() != expected {
        return Err(invalid_argument(
            "ONNX pose lift output values do not match shape",
        ));
    }
    let mut frames = Vec::with_capacity(frame_count);
    for frame_index in 0..frame_count {
        let source_pose = &source.frames[frame_index];
        if source_pose.keypoints.len() != joint_count {
            return Err(invalid_argument(format!(
                "ONNX pose lift output joint count {joint_count} does not match input joint count {}",
                source_pose.keypoints.len()
            )));
        }
        let mut keypoints = Vec::with_capacity(joint_count);
        for joint_index in 0..joint_count {
            let start = (frame_index * joint_count + joint_index) * 3;
            let source_keypoint = &source_pose.keypoints[joint_index];
            keypoints.push(Keypoint3d::new(
                source_keypoint.name.clone(),
                Point3::new(
                    tensor.values[start],
                    tensor.values[start + 1],
                    tensor.values[start + 2],
                ),
            )?);
        }
        let mut pose = Pose3dEstimate::new(keypoints)?;
        pose.id = source_pose.id.clone();
        pose.label = source_pose.label.clone();
        pose.score = source_pose.score;
        pose.attributes = source_pose.attributes.clone();
        frames.push(pose);
    }
    Ok(PoseSequence::new(frames))
}

#[cfg(any(feature = "onnx", test))]
fn pose_2d_output_tensor(
    outputs: &[runtime_onnx::OnnxNamedTensor],
) -> Result<&runtime_onnx::OnnxF32Tensor> {
    runtime_onnx::f32_output_by_preferred_name_or_index(
        outputs,
        &["keypoints", "poses", "output"],
        0,
    )
    .map_err(runtime_onnx_error)
}

#[cfg(any(feature = "onnx", test))]
fn pose_scores_output(
    outputs: &[runtime_onnx::OnnxNamedTensor],
) -> Option<&runtime_onnx::OnnxF32Tensor> {
    outputs.iter().find_map(|output| {
        output
            .name
            .to_ascii_lowercase()
            .contains("scores")
            .then_some(&output.tensor)
            .and_then(|tensor| match tensor {
                runtime_onnx::OnnxTensorValue::F32(tensor) => Some(tensor),
                _ => None,
            })
    })
}

#[cfg(any(feature = "onnx", test))]
fn pose_lift_output_tensor(
    outputs: &[runtime_onnx::OnnxNamedTensor],
) -> Result<&runtime_onnx::OnnxF32Tensor> {
    runtime_onnx::f32_output_by_preferred_name_or_index(
        outputs,
        &["keypoints", "poses", "output"],
        0,
    )
    .map_err(runtime_onnx_error)
}

/// Returns preprocess frame.
pub fn preprocess_frame_for_model(
    frame: &VideoFrame<'_>,
    options: &ImageModelPreprocessing,
) -> Result<ImageModelTensor> {
    let image = ImageView::from_video_frame(frame)?;
    preprocess_image_for_model(&image, options)
}

/// Validates ONNX pose bundle.
pub fn validate_onnx_pose_bundle(
    bundle: &model_runtime::ModelBundle,
    task: model_runtime::ModelTask,
) -> Result<OnnxPoseBundleInfo> {
    if bundle.manifest.task != task {
        return Err(DetectError::InvalidArgument(format!(
            "ONNX pose bundle {} task must be {:?}, got {:?}",
            bundle_descriptor(bundle),
            task,
            bundle.manifest.task
        )));
    }
    let config_path = required_bundle_file(bundle, "config.json")?;
    let preprocessor_config_path = bundle.file_path("preprocessor_config.json");
    let onnx_files = bundle_files_with_extension(bundle, "onnx");
    let model_path = match onnx_files.as_slice() {
        [path] => path.clone(),
        [] => {
            return Err(DetectError::InvalidArgument(format!(
                "ONNX pose bundle {} must contain exactly one `.onnx` model file selected by extension, found 0",
                bundle_descriptor(bundle)
            )))
        }
        files => {
            return Err(DetectError::InvalidArgument(format!(
                "ONNX pose bundle {} must contain exactly one `.onnx` model file selected by extension, found {}",
                bundle_descriptor(bundle),
                files.len()
            )))
        }
    };
    Ok(OnnxPoseBundleInfo {
        config_path,
        preprocessor_config_path,
        model_path,
    })
}

/// Returns pose 2d options from bundle.
pub fn pose_2d_options_from_bundle(
    bundle: &model_runtime::ModelBundle,
) -> Result<OnnxPose2dOptions> {
    let info = validate_onnx_pose_bundle(bundle, model_runtime::ModelTask::PoseEstimation2d)?;
    let config = read_json(&info.config_path)?;
    let preprocessing = if let Some(path) = &info.preprocessor_config_path {
        image_model_preprocessing_from_config(&read_json(path)?)?
    } else {
        ImageModelPreprocessing::default()
    };
    validate_image_model_preprocessing(&preprocessing)?;
    Ok(OnnxPose2dOptions {
        preprocessing,
        skeleton: skeleton_from_config(&config),
        output_space: KeypointSpace::Normalized,
        min_pose_score: 0.0,
        min_keypoint_score: 0.0,
    })
}

fn skeleton_from_config(config: &serde_json::Value) -> Skeleton {
    if let Some(names) = config
        .get("video_analysis")
        .and_then(|value| value.get("keypoint_names"))
        .or_else(|| config.get("keypoint_names"))
        .and_then(serde_json::Value::as_array)
    {
        let keypoints = names
            .iter()
            .filter_map(serde_json::Value::as_str)
            .map(str::to_string)
            .collect::<Vec<_>>();
        if !keypoints.is_empty() {
            return Skeleton::new(keypoints);
        }
    }
    if let Some(id2label) = config
        .get("id2label")
        .and_then(serde_json::Value::as_object)
    {
        let mut labels = id2label
            .iter()
            .filter_map(|(key, value)| {
                let index = key.parse::<usize>().ok()?;
                let label = value.as_str()?;
                Some((index, label.to_string()))
            })
            .collect::<Vec<_>>();
        labels.sort_by_key(|(index, _)| *index);
        let keypoints = labels
            .into_iter()
            .map(|(_, label)| label)
            .collect::<Vec<_>>();
        if !keypoints.is_empty() {
            return Skeleton::new(keypoints);
        }
    }
    Skeleton::coco_17()
}

fn validate_threshold(value: f32) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(DetectError::InvalidArgument(
            "ONNX score threshold must be finite and in the range 0..=1".to_string(),
        ));
    }
    Ok(())
}

fn required_bundle_file(bundle: &model_runtime::ModelBundle, remote_path: &str) -> Result<PathBuf> {
    bundle.file_path(remote_path).ok_or_else(|| {
        DetectError::InvalidArgument(format!(
            "model bundle {} is missing required file `{remote_path}`",
            bundle_descriptor(bundle)
        ))
    })
}

fn bundle_descriptor(bundle: &model_runtime::ModelBundle) -> String {
    format!(
        "`{}` (task {:?})",
        bundle.manifest.name, bundle.manifest.task
    )
}

fn bundle_files_with_extension(
    bundle: &model_runtime::ModelBundle,
    extension: &str,
) -> Vec<PathBuf> {
    bundle
        .manifest
        .files
        .values()
        .filter(|file| {
            Path::new(&file.remote_path)
                .extension()
                .and_then(|value| value.to_str())
                == Some(extension)
        })
        .map(|file| bundle.root.join(&file.local_path))
        .collect()
}

fn read_json(path: &Path) -> Result<serde_json::Value> {
    let data = std::fs::read(path)?;
    serde_json::from_slice(&data).map_err(|err| {
        DetectError::Source(format!("failed to parse JSON `{}`: {err}", path.display()))
    })
}

#[cfg(any(feature = "onnx", test))]
fn runtime_onnx_error(error: runtime_onnx::OnnxRuntimeError) -> DetectError {
    match error {
        runtime_onnx::OnnxRuntimeError::InvalidArgument(message)
        | runtime_onnx::OnnxRuntimeError::InvalidTensorShape(message) => invalid_argument(message),
        runtime_onnx::OnnxRuntimeError::Io(error) => DetectError::Io(error),
        other => DetectError::Source(other.to_string()),
    }
}

fn invalid_argument(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(message.into())
}

#[cfg(test)]
mod tests {
    use num_rational::Rational64;
    use video_analysis_core::{FramePosition, OwnedVideoFrame, PixelFormat};

    use super::*;

    fn frame() -> OwnedVideoFrame {
        OwnedVideoFrame {
            position: FramePosition::from_frame_index(0, Rational64::new(30, 1)),
            width: 100,
            height: 100,
            pixel_format: PixelFormat::Rgb24,
            data: vec![0; 100 * 100 * 3],
            stride: 100 * 3,
        }
    }

    fn f32_tensor(shape: Vec<usize>, values: Vec<f32>) -> runtime_onnx::OnnxF32Tensor {
        runtime_onnx::OnnxF32Tensor::new(shape, values).unwrap()
    }

    fn yolov8_detection(score: f32, xc: f32, yc: f32) -> Vec<f32> {
        let mut values = vec![xc, yc, 100.0, 200.0, score];
        for index in 0..17 {
            values.push(64.0 + index as f32);
            values.push(128.0 + index as f32);
            values.push(0.5 + index as f32 * 0.01);
        }
        values
    }

    fn yolov8_channel_first(rows: &[Vec<f32>]) -> Vec<f32> {
        let mut values = Vec::with_capacity(rows.len() * 56);
        for channel in 0..56 {
            for row in rows {
                values.push(row[channel]);
            }
        }
        values
    }

    #[test]
    fn joint_angle_reports_right_angle() {
        let a = Keypoint::new("a", 0.0, 0.0).unwrap();
        let b = Keypoint::new("b", 0.0, 1.0).unwrap();
        let c = Keypoint::new("c", 1.0, 1.0).unwrap();

        let angle = joint_angle_degrees(&a, &b, &c).unwrap();

        assert!((angle - 90.0).abs() < 0.001);
    }

    #[test]
    fn analyzer_emits_pose_observation() {
        struct Backend;

        impl PostureBackend for Backend {
            fn estimate_frame(&mut self, _frame: &VideoFrame<'_>) -> Result<Vec<PoseEstimate>> {
                Ok(vec![PoseEstimate::new([
                    Keypoint::new("left_shoulder", 10.0, 20.0)?.score(0.9)?,
                    Keypoint::new("right_shoulder", 30.0, 20.0)?.score(0.8)?,
                ])?
                .id("pose-1")
                .score(0.95)?])
            }
        }

        let owned = frame();
        let frame = owned.as_frame();
        let mut analyzer = PostureAnalyzer::new("posture", Backend);
        let observations = analyzer.process_frame(&frame).unwrap();
        assert_eq!(observations.len(), 1);
        assert_eq!(observations[0].track_id.as_deref(), Some("pose-1"));
        assert_eq!(observations[0].attributes["keypoint_count"], "2");
    }

    #[test]
    fn analyzer_filters_low_score_keypoints() {
        struct Backend;

        impl PostureBackend for Backend {
            fn estimate_frame(&mut self, _frame: &VideoFrame<'_>) -> Result<Vec<PoseEstimate>> {
                Ok(vec![PoseEstimate::new([
                    Keypoint::new("left_shoulder", 10.0, 20.0)?.score(0.2)?,
                    Keypoint::new("right_shoulder", 30.0, 20.0)?.score(0.3)?,
                ])?
                .score(0.95)?])
            }
        }

        let owned = frame();
        let frame = owned.as_frame();
        let mut analyzer = PostureAnalyzer::new("posture", Backend)
            .options(PostureOptions {
                min_keypoint_score: Some(0.5),
                ..PostureOptions::default()
            })
            .unwrap();
        let observations = analyzer.process_frame(&frame).unwrap();
        assert!(observations.is_empty());
    }

    #[test]
    fn keypoints_convert_through_shared_geometry() {
        let keypoint = Keypoint::new("nose", 32.0, 24.0).unwrap();
        let size = Size2u::new(64, 48).unwrap();
        let normalized = keypoint.to_normalized_point2(size).unwrap();
        let round_trip = Keypoint::from_normalized_point2("nose", normalized, size).unwrap();
        assert!((round_trip.x - keypoint.x).abs() < 1.0e-6);
        assert!((round_trip.y - keypoint.y).abs() < 1.0e-6);
    }

    #[test]
    fn three_d_posture_helpers_cover_normalization_interpolation_and_smoothing() {
        let skeleton = Skeleton::coco_17();
        let pose = Pose3dEstimate::new([
            Keypoint3d::new("left_shoulder", Point3::new(0.0, 0.0, 0.0)).unwrap(),
            Keypoint3d::new("right_shoulder", Point3::new(1.0, 0.0, 0.0)).unwrap(),
            Keypoint3d::new("left_elbow", Point3::new(0.0, -1.0, 0.0)).unwrap(),
        ])
        .unwrap();
        let normalized = normalize_pose3d(&pose, "left_shoulder").unwrap();
        assert_eq!(
            normalized.keypoint("left_shoulder").unwrap().position,
            Point3::new(0.0, 0.0, 0.0)
        );

        let interpolated = interpolate_missing_joints(&pose, &skeleton).unwrap();
        assert!(interpolated.keypoint("right_elbow").is_some());

        let sequence = PoseSequence::new([pose.clone(), normalized.clone()]);
        let smoothed = smooth_pose_sequence(&sequence, 1).unwrap();
        assert_eq!(smoothed.len(), 2);

        let figure = pose.to_stick_figure(skeleton).unwrap();
        assert!(!figure.segments().unwrap().is_empty());
    }

    #[test]
    fn three_d_joint_angle_and_bone_lengths_are_reported() {
        let a = Keypoint3d::new("a", Point3::new(0.0, 0.0, 0.0)).unwrap();
        let b = Keypoint3d::new("b", Point3::new(0.0, 1.0, 0.0)).unwrap();
        let c = Keypoint3d::new("c", Point3::new(1.0, 1.0, 0.0)).unwrap();
        let angle = joint_angle_3d_degrees(&a, &b, &c).unwrap();
        assert!((angle - 90.0).abs() < 0.001);

        let skeleton = Skeleton::new(["a", "b", "c"]).edge("a", "b").edge("b", "c");
        let pose = Pose3dEstimate::new([a, b, c]).unwrap();
        let lengths = bone_lengths(&pose, &skeleton).unwrap();
        assert_eq!(lengths.len(), 2);
    }

    #[test]
    fn decodes_pose_2d_tensor_layout() {
        let skeleton = Skeleton::new(["left", "right"]);
        let tensor = f32_tensor(vec![1, 2, 3], vec![0.1, 0.2, 0.9, 0.3, 0.4, 0.8]);
        let poses = decode_pose_2d_tensor(&tensor, Some(&skeleton), Some(&[0.95])).unwrap();
        assert_eq!(poses.len(), 1);
        assert_eq!(poses[0].score, Some(0.95));
        assert_eq!(poses[0].keypoints[0].name, "left");
        assert_eq!(poses[0].keypoints[1].score, Some(0.8));
    }

    #[test]
    fn decodes_pose_2d_batched_tensor_layout() {
        let tensor = f32_tensor(vec![1, 1, 2, 3], vec![0.1, 0.2, 0.9, 0.3, 0.4, 0.8]);
        let poses = decode_pose_2d_tensor(&tensor, None, None).unwrap();
        assert_eq!(poses.len(), 1);
        assert_eq!(poses[0].keypoints.len(), 2);
    }

    #[test]
    fn skeleton_from_config_reads_numeric_id2label() {
        let config = serde_json::json!({
            "id2label": {
                "2": "right_eye",
                "0": "nose",
                "1": "left_eye"
            }
        });

        let skeleton = skeleton_from_config(&config);

        assert_eq!(skeleton.keypoints, vec!["nose", "left_eye", "right_eye"]);
    }

    #[test]
    fn decodes_yolov8_pose_output_channel_first_layout() {
        let rows = vec![yolov8_detection(0.9, 320.0, 320.0)];
        let tensor = f32_tensor(vec![1, 56, 1], yolov8_channel_first(&rows));

        let poses = decode_yolov8_pose_tensor(&tensor, &Skeleton::coco_17(), 640, 640).unwrap();

        assert_eq!(poses.len(), 1);
        assert_eq!(poses[0].score, Some(0.9));
        assert_eq!(poses[0].keypoints.len(), 17);
        assert_eq!(poses[0].keypoints[0].name, "nose");
        assert!((poses[0].keypoints[0].x - 0.1).abs() < 1.0e-6);
        assert!((poses[0].keypoints[0].y - 0.2).abs() < 1.0e-6);
    }

    #[test]
    fn decodes_yolov8_pose_output_detection_first_layout() {
        let rows = [yolov8_detection(0.8, 300.0, 300.0)];
        let tensor = f32_tensor(vec![1, 1, 56], rows.concat());

        let poses = decode_yolov8_pose_tensor(&tensor, &Skeleton::coco_17(), 640, 640).unwrap();

        assert_eq!(poses.len(), 1);
        assert_eq!(poses[0].score, Some(0.8));
        assert_eq!(poses[0].keypoints[16].name, "right_ankle");
    }

    #[test]
    fn yolov8_pose_decoder_filters_low_confidence_detections() {
        let rows = [
            yolov8_detection(0.29, 320.0, 320.0),
            yolov8_detection(0.31, 120.0, 120.0),
        ];
        let tensor = f32_tensor(vec![1, 2, 56], rows.concat());

        let poses = decode_yolov8_pose_tensor(&tensor, &Skeleton::coco_17(), 640, 640).unwrap();

        assert_eq!(poses.len(), 1);
        assert_eq!(poses[0].score, Some(0.31));
    }

    #[test]
    fn yolov8_pose_decoder_removes_duplicate_boxes_by_iou() {
        let rows = [
            yolov8_detection(0.7, 320.0, 320.0),
            yolov8_detection(0.95, 322.0, 322.0),
            yolov8_detection(0.8, 100.0, 100.0),
        ];
        let tensor = f32_tensor(vec![1, 3, 56], rows.concat());

        let poses = decode_yolov8_pose_tensor(&tensor, &Skeleton::coco_17(), 640, 640).unwrap();

        assert_eq!(poses.len(), 2);
        assert_eq!(poses[0].score, Some(0.95));
        assert_eq!(poses[1].score, Some(0.8));
    }

    #[test]
    fn pose_2d_output_selection_prefers_named_keypoints() {
        let outputs = vec![
            runtime_onnx::OnnxNamedTensor {
                name: "scores".to_string(),
                tensor: runtime_onnx::OnnxTensorValue::F32(
                    runtime_onnx::OnnxF32Tensor::new(vec![1], vec![0.95]).unwrap(),
                ),
            },
            runtime_onnx::OnnxNamedTensor {
                name: "keypoints".to_string(),
                tensor: runtime_onnx::OnnxTensorValue::F32(f32_tensor(
                    vec![1, 1, 3],
                    vec![0.1, 0.2, 0.9],
                )),
            },
        ];
        assert_eq!(
            pose_2d_output_tensor(&outputs).unwrap().shape,
            vec![1, 1, 3]
        );
        assert_eq!(pose_scores_output(&outputs).unwrap().values, vec![0.95]);
    }

    #[test]
    fn pose_estimator_validates_and_applies_skeleton() {
        struct Backend;

        impl OnnxPose2dRunner for Backend {
            fn run_pose_2d(&mut self, _input: &ImageModelTensor) -> Result<Vec<PoseEstimate>> {
                Ok(vec![PoseEstimate::new([
                    Keypoint::new("a", 0.1, 0.2)?.score(0.9)?,
                    Keypoint::new("b", 0.3, 0.4)?.score(0.8)?,
                ])?])
            }
        }

        let options = OnnxPose2dOptions {
            preprocessing: ImageModelPreprocessing {
                input_width: 100,
                input_height: 100,
                ..ImageModelPreprocessing::default()
            },
            skeleton: Skeleton::new(["hip", "knee"]),
            ..OnnxPose2dOptions::default()
        };
        let owned = frame();
        let mut estimator = OnnxPose2dEstimator::with_options(options, Backend).unwrap();
        let poses = estimator.predict_frame(&owned.as_frame()).unwrap();
        assert_eq!(poses[0].keypoints[0].name, "hip");
        assert_eq!(poses[0].keypoints[1].name, "knee");
    }

    #[test]
    fn pose_estimator_applies_pose_and_keypoint_thresholds_after_validation() {
        struct Backend;

        impl OnnxPose2dRunner for Backend {
            fn run_pose_2d(&mut self, _input: &ImageModelTensor) -> Result<Vec<PoseEstimate>> {
                Ok(vec![
                    PoseEstimate::new([
                        Keypoint::new("a", 0.1, 0.2)?.score(0.9)?,
                        Keypoint::new("b", 0.3, 0.4)?.score(0.9)?,
                    ])?
                    .score(0.4)?,
                    PoseEstimate::new([
                        Keypoint::new("a", 0.1, 0.2)?.score(0.2)?,
                        Keypoint::new("b", 0.3, 0.4)?.score(0.8)?,
                    ])?
                    .score(0.9)?,
                ])
            }
        }

        let options = OnnxPose2dOptions {
            skeleton: Skeleton::new(["hip", "knee"]),
            min_pose_score: 0.5,
            min_keypoint_score: 0.5,
            ..OnnxPose2dOptions::default()
        };
        let owned = frame();
        let mut estimator = OnnxPose2dEstimator::with_options(options, Backend).unwrap();
        let poses = estimator.predict_frame(&owned.as_frame()).unwrap();
        assert_eq!(poses.len(), 1);
        assert_eq!(poses[0].keypoints.len(), 1);
        assert_eq!(poses[0].keypoints[0].name, "knee");
    }

    #[test]
    fn pose_2d_tensor_rejects_unsupported_shape() {
        let tensor = f32_tensor(vec![2, 2], vec![0.0; 4]);
        assert!(decode_pose_2d_tensor(&tensor, None, None).is_err());
    }

    #[test]
    fn pose_2d_tensor_rejects_batch_larger_than_one() {
        let tensor = f32_tensor(vec![2, 1, 1, 3], vec![0.0; 6]);
        let err = decode_pose_2d_tensor(&tensor, None, None).unwrap_err();
        assert!(err.to_string().contains("batch size"));
    }

    #[test]
    fn decodes_pose_lift_tensor_and_preserves_source_metadata() {
        let source_pose = PoseEstimate::new([
            Keypoint::new("hip", 0.1, 0.2).unwrap().score(0.9).unwrap(),
            Keypoint::new("knee", 0.3, 0.4).unwrap().score(0.8).unwrap(),
        ])
        .unwrap()
        .id("pose-1")
        .label("person")
        .attribute("timestamp", "00:00:01");
        let source = PoseSequence::new([source_pose]);
        let tensor = f32_tensor(vec![1, 2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let lifted = decode_pose_lift_tensor(&tensor, &source).unwrap();
        assert_eq!(lifted.frames[0].id.as_deref(), Some("pose-1"));
        assert_eq!(lifted.frames[0].label.as_deref(), Some("person"));
        assert_eq!(lifted.frames[0].attributes["timestamp"], "00:00:01");
        assert_eq!(
            lifted.frames[0].keypoint("knee").unwrap().position,
            Point3::new(4.0, 5.0, 6.0)
        );
    }

    #[test]
    fn decodes_pose_lift_batched_tensor_layout() {
        let source = PoseSequence::new([PoseEstimate::new([
            Keypoint::new("hip", 0.1, 0.2).unwrap(),
            Keypoint::new("knee", 0.3, 0.4).unwrap(),
        ])
        .unwrap()]);
        let tensor = f32_tensor(vec![1, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let lifted = decode_pose_lift_tensor(&tensor, &source).unwrap();
        assert_eq!(
            lifted.frames[0].keypoint("knee").unwrap().position,
            Point3::new(4.0, 5.0, 6.0)
        );
    }

    #[test]
    fn pose_lift_output_selection_prefers_named_output() {
        let outputs = vec![
            runtime_onnx::OnnxNamedTensor {
                name: "aux".to_string(),
                tensor: runtime_onnx::OnnxTensorValue::F32(
                    runtime_onnx::OnnxF32Tensor::new(vec![1], vec![0.0]).unwrap(),
                ),
            },
            runtime_onnx::OnnxNamedTensor {
                name: "output_poses".to_string(),
                tensor: runtime_onnx::OnnxTensorValue::F32(f32_tensor(
                    vec![1, 1, 3],
                    vec![1.0, 2.0, 3.0],
                )),
            },
        ];
        assert_eq!(
            pose_lift_output_tensor(&outputs).unwrap().shape,
            vec![1, 1, 3]
        );
    }

    #[test]
    fn pose_lift_tensor_rejects_unsupported_shape() {
        let source = PoseSequence::new([PoseEstimate::new([
            Keypoint::new("hip", 0.1, 0.2).unwrap()
        ])
        .unwrap()]);
        let tensor = f32_tensor(vec![1, 3], vec![0.0; 3]);
        assert!(decode_pose_lift_tensor(&tensor, &source).is_err());
    }

    #[test]
    fn pose_lift_tensor_rejects_batch_larger_than_one() {
        let source = PoseSequence::new([PoseEstimate::new([
            Keypoint::new("hip", 0.1, 0.2).unwrap()
        ])
        .unwrap()]);
        let tensor = f32_tensor(vec![2, 1, 1, 3], vec![0.0; 6]);
        let err = decode_pose_lift_tensor(&tensor, &source).unwrap_err();
        assert!(err.to_string().contains("batch size"));
    }
}
