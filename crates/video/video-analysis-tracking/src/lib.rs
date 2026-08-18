#![doc = include_str!("../README.md")]

pub mod surface;
use std::collections::{BTreeMap, BTreeSet};

use math_geometry_2d::{
    broad_phase_pairs_2d, BroadPhase2Config, BroadPhase2Strategy, RectU32, SpatialCellSize2,
};
use video_analysis_core::{
    BoundingBox, DetectError, FramePosition, Observation, ObservationKind, Result, VideoAnalyzer,
    VideoFrame,
};
use vision_core::{VisualDetection, VisualDetectionKind, VisualRegion};

#[derive(Debug, Clone, PartialEq)]
/// Data type for tracked detection.
pub struct TrackedDetection {
    /// The kind value.
    pub kind: ObservationKind,
    /// The region value.
    pub region: BoundingBox,
    /// Label assigned to this value.
    pub label: Option<String>,
    /// Score assigned to this value.
    pub score: Option<f32>,
    /// The track hint value.
    pub track_hint: Option<String>,
    /// The attributes value.
    pub attributes: BTreeMap<String, String>,
}

impl TrackedDetection {
    /// Creates a new value.
    pub fn new(region: BoundingBox) -> Self {
        Self {
            kind: ObservationKind::Object,
            region,
            label: None,
            score: None,
            track_hint: None,
            attributes: BTreeMap::new(),
        }
    }

    /// Returns kind.
    pub fn kind(mut self, kind: ObservationKind) -> Self {
        self.kind = kind;
        self
    }

    /// Returns label.
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Returns score.
    pub fn score(mut self, score: f32) -> Self {
        self.score = Some(score);
        self
    }

    /// Returns track hint.
    pub fn track_hint(mut self, track_hint: impl Into<String>) -> Self {
        self.track_hint = Some(track_hint.into());
        self
    }

    /// Returns attribute.
    pub fn attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// Converts this tracked detection into a shared visual detection.
    pub fn to_visual_detection(&self) -> Result<VisualDetection> {
        let kind = if self
            .label
            .as_deref()
            .is_some_and(|label| label.eq_ignore_ascii_case("person"))
        {
            VisualDetectionKind::Person
        } else {
            visual_kind_from_observation(&self.kind)
        };
        let mut visual =
            VisualDetection::new(kind, VisualRegion::from(self.region)).map_err(vision_error)?;
        if let Some(label) = &self.label {
            visual = visual.label(label.clone()).map_err(vision_error)?;
        }
        if let Some(score) = self.score {
            visual = visual.score(score).map_err(vision_error)?;
        }
        if let Some(track_hint) = &self.track_hint {
            visual = visual.id(track_hint.clone()).map_err(vision_error)?;
        }
        visual.attributes = self.attributes.clone();
        Ok(visual)
    }
}

impl TryFrom<TrackedDetection> for VisualDetection {
    type Error = DetectError;

    fn try_from(value: TrackedDetection) -> std::result::Result<Self, Self::Error> {
        value.to_visual_detection()
    }
}

impl TryFrom<&TrackedDetection> for VisualDetection {
    type Error = DetectError;

    fn try_from(value: &TrackedDetection) -> std::result::Result<Self, Self::Error> {
        value.to_visual_detection()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Options for object collision detection.
pub struct CollisionOptions {
    /// Minimum IoU required for two overlapping objects to count as a collision.
    pub min_iou: f32,
}

impl Default for CollisionOptions {
    fn default() -> Self {
        Self { min_iou: 0.0 }
    }
}

impl CollisionOptions {
    /// Validates this value.
    pub fn validate(self) -> Result<()> {
        if !self.min_iou.is_finite() || !(0.0..=1.0).contains(&self.min_iou) {
            return Err(DetectError::InvalidArgument(
                "collision min_iou must be finite and between 0.0 and 1.0".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Strategy used for broad-phase object collision detection.
pub enum CollisionBroadPhaseStrategy {
    /// Selects an implementation from the input shape.
    Auto,
    /// Checks every pair.
    BruteForce,
    /// Uses a spatial hash grid.
    SpatialHashGrid,
    /// Uses sweep and prune along the x axis.
    SweepAndPrune,
}

impl From<CollisionBroadPhaseStrategy> for BroadPhase2Strategy {
    fn from(value: CollisionBroadPhaseStrategy) -> Self {
        match value {
            CollisionBroadPhaseStrategy::Auto => Self::Auto,
            CollisionBroadPhaseStrategy::BruteForce => Self::BruteForce,
            CollisionBroadPhaseStrategy::SpatialHashGrid => Self::SpatialHashGrid,
            CollisionBroadPhaseStrategy::SweepAndPrune => Self::SweepAndPrune,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Cell size selection for broad-phase object collision detection.
pub enum CollisionCellSize {
    /// Uses median region dimensions.
    Auto,
    /// Uses a fixed cell size.
    Fixed {
        /// Cell width in pixels.
        width: u32,
        /// Cell height in pixels.
        height: u32,
    },
}

impl From<CollisionCellSize> for SpatialCellSize2 {
    fn from(value: CollisionCellSize) -> Self {
        match value {
            CollisionCellSize::Auto => Self::Auto,
            CollisionCellSize::Fixed { width, height } => Self::Fixed { width, height },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Options for broad-phase object collision detection.
pub struct CollisionBroadPhaseOptions {
    /// Strategy to use.
    pub strategy: CollisionBroadPhaseStrategy,
    /// Maximum region count handled with brute force in auto mode.
    pub brute_force_threshold: usize,
    /// Maximum cells a single region may span before auto mode uses sweep and prune.
    pub max_cells_per_region: usize,
    /// Spatial hash grid cell size.
    pub cell_size: CollisionCellSize,
}

impl Default for CollisionBroadPhaseOptions {
    fn default() -> Self {
        Self {
            strategy: CollisionBroadPhaseStrategy::Auto,
            brute_force_threshold: 128,
            max_cells_per_region: 1024,
            cell_size: CollisionCellSize::Auto,
        }
    }
}

impl CollisionBroadPhaseOptions {
    /// Validates this value.
    pub fn validate(self) -> Result<()> {
        Ok(to_broad_phase_config(self).validate()?)
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for a pair of colliding objects.
pub struct ObjectCollision {
    /// Index of the first object in the input collection.
    pub left_index: usize,
    /// Index of the second object in the input collection.
    pub right_index: usize,
    /// Track identifier for the first object, when available.
    pub left_id: Option<String>,
    /// Track identifier for the second object, when available.
    pub right_id: Option<String>,
    /// Region of the first object.
    pub left_region: BoundingBox,
    /// Region of the second object.
    pub right_region: BoundingBox,
    /// Overlapping region shared by both objects.
    pub intersection: BoundingBox,
    /// Intersection-over-union score for the two regions.
    pub iou: f32,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for object track.
pub struct ObjectTrack {
    /// Identifier for this value.
    pub id: String,
    /// The kind value.
    pub kind: ObservationKind,
    /// The region value.
    pub region: BoundingBox,
    /// Label assigned to this value.
    pub label: Option<String>,
    /// Score assigned to this value.
    pub score: Option<f32>,
    /// The first position value.
    pub first_position: FramePosition,
    /// The last position value.
    pub last_position: FramePosition,
    /// The age frames value.
    pub age_frames: u64,
    /// The missed frames value.
    pub missed_frames: u64,
    /// The attributes value.
    pub attributes: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Data type for tracking options.
pub struct TrackingOptions {
    /// The min IoU value.
    pub min_iou: f32,
    /// The max missed frames value.
    pub max_missed_frames: u64,
    /// The min score value.
    pub min_score: Option<f32>,
}

impl Default for TrackingOptions {
    fn default() -> Self {
        Self {
            min_iou: 0.3,
            max_missed_frames: 15,
            min_score: None,
        }
    }
}

impl TrackingOptions {
    /// Validates this value.
    pub fn validate(self) -> Result<()> {
        if !self.min_iou.is_finite() || !(0.0..=1.0).contains(&self.min_iou) {
            return Err(DetectError::InvalidArgument(
                "tracking min_iou must be finite and between 0.0 and 1.0".to_string(),
            ));
        }
        if let Some(min_score) = self.min_score {
            if !min_score.is_finite() {
                return Err(DetectError::InvalidArgument(
                    "tracking min_score must be finite".to_string(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
/// Data type for IoU tracker.
pub struct IouTracker {
    options: TrackingOptions,
    tracks: BTreeMap<String, ObjectTrack>,
    next_id: u64,
}

impl IouTracker {
    /// Creates a new value.
    pub fn new(options: TrackingOptions) -> Result<Self> {
        options.validate()?;
        Ok(Self {
            options,
            tracks: BTreeMap::new(),
            next_id: 1,
        })
    }

    /// Returns options.
    pub fn options(&self) -> TrackingOptions {
        self.options
    }

    /// Returns tracks.
    pub fn tracks(&self) -> impl Iterator<Item = &ObjectTrack> {
        self.tracks.values()
    }

    /// Returns collisions between active tracks.
    pub fn collisions(&self, options: CollisionOptions) -> Result<Vec<ObjectCollision>> {
        detect_track_collisions(self.tracks.values(), options)
    }

    /// Returns collisions between active tracks using explicit broad-phase options.
    pub fn collisions_with_broad_phase(
        &self,
        options: CollisionOptions,
        broad_phase: CollisionBroadPhaseOptions,
    ) -> Result<Vec<ObjectCollision>> {
        detect_track_collisions_with_broad_phase(self.tracks.values(), options, broad_phase)
    }

    /// Returns update.
    pub fn update(
        &mut self,
        position: FramePosition,
        detections: impl IntoIterator<Item = TrackedDetection>,
    ) -> Result<Vec<ObjectTrack>> {
        self.options.validate()?;
        let detections = detections
            .into_iter()
            .filter(|detection| {
                self.options
                    .min_score
                    .zip(detection.score)
                    .map(|(minimum, score)| score >= minimum)
                    .unwrap_or(true)
            })
            .collect::<Vec<_>>();

        let active_ids = self.tracks.keys().cloned().collect::<Vec<_>>();
        let mut assignments = vec![None::<String>; detections.len()];
        let mut used_tracks = BTreeSet::new();

        for (index, detection) in detections.iter().enumerate() {
            let Some(hint) = &detection.track_hint else {
                continue;
            };
            let Some(track) = self.tracks.get(hint) else {
                continue;
            };
            if !used_tracks.contains(hint) && compatible(track, detection) {
                assignments[index] = Some(hint.clone());
                used_tracks.insert(hint.clone());
            }
        }

        for (index, detection) in detections.iter().enumerate() {
            if assignments[index].is_some() {
                continue;
            }
            let mut best = None;
            for track_id in &active_ids {
                if used_tracks.contains(track_id) {
                    continue;
                }
                let Some(track) = self.tracks.get(track_id) else {
                    continue;
                };
                if !compatible(track, detection) {
                    continue;
                }
                let iou = bbox_iou(track.region, detection.region);
                if iou >= self.options.min_iou
                    && best
                        .as_ref()
                        .map(|(_, best_iou)| iou > *best_iou)
                        .unwrap_or(true)
                {
                    best = Some((track_id.clone(), iou));
                }
            }
            if let Some((track_id, _)) = best {
                assignments[index] = Some(track_id.clone());
                used_tracks.insert(track_id);
            }
        }

        let assigned = assignments
            .iter()
            .filter_map(|assignment| assignment.as_ref())
            .cloned()
            .collect::<BTreeSet<_>>();
        for track_id in active_ids {
            if !assigned.contains(&track_id) {
                if let Some(track) = self.tracks.get_mut(&track_id) {
                    track.missed_frames += 1;
                }
            }
        }

        let mut visible = Vec::new();
        for (detection, assignment) in detections.into_iter().zip(assignments) {
            let track_id = assignment.unwrap_or_else(|| self.allocate_track_id());
            let track = self
                .tracks
                .entry(track_id.clone())
                .or_insert_with(|| ObjectTrack {
                    id: track_id.clone(),
                    kind: detection.kind.clone(),
                    region: detection.region,
                    label: detection.label.clone(),
                    score: detection.score,
                    first_position: position,
                    last_position: position,
                    age_frames: 0,
                    missed_frames: 0,
                    attributes: BTreeMap::new(),
                });
            track.kind = detection.kind;
            track.region = detection.region;
            track.label = detection.label;
            track.score = detection.score;
            track.last_position = position;
            track.age_frames += 1;
            track.missed_frames = 0;
            track.attributes = detection.attributes;
            visible.push(track.clone());
        }

        self.tracks
            .retain(|_, track| track.missed_frames <= self.options.max_missed_frames);
        Ok(visible)
    }

    /// Returns reset.
    pub fn reset(&mut self) {
        self.tracks.clear();
        self.next_id = 1;
    }

    fn allocate_track_id(&mut self) -> String {
        let id = format!("track-{}", self.next_id);
        self.next_id += 1;
        id
    }
}

/// Trait for object detection backend implementations.
pub trait ObjectDetectionBackend {
    /// Returns detect frame.
    fn detect_frame(&mut self, frame: &VideoFrame<'_>) -> Result<Vec<TrackedDetection>>;
}

/// Data type for object tracking analyzer.
pub struct ObjectTrackingAnalyzer<B> {
    name: String,
    backend: B,
    tracker: IouTracker,
}

impl<B> ObjectTrackingAnalyzer<B> {
    /// Creates a new value.
    pub fn new(name: impl Into<String>, backend: B, options: TrackingOptions) -> Result<Self> {
        Ok(Self {
            name: name.into(),
            backend,
            tracker: IouTracker::new(options)?,
        })
    }

    /// Returns backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Returns backend mut.
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    /// Returns tracker.
    pub fn tracker(&self) -> &IouTracker {
        &self.tracker
    }

    /// Returns tracker mut.
    pub fn tracker_mut(&mut self) -> &mut IouTracker {
        &mut self.tracker
    }
}

impl<B: ObjectDetectionBackend> VideoAnalyzer for ObjectTrackingAnalyzer<B> {
    fn name(&self) -> &str {
        &self.name
    }

    fn process_frame(&mut self, frame: &VideoFrame<'_>) -> Result<Vec<Observation>> {
        let detections = self.backend.detect_frame(frame)?;
        let tracks = self.tracker.update(frame.position, detections)?;
        Ok(tracks
            .into_iter()
            .map(|track| observation_for_track(self.name(), track))
            .collect())
    }
}

/// Returns bounding box intersection.
pub fn bbox_intersection(left: BoundingBox, right: BoundingBox) -> Option<BoundingBox> {
    let left_x1 = left.x.saturating_add(left.width);
    let left_y1 = left.y.saturating_add(left.height);
    let right_x1 = right.x.saturating_add(right.width);
    let right_y1 = right.y.saturating_add(right.height);

    let ix0 = left.x.max(right.x);
    let iy0 = left.y.max(right.y);
    let ix1 = left_x1.min(right_x1);
    let iy1 = left_y1.min(right_y1);
    if ix1 <= ix0 || iy1 <= iy0 {
        return None;
    }

    BoundingBox::new(ix0, iy0, ix1 - ix0, iy1 - iy0).ok()
}

/// Returns whether two bounding boxes overlap.
pub fn bbox_intersects(left: BoundingBox, right: BoundingBox) -> bool {
    bbox_intersection(left, right).is_some()
}

/// Returns bbox IoU.
pub fn bbox_iou(left: BoundingBox, right: BoundingBox) -> f32 {
    let Some(intersection) = bbox_intersection(left, right) else {
        return 0.0;
    };
    let intersection = intersection.width as f32 * intersection.height as f32;
    let left_area = left.width as f32 * left.height as f32;
    let right_area = right.width as f32 * right.height as f32;
    intersection / (left_area + right_area - intersection)
}

/// Returns collisions between tracked detections.
pub fn detect_detection_collisions(
    detections: &[TrackedDetection],
    options: CollisionOptions,
) -> Result<Vec<ObjectCollision>> {
    detect_detection_collisions_with_broad_phase(
        detections,
        options,
        CollisionBroadPhaseOptions::default(),
    )
}

/// Returns collisions between tracked detections using explicit broad-phase options.
pub fn detect_detection_collisions_with_broad_phase(
    detections: &[TrackedDetection],
    options: CollisionOptions,
    broad_phase: CollisionBroadPhaseOptions,
) -> Result<Vec<ObjectCollision>> {
    options.validate()?;
    broad_phase.validate()?;
    let regions = detections
        .iter()
        .map(|detection| CollisionRegion {
            id: None,
            region: detection.region,
        })
        .collect::<Vec<_>>();
    detect_collisions(&regions, options, broad_phase)
}

/// Returns collisions between object tracks.
pub fn detect_track_collisions<'a>(
    tracks: impl IntoIterator<Item = &'a ObjectTrack>,
    options: CollisionOptions,
) -> Result<Vec<ObjectCollision>> {
    detect_track_collisions_with_broad_phase(tracks, options, CollisionBroadPhaseOptions::default())
}

/// Returns collisions between object tracks using explicit broad-phase options.
pub fn detect_track_collisions_with_broad_phase<'a>(
    tracks: impl IntoIterator<Item = &'a ObjectTrack>,
    options: CollisionOptions,
    broad_phase: CollisionBroadPhaseOptions,
) -> Result<Vec<ObjectCollision>> {
    options.validate()?;
    broad_phase.validate()?;
    let regions = tracks
        .into_iter()
        .map(|track| CollisionRegion {
            id: Some(track.id.as_str()),
            region: track.region,
        })
        .collect::<Vec<_>>();
    detect_collisions(&regions, options, broad_phase)
}

fn compatible(track: &ObjectTrack, detection: &TrackedDetection) -> bool {
    track.kind == detection.kind
        && match (&track.label, &detection.label) {
            (Some(left), Some(right)) => left == right,
            _ => true,
        }
}

#[derive(Debug, Clone, Copy)]
struct CollisionRegion<'a> {
    id: Option<&'a str>,
    region: BoundingBox,
}

fn detect_collisions(
    regions: &[CollisionRegion<'_>],
    options: CollisionOptions,
    broad_phase: CollisionBroadPhaseOptions,
) -> Result<Vec<ObjectCollision>> {
    let rects = regions
        .iter()
        .map(|region| rect_for_region(region.region))
        .collect::<Option<Vec<_>>>();
    let pairs = if let Some(rects) = rects {
        broad_phase_pairs_2d(&rects, to_broad_phase_config(broad_phase))?
    } else {
        brute_force_collision_pairs(regions)
    };

    let mut collisions = Vec::new();
    for pair in pairs {
        let left_index = pair.left_index;
        let right_index = pair.right_index;
        let left = regions[left_index];
        let right = regions[right_index];
        let Some(intersection) = bbox_intersection(left.region, right.region) else {
            continue;
        };
        let iou = bbox_iou_with_intersection(left.region, right.region, intersection);
        if iou < options.min_iou {
            continue;
        }
        collisions.push(ObjectCollision {
            left_index,
            right_index,
            left_id: left.id.map(str::to_string),
            right_id: right.id.map(str::to_string),
            left_region: left.region,
            right_region: right.region,
            intersection,
            iou,
        });
    }
    collisions.sort_by_key(|collision| (collision.left_index, collision.right_index));
    Ok(collisions)
}

fn to_broad_phase_config(options: CollisionBroadPhaseOptions) -> BroadPhase2Config {
    BroadPhase2Config {
        strategy: options.strategy.into(),
        brute_force_threshold: options.brute_force_threshold,
        max_cells_per_item: options.max_cells_per_region,
        cell_size: options.cell_size.into(),
    }
}

fn rect_for_region(region: BoundingBox) -> Option<RectU32> {
    let _ = region.x.checked_add(region.width)?;
    let _ = region.y.checked_add(region.height)?;
    RectU32::new(region.x, region.y, region.width, region.height).ok()
}

fn brute_force_collision_pairs(
    regions: &[CollisionRegion<'_>],
) -> Vec<math_geometry_2d::CollisionPair> {
    let mut pairs = Vec::new();
    for left_index in 0..regions.len() {
        for right_index in (left_index + 1)..regions.len() {
            if bbox_intersects(regions[left_index].region, regions[right_index].region) {
                pairs.push(math_geometry_2d::CollisionPair {
                    left_index,
                    right_index,
                });
            }
        }
    }
    pairs
}

fn bbox_iou_with_intersection(
    left: BoundingBox,
    right: BoundingBox,
    intersection: BoundingBox,
) -> f32 {
    let intersection = intersection.width as f32 * intersection.height as f32;
    let left_area = left.width as f32 * left.height as f32;
    let right_area = right.width as f32 * right.height as f32;
    intersection / (left_area + right_area - intersection)
}

fn observation_for_track(analyzer: &str, track: ObjectTrack) -> Observation {
    let mut observation = Observation::new(analyzer, track.kind)
        .at_frame(track.last_position)
        .region(track.region)
        .track_id(track.id)
        .attribute("track_age_frames", track.age_frames.to_string())
        .attribute("track_missed_frames", track.missed_frames.to_string())
        .attribute(
            "track_first_frame",
            track.first_position.frame_index.to_string(),
        );
    if let Some(label) = track.label {
        observation = observation.label(label);
    }
    if let Some(score) = track.score {
        observation = observation.score(score);
    }
    for (key, value) in track.attributes {
        observation = observation.attribute(key, value);
    }
    observation
}

fn visual_kind_from_observation(kind: &ObservationKind) -> VisualDetectionKind {
    match kind {
        ObservationKind::Face => VisualDetectionKind::Face,
        ObservationKind::Object => VisualDetectionKind::Object,
        ObservationKind::Text => VisualDetectionKind::TextRegion,
        ObservationKind::Custom(value) => {
            if value.trim().is_empty() {
                VisualDetectionKind::Object
            } else {
                VisualDetectionKind::Custom(value.clone())
            }
        }
        _ => VisualDetectionKind::Custom(format!("{kind:?}")),
    }
}

fn vision_error(error: vision_core::VisionError) -> DetectError {
    DetectError::InvalidArgument(error.to_string())
}

#[cfg(test)]
mod tests {
    use num_rational::Rational64;
    use video_analysis_core::{FramePosition, OwnedVideoFrame, PixelFormat};

    use super::*;

    fn position(frame_index: u64) -> FramePosition {
        FramePosition::from_frame_index(frame_index, Rational64::new(30, 1))
    }

    fn frame(frame_index: u64) -> OwnedVideoFrame {
        OwnedVideoFrame {
            position: position(frame_index),
            width: 64,
            height: 64,
            pixel_format: PixelFormat::Rgb24,
            data: vec![0; 64 * 64 * 3],
            stride: 64 * 3,
        }
    }

    #[test]
    fn tracker_keeps_id_for_overlapping_detections() {
        let mut tracker = IouTracker::new(TrackingOptions::default()).unwrap();

        let first = tracker
            .update(
                position(0),
                [
                    TrackedDetection::new(BoundingBox::new(10, 10, 20, 20).unwrap())
                        .label("person"),
                ],
            )
            .unwrap();
        let second = tracker
            .update(
                position(1),
                [
                    TrackedDetection::new(BoundingBox::new(12, 10, 20, 20).unwrap())
                        .label("person"),
                ],
            )
            .unwrap();

        assert_eq!(first[0].id, second[0].id);
        assert_eq!(second[0].age_frames, 2);
    }

    #[test]
    fn tracker_uses_new_id_for_incompatible_label() {
        let mut tracker = IouTracker::new(TrackingOptions::default()).unwrap();

        let first = tracker
            .update(
                position(0),
                [
                    TrackedDetection::new(BoundingBox::new(10, 10, 20, 20).unwrap())
                        .label("person"),
                ],
            )
            .unwrap();
        let second = tracker
            .update(
                position(1),
                [TrackedDetection::new(BoundingBox::new(11, 11, 20, 20).unwrap()).label("car")],
            )
            .unwrap();

        assert_ne!(first[0].id, second[0].id);
    }

    #[test]
    fn detects_collisions_between_overlapping_detections() {
        let detections = [
            TrackedDetection::new(BoundingBox::new(0, 0, 10, 10).unwrap()).label("person"),
            TrackedDetection::new(BoundingBox::new(5, 4, 10, 10).unwrap()).label("bike"),
            TrackedDetection::new(BoundingBox::new(30, 30, 4, 4).unwrap()).label("car"),
        ];

        let collisions =
            detect_detection_collisions(&detections, CollisionOptions::default()).unwrap();

        assert_eq!(collisions.len(), 1);
        assert_eq!(collisions[0].left_index, 0);
        assert_eq!(collisions[0].right_index, 1);
        assert_eq!(
            collisions[0].intersection,
            BoundingBox::new(5, 4, 5, 6).unwrap()
        );
        assert!(collisions[0].iou > 0.0);
    }

    #[test]
    fn filters_collisions_by_iou() {
        let detections = [
            TrackedDetection::new(BoundingBox::new(0, 0, 10, 10).unwrap()),
            TrackedDetection::new(BoundingBox::new(9, 9, 10, 10).unwrap()),
        ];

        let collisions =
            detect_detection_collisions(&detections, CollisionOptions { min_iou: 0.1 }).unwrap();

        assert!(collisions.is_empty());
    }

    #[test]
    fn broad_phase_collision_strategies_match_brute_force() {
        let detections = (0..256)
            .map(|index| {
                let x = ((index % 32) * 8) as u32;
                let y = ((index / 32) * 8) as u32;
                TrackedDetection::new(BoundingBox::new(x, y, 6, 6).unwrap())
            })
            .chain([
                TrackedDetection::new(BoundingBox::new(0, 0, 20, 20).unwrap()),
                TrackedDetection::new(BoundingBox::new(3, 3, 5, 5).unwrap()),
            ])
            .collect::<Vec<_>>();
        let brute = detect_detection_collisions_with_broad_phase(
            &detections,
            CollisionOptions { min_iou: 0.0 },
            CollisionBroadPhaseOptions {
                strategy: CollisionBroadPhaseStrategy::BruteForce,
                ..CollisionBroadPhaseOptions::default()
            },
        )
        .unwrap();
        let grid = detect_detection_collisions_with_broad_phase(
            &detections,
            CollisionOptions { min_iou: 0.0 },
            CollisionBroadPhaseOptions {
                strategy: CollisionBroadPhaseStrategy::SpatialHashGrid,
                cell_size: CollisionCellSize::Fixed {
                    width: 8,
                    height: 8,
                },
                ..CollisionBroadPhaseOptions::default()
            },
        )
        .unwrap();
        let sweep = detect_detection_collisions_with_broad_phase(
            &detections,
            CollisionOptions { min_iou: 0.0 },
            CollisionBroadPhaseOptions {
                strategy: CollisionBroadPhaseStrategy::SweepAndPrune,
                ..CollisionBroadPhaseOptions::default()
            },
        )
        .unwrap();
        let auto =
            detect_detection_collisions(&detections, CollisionOptions { min_iou: 0.0 }).unwrap();

        assert_eq!(brute, grid);
        assert_eq!(brute, sweep);
        assert_eq!(brute, auto);
    }

    #[test]
    fn broad_phase_handles_large_sparse_detection_sets() {
        let detections = (0..1_000)
            .map(|index| {
                TrackedDetection::new(
                    BoundingBox::new((index * 10) as u32, (index * 7) as u32, 2, 2).unwrap(),
                )
            })
            .collect::<Vec<_>>();

        let collisions =
            detect_detection_collisions(&detections, CollisionOptions::default()).unwrap();

        assert!(collisions.is_empty());
    }

    #[test]
    fn tracker_reports_collisions_between_active_tracks() {
        let mut tracker = IouTracker::new(TrackingOptions::default()).unwrap();
        tracker
            .update(
                position(0),
                [
                    TrackedDetection::new(BoundingBox::new(0, 0, 12, 12).unwrap())
                        .track_hint("left"),
                    TrackedDetection::new(BoundingBox::new(6, 0, 12, 12).unwrap())
                        .track_hint("right"),
                ],
            )
            .unwrap();

        let collisions = tracker.collisions(CollisionOptions::default()).unwrap();

        assert_eq!(collisions.len(), 1);
        assert_eq!(collisions[0].left_id.as_deref(), Some("track-1"));
        assert_eq!(collisions[0].right_id.as_deref(), Some("track-2"));

        let explicit = tracker
            .collisions_with_broad_phase(
                CollisionOptions::default(),
                CollisionBroadPhaseOptions {
                    strategy: CollisionBroadPhaseStrategy::SpatialHashGrid,
                    cell_size: CollisionCellSize::Fixed {
                        width: 4,
                        height: 4,
                    },
                    ..CollisionBroadPhaseOptions::default()
                },
            )
            .unwrap();
        assert_eq!(collisions, explicit);
    }

    #[test]
    fn analyzer_emits_track_observations() {
        struct Backend;

        impl ObjectDetectionBackend for Backend {
            fn detect_frame(&mut self, _frame: &VideoFrame<'_>) -> Result<Vec<TrackedDetection>> {
                Ok(vec![TrackedDetection::new(
                    BoundingBox::new(1, 2, 10, 12).unwrap(),
                )
                .label("person")
                .score(0.9)])
            }
        }

        let mut analyzer =
            ObjectTrackingAnalyzer::new("tracker", Backend, TrackingOptions::default()).unwrap();
        let observations = analyzer.process_frame(&frame(0).as_frame()).unwrap();

        assert_eq!(observations.len(), 1);
        assert_eq!(observations[0].kind, ObservationKind::Object);
        assert_eq!(observations[0].track_id.as_deref(), Some("track-1"));
        assert_eq!(observations[0].label.as_deref(), Some("person"));
    }
}
