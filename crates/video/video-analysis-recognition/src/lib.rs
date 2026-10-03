#![doc = include_str!("../README.md")]

mod external_command;
mod model_analyzers;
mod model_predictions;
pub mod surface;
mod video_text_semantics;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use video_analysis_core::{
    BoundingBox, DetectError, FramePosition, Observation, ObservationKind, Result, VideoAnalyzer,
    VideoFrame,
};
use vision_core::{IdentityMatch, VisualDetectionKind, VisualEmbedding, VisualRegion};

pub use external_command::*;
pub use model_analyzers::*;
pub use model_predictions::*;
pub use video_text_semantics::*;

#[cfg(feature = "onnx")]
#[derive(Debug)]
/// ONNX object detector backend for video frames.
pub struct OnnxObjectDetectorBackend<R = runtime_onnx::OnnxSession> {
    detector: image_analysis_detection::OnnxObjectDetector<R>,
}

#[cfg(feature = "onnx")]
/// Compatibility alias for the video ONNX object detector name.
pub type OnnxObjectDetector<R = runtime_onnx::OnnxSession> = OnnxObjectDetectorBackend<R>;

#[cfg(feature = "onnx")]
impl OnnxObjectDetectorBackend<runtime_onnx::OnnxSession> {
    /// Builds this value from bundle.
    pub fn from_bundle(bundle: model_runtime::ModelBundle) -> Result<Self> {
        Ok(Self {
            detector: image_analysis_detection::OnnxObjectDetector::from_bundle(bundle)?,
        })
    }
}

#[cfg(feature = "onnx")]
impl<R: image_analysis_detection::OnnxObjectDetectionRunner> OnnxObjectDetectorBackend<R> {
    /// Builds this value from runner.
    pub fn from_runner(bundle: model_runtime::ModelBundle, runner: R) -> Result<Self> {
        Ok(Self {
            detector: image_analysis_detection::OnnxObjectDetector::from_runner(bundle, runner)?,
        })
    }

    /// Returns predict frame with original size.
    pub fn predict_frame_with_original_size(
        &mut self,
        frame: &VideoFrame<'_>,
    ) -> Result<Vec<RawPrediction>> {
        let image = image_analysis_core::ImageView::from_video_frame(frame)?;
        let detections = self.detector.detect_image(&image)?;
        Ok(detections
            .into_iter()
            .map(|detection| RawPrediction {
                kind: Some("object".to_string()),
                label: Some(detection.label),
                text: None,
                score: detection.score,
                region: Some(RawBoundingBox::xywh(
                    detection.region.x as f32,
                    detection.region.y as f32,
                    detection.region.width as f32,
                    detection.region.height as f32,
                )),
                attributes: detection.attributes,
            })
            .collect())
    }
}

#[cfg(feature = "onnx")]
impl<R: image_analysis_detection::OnnxObjectDetectionRunner> VisionModelBackend
    for OnnxObjectDetectorBackend<R>
{
    fn task(&self) -> model_runtime::ModelTask {
        model_runtime::ModelTask::ObjectDetection
    }

    fn runtime_backend(&self) -> model_runtime::ModelRuntimeBackend {
        model_runtime::ModelRuntimeBackend::Onnx
    }

    fn predict_frame(&mut self, frame: &VideoFrame<'_>) -> Result<Vec<RawPrediction>> {
        self.predict_frame_with_original_size(frame)
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for embedding.
pub struct Embedding {
    values: Vec<f32>,
}

impl Embedding {
    /// Creates a new value.
    pub fn new(values: impl Into<Vec<f32>>) -> Result<Self> {
        let mut values = values.into();
        normalize_values(&mut values)?;
        Ok(Self { values })
    }

    /// Returns values.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Returns dimensions.
    pub fn dimensions(&self) -> usize {
        self.values.len()
    }

    /// Returns cosine similarity.
    pub fn cosine_similarity(&self, other: &Self) -> Result<f32> {
        if self.dimensions() != other.dimensions() {
            return Err(DetectError::InvalidArgument(format!(
                "embedding dimensions differ: {} != {}",
                self.dimensions(),
                other.dimensions()
            )));
        }
        Ok(self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(left, right)| left * right)
            .sum())
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for reference identity.
pub struct ReferenceIdentity {
    id: String,
    label: String,
    kind: ObservationKind,
    embeddings: Vec<Embedding>,
    attributes: BTreeMap<String, String>,
}

impl ReferenceIdentity {
    /// Creates a new value.
    pub fn new(id: impl Into<String>, label: impl Into<String>, kind: ObservationKind) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            kind,
            embeddings: Vec::new(),
            attributes: BTreeMap::new(),
        }
    }

    /// Returns this value with embedding.
    pub fn with_embedding(mut self, embedding: impl Into<Vec<f32>>) -> Result<Self> {
        self.add_embedding(embedding)?;
        Ok(self)
    }

    /// Returns attribute.
    pub fn attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// Adds add embedding to this value.
    pub fn add_embedding(&mut self, embedding: impl Into<Vec<f32>>) -> Result<()> {
        let embedding = Embedding::new(embedding)?;
        if let Some(dimensions) = self.dimensions() {
            if dimensions != embedding.dimensions() {
                return Err(DetectError::InvalidArgument(format!(
                    "reference `{}` has mixed embedding dimensions: {} != {}",
                    self.id,
                    dimensions,
                    embedding.dimensions()
                )));
            }
        }
        self.embeddings.push(embedding);
        Ok(())
    }

    /// Returns identifier.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Returns kind.
    pub fn kind(&self) -> &ObservationKind {
        &self.kind
    }

    /// Returns embeddings.
    pub fn embeddings(&self) -> &[Embedding] {
        &self.embeddings
    }

    /// Returns attributes.
    pub fn attributes(&self) -> &BTreeMap<String, String> {
        &self.attributes
    }

    /// Returns dimensions.
    pub fn dimensions(&self) -> Option<usize> {
        self.embeddings.first().map(Embedding::dimensions)
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
/// Data type for reference library.
pub struct ReferenceLibrary {
    identities: BTreeMap<String, ReferenceIdentity>,
    dimensions: Option<usize>,
}

impl ReferenceLibrary {
    /// Creates a new value.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds add identity to this value.
    pub fn add_identity(&mut self, identity: ReferenceIdentity) -> Result<()> {
        if identity.id().is_empty() {
            return Err(DetectError::InvalidArgument(
                "reference id must not be empty".to_string(),
            ));
        }
        if identity.label().is_empty() {
            return Err(DetectError::InvalidArgument(
                "reference label must not be empty".to_string(),
            ));
        }
        let dimensions = identity.dimensions().ok_or_else(|| {
            DetectError::InvalidArgument(format!(
                "reference `{}` must contain at least one embedding",
                identity.id()
            ))
        })?;
        self.validate_dimensions(dimensions)?;
        if self.identities.contains_key(identity.id()) {
            return Err(DetectError::InvalidArgument(
                "duplicate reference id".to_string(),
            ));
        }
        self.identities.insert(identity.id().to_string(), identity);
        self.dimensions = Some(dimensions);
        Ok(())
    }

    /// Adds add reference to this value.
    pub fn add_reference(
        &mut self,
        id: impl Into<String>,
        label: impl Into<String>,
        kind: ObservationKind,
        embedding: impl Into<Vec<f32>>,
    ) -> Result<()> {
        let identity = ReferenceIdentity::new(id, label, kind).with_embedding(embedding)?;
        self.add_identity(identity)
    }

    /// Adds add sample to this value.
    pub fn add_sample(&mut self, id: &str, embedding: impl Into<Vec<f32>>) -> Result<()> {
        let embedding = Embedding::new(embedding)?;
        self.validate_dimensions(embedding.dimensions())?;
        let identity = self
            .identities
            .get_mut(id)
            .ok_or_else(|| DetectError::InvalidArgument(format!("unknown reference id `{id}`")))?;
        identity.embeddings.push(embedding);
        Ok(())
    }

    /// Returns identity.
    pub fn identity(&self, id: &str) -> Option<&ReferenceIdentity> {
        self.identities.get(id)
    }

    /// Returns identities.
    pub fn identities(&self) -> impl Iterator<Item = &ReferenceIdentity> {
        self.identities.values()
    }

    /// Returns len.
    pub fn len(&self) -> usize {
        self.identities.len()
    }

    /// Returns whether is empty.
    pub fn is_empty(&self) -> bool {
        self.identities.is_empty()
    }

    /// Returns dimensions.
    pub fn dimensions(&self) -> Option<usize> {
        self.dimensions
    }

    /// Returns search.
    pub fn search(
        &self,
        embedding: &Embedding,
        kind: Option<&ObservationKind>,
        options: &MatchOptions,
    ) -> Result<Vec<RecognitionMatch>> {
        options.validate()?;
        self.validate_dimensions(embedding.dimensions())?;

        let mut matches = Vec::new();
        for identity in self.identities.values() {
            if kind
                .map(|candidate_kind| identity.kind() != candidate_kind)
                .unwrap_or(false)
            {
                continue;
            }
            let Some(score) = best_identity_score(identity, embedding)? else {
                continue;
            };
            if score < options.min_score {
                continue;
            }
            matches.push(RecognitionMatch {
                reference_id: identity.id().to_string(),
                label: identity.label().to_string(),
                kind: identity.kind().clone(),
                score,
                attributes: identity.attributes().clone(),
            });
        }

        matches.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.reference_id.cmp(&right.reference_id))
        });
        matches.truncate(options.max_results);
        Ok(matches)
    }

    fn validate_dimensions(&self, dimensions: usize) -> Result<()> {
        if let Some(expected) = self.dimensions {
            if expected != dimensions {
                return Err(DetectError::InvalidArgument(format!(
                    "embedding dimensions differ from reference library: {dimensions} != {expected}"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Data type for match options.
pub struct MatchOptions {
    /// The min score value.
    pub min_score: f32,
    /// The max results value.
    pub max_results: usize,
}

impl MatchOptions {
    /// Creates a new value.
    pub fn new(min_score: f32) -> Result<Self> {
        let options = Self {
            min_score,
            ..Self::default()
        };
        options.validate()?;
        Ok(options)
    }

    /// Returns max results.
    pub fn max_results(mut self, value: usize) -> Self {
        self.max_results = value;
        self
    }

    fn validate(&self) -> Result<()> {
        if !self.min_score.is_finite() || !(-1.0..=1.0).contains(&self.min_score) {
            return Err(DetectError::InvalidArgument(
                "minimum recognition score must be finite and between -1.0 and 1.0".to_string(),
            ));
        }
        if self.max_results == 0 {
            return Err(DetectError::InvalidArgument(
                "max recognition results must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for MatchOptions {
    fn default() -> Self {
        Self {
            min_score: 0.75,
            max_results: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for recognition match.
pub struct RecognitionMatch {
    /// The reference identifier value.
    pub reference_id: String,
    /// Label assigned to this value.
    pub label: String,
    /// The kind value.
    pub kind: ObservationKind,
    /// Score assigned to this value.
    pub score: f32,
    /// The attributes value.
    pub attributes: BTreeMap<String, String>,
}

impl RecognitionMatch {
    /// Converts this recognition result into a shared visual identity match.
    pub fn to_identity_match(&self) -> Result<IdentityMatch> {
        let mut visual = IdentityMatch::new(self.reference_id.clone(), self.score)
            .map_err(vision_error)?
            .kind(visual_kind_from_observation(&self.kind))
            .map_err(vision_error)?;
        if !self.label.trim().is_empty() {
            visual = visual.label(self.label.clone()).map_err(vision_error)?;
        }
        visual.attributes = self.attributes.clone();
        Ok(visual)
    }
}

impl TryFrom<RecognitionMatch> for IdentityMatch {
    type Error = DetectError;

    fn try_from(value: RecognitionMatch) -> std::result::Result<Self, Self::Error> {
        value.to_identity_match()
    }
}

impl TryFrom<&RecognitionMatch> for IdentityMatch {
    type Error = DetectError;

    fn try_from(value: &RecognitionMatch) -> std::result::Result<Self, Self::Error> {
        value.to_identity_match()
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for recognition candidate.
pub struct RecognitionCandidate {
    /// The kind value.
    pub kind: ObservationKind,
    /// The embedding value.
    pub embedding: Embedding,
    /// The region value.
    pub region: Option<BoundingBox>,
    /// The detector label value.
    pub detector_label: Option<String>,
    /// The detection score value.
    pub detection_score: Option<f32>,
    /// The track identifier value.
    pub track_id: Option<String>,
    /// The attributes value.
    pub attributes: BTreeMap<String, String>,
}

impl RecognitionCandidate {
    /// Creates a new value.
    pub fn new(kind: ObservationKind, embedding: impl Into<Vec<f32>>) -> Result<Self> {
        Ok(Self {
            kind,
            embedding: Embedding::new(embedding)?,
            region: None,
            detector_label: None,
            detection_score: None,
            track_id: None,
            attributes: BTreeMap::new(),
        })
    }

    /// Returns region.
    pub fn region(mut self, region: BoundingBox) -> Self {
        self.region = Some(region);
        self
    }

    /// Returns detector label.
    pub fn detector_label(mut self, label: impl Into<String>) -> Self {
        self.detector_label = Some(label.into());
        self
    }

    /// Returns detection score.
    pub fn detection_score(mut self, score: f32) -> Self {
        self.detection_score = Some(score);
        self
    }

    /// Returns track identifier.
    pub fn track_id(mut self, track_id: impl Into<String>) -> Self {
        self.track_id = Some(track_id.into());
        self
    }

    /// Returns attribute.
    pub fn attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// Converts this recognition candidate into a shared visual embedding.
    pub fn to_visual_embedding(&self) -> Result<VisualEmbedding> {
        let mut visual = VisualEmbedding::new(self.embedding.values().to_vec())
            .map_err(vision_error)?
            .kind(visual_kind_from_observation(&self.kind))
            .map_err(vision_error)?;
        if let Some(region) = self.region {
            visual = visual
                .region(VisualRegion::from(region))
                .map_err(vision_error)?;
        }
        if let Some(track_id) = &self.track_id {
            visual = visual
                .source_detection_id(track_id.clone())
                .map_err(vision_error)?;
        }
        visual.attributes = self.attributes.clone();
        Ok(visual)
    }
}

impl TryFrom<RecognitionCandidate> for VisualEmbedding {
    type Error = DetectError;

    fn try_from(value: RecognitionCandidate) -> std::result::Result<Self, Self::Error> {
        value.to_visual_embedding()
    }
}

impl TryFrom<&RecognitionCandidate> for VisualEmbedding {
    type Error = DetectError;

    fn try_from(value: &RecognitionCandidate) -> std::result::Result<Self, Self::Error> {
        value.to_visual_embedding()
    }
}

/// Trait for recognition backend implementations.
pub trait RecognitionBackend {
    /// Returns process frame.
    fn process_frame(&mut self, frame: &VideoFrame<'_>) -> Result<Vec<RecognitionCandidate>>;
}

/// Data type for recognition video analyzer.
pub struct RecognitionVideoAnalyzer<B> {
    name: String,
    backend: B,
    library: ReferenceLibrary,
    match_options: MatchOptions,
    temporal_options: TemporalRecognitionOptions,
    aggregator: TemporalRecognitionAggregator,
}

impl<B> RecognitionVideoAnalyzer<B> {
    /// Creates a new value.
    pub fn new(name: impl Into<String>, backend: B, library: ReferenceLibrary) -> Self {
        Self {
            name: name.into(),
            backend,
            library,
            match_options: MatchOptions::default(),
            temporal_options: TemporalRecognitionOptions::immediate(),
            aggregator: TemporalRecognitionAggregator::new(TemporalRecognitionOptions::immediate()),
        }
    }

    /// Returns match options.
    pub fn match_options(mut self, value: MatchOptions) -> Self {
        self.match_options = value;
        self
    }

    /// Returns temporal options.
    pub fn temporal_options(mut self, value: TemporalRecognitionOptions) -> Self {
        self.temporal_options = value;
        self.aggregator = TemporalRecognitionAggregator::new(value);
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

    /// Returns library.
    pub fn library(&self) -> &ReferenceLibrary {
        &self.library
    }
}

impl<B: RecognitionBackend> VideoAnalyzer for RecognitionVideoAnalyzer<B> {
    fn name(&self) -> &str {
        &self.name
    }

    fn process_frame(&mut self, frame: &VideoFrame<'_>) -> Result<Vec<Observation>> {
        let candidates = self.backend.process_frame(frame)?;
        let mut observations = Vec::new();
        let analyzer_name = self.name.clone();
        self.aggregator.prune(frame.position);

        for (candidate_index, candidate) in candidates.into_iter().enumerate() {
            let matches = self.library.search(
                &candidate.embedding,
                Some(&candidate.kind),
                &self.match_options,
            )?;
            if self.temporal_options.min_hits <= 1 {
                observations.extend(matches.into_iter().map(|recognition_match| {
                    observation_for_match(
                        &analyzer_name,
                        frame.position,
                        &candidate,
                        recognition_match,
                    )
                }));
                continue;
            }
            for recognition_match in matches {
                if let Some(observation) = self.aggregator.accept(
                    &analyzer_name,
                    frame.position,
                    candidate_index,
                    &candidate,
                    recognition_match,
                ) {
                    observations.push(observation);
                }
            }
        }

        Ok(observations)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Data type for temporal recognition options.
pub struct TemporalRecognitionOptions {
    /// The min hits value.
    pub min_hits: usize,
    /// The max gap frames value.
    pub max_gap_frames: u64,
    /// The min average score value.
    pub min_average_score: f32,
}

impl TemporalRecognitionOptions {
    /// Returns immediate.
    pub fn immediate() -> Self {
        Self {
            min_hits: 1,
            max_gap_frames: 0,
            min_average_score: 0.0,
        }
    }

    /// Returns track window.
    pub fn track_window(
        min_hits: usize,
        max_gap_frames: u64,
        min_average_score: f32,
    ) -> Result<Self> {
        let options = Self {
            min_hits,
            max_gap_frames,
            min_average_score,
        };
        options.validate()?;
        Ok(options)
    }

    fn validate(&self) -> Result<()> {
        if self.min_hits == 0 {
            return Err(DetectError::InvalidArgument(
                "temporal recognition min_hits must be greater than zero".to_string(),
            ));
        }
        if !self.min_average_score.is_finite() || !(-1.0..=1.0).contains(&self.min_average_score) {
            return Err(DetectError::InvalidArgument(
                "temporal recognition minimum average score must be finite and between -1.0 and 1.0"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
/// Data type for temporal recognition aggregator.
pub struct TemporalRecognitionAggregator {
    options: TemporalRecognitionOptions,
    tracks: BTreeMap<String, TrackState>,
}

impl TemporalRecognitionAggregator {
    /// Creates a new value.
    pub fn new(options: TemporalRecognitionOptions) -> Self {
        Self {
            options,
            tracks: BTreeMap::new(),
        }
    }

    /// Returns accept.
    pub fn accept(
        &mut self,
        analyzer: &str,
        position: FramePosition,
        candidate_index: usize,
        candidate: &RecognitionCandidate,
        recognition_match: RecognitionMatch,
    ) -> Option<Observation> {
        let key = track_key(position, candidate_index, candidate);
        let state = self.tracks.entry(key).or_default();
        state.last_frame = position.frame_index;
        state.evidence.push_back(TrackEvidence {
            position,
            candidate: candidate.clone(),
            recognition_match,
        });

        let latest = state.evidence.back()?;
        let reference_id = latest.recognition_match.reference_id.clone();
        if state.emitted_reference_ids.contains(&reference_id) {
            return None;
        }

        let mut hits = 0usize;
        let mut score_sum = 0.0f32;
        let mut latest_matching = None;
        for evidence in state.evidence.iter().rev() {
            if evidence.recognition_match.reference_id == reference_id {
                hits += 1;
                score_sum += evidence.recognition_match.score;
                latest_matching = Some(evidence);
            }
        }

        if hits < self.options.min_hits {
            return None;
        }
        let average = score_sum / hits as f32;
        if average < self.options.min_average_score {
            return None;
        }

        state.emitted_reference_ids.insert(reference_id);
        latest_matching.map(|evidence| {
            let mut observation = observation_for_match(
                analyzer,
                evidence.position,
                &evidence.candidate,
                evidence.recognition_match.clone(),
            );
            observation.score = Some(average);
            observation = observation
                .attribute("temporal_hits", hits.to_string())
                .attribute("temporal_average_score", average.to_string());
            observation
        })
    }

    /// Returns prune.
    pub fn prune(&mut self, position: FramePosition) {
        let max_gap = self.options.max_gap_frames;
        self.tracks
            .retain(|_, state| position.frame_index.saturating_sub(state.last_frame) <= max_gap);
    }

    /// Returns reset.
    pub fn reset(&mut self) {
        self.tracks.clear();
    }
}

#[derive(Debug, Clone, Default)]
struct TrackState {
    last_frame: u64,
    evidence: VecDeque<TrackEvidence>,
    emitted_reference_ids: BTreeSet<String>,
}

#[derive(Debug, Clone)]
struct TrackEvidence {
    position: FramePosition,
    candidate: RecognitionCandidate,
    recognition_match: RecognitionMatch,
}

fn best_identity_score(identity: &ReferenceIdentity, embedding: &Embedding) -> Result<Option<f32>> {
    let mut best = None;
    for reference_embedding in identity.embeddings() {
        let score = reference_embedding.cosine_similarity(embedding)?;
        best = Some(best.map(|value: f32| value.max(score)).unwrap_or(score));
    }
    Ok(best)
}

fn normalize_values(values: &mut [f32]) -> Result<()> {
    if values.is_empty() {
        return Err(DetectError::InvalidArgument(
            "embedding must not be empty".to_string(),
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(DetectError::InvalidArgument(
            "embedding values must be finite".to_string(),
        ));
    }

    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return Err(DetectError::InvalidArgument(
            "embedding norm must be finite and non-zero".to_string(),
        ));
    }
    for value in values {
        *value /= norm;
    }
    Ok(())
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

fn observation_for_match(
    analyzer: &str,
    position: FramePosition,
    candidate: &RecognitionCandidate,
    recognition_match: RecognitionMatch,
) -> Observation {
    let mut observation = Observation::new(analyzer, recognition_match.kind)
        .at_frame(position)
        .label(recognition_match.label)
        .score(recognition_match.score)
        .attribute("reference_id", recognition_match.reference_id);
    if let Some(region) = candidate.region {
        observation = observation.region(region);
    }
    if let Some(track_id) = &candidate.track_id {
        observation = observation.track_id(track_id.clone());
    }
    if let Some(label) = &candidate.detector_label {
        observation = observation.attribute("detector_label", label.clone());
    }
    if let Some(score) = candidate.detection_score {
        observation = observation.attribute("detection_score", score.to_string());
    }
    for (key, value) in &candidate.attributes {
        observation = observation.attribute(key.clone(), value.clone());
    }
    for (key, value) in recognition_match.attributes {
        observation = observation.attribute(format!("reference.{key}"), value);
    }
    observation
}

fn track_key(
    position: FramePosition,
    candidate_index: usize,
    candidate: &RecognitionCandidate,
) -> String {
    if let Some(track_id) = &candidate.track_id {
        return format!("track:{track_id}");
    }
    if let Some(region) = candidate.region {
        return format!(
            "region:{:?}:{}:{}:{}:{}",
            candidate.kind,
            region.x / 16,
            region.y / 16,
            region.width / 16,
            region.height / 16
        );
    }
    format!("frame:{}:{candidate_index}", position.frame_index)
}

#[cfg(test)]
mod tests {
    use num_rational::Rational64;
    use video_analysis_core::{OwnedVideoFrame, PixelFormat};

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

    fn library() -> ReferenceLibrary {
        let mut library = ReferenceLibrary::new();
        library
            .add_reference(
                "einstein",
                "Albert Einstein",
                ObservationKind::Face,
                [1.0, 0.0],
            )
            .unwrap();
        library
            .add_reference("curie", "Marie Curie", ObservationKind::Face, [0.0, 1.0])
            .unwrap();
        library
    }

    #[test]
    fn search_returns_nearest_reference() {
        let library = library();
        let query = Embedding::new([0.9, 0.1]).unwrap();
        let options = MatchOptions::new(0.7).unwrap();

        let matches = library
            .search(&query, Some(&ObservationKind::Face), &options)
            .unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].reference_id, "einstein");
        assert!(matches[0].score > 0.99);
    }

    #[test]
    fn search_filters_by_observation_kind() {
        let library = library();
        let query = Embedding::new([1.0, 0.0]).unwrap();

        let matches = library
            .search(
                &query,
                Some(&ObservationKind::Object),
                &MatchOptions::default(),
            )
            .unwrap();

        assert!(matches.is_empty());
    }

    #[test]
    fn library_rejects_dimension_mismatches() {
        let mut library = library();

        let err = library
            .add_reference("short", "Short", ObservationKind::Face, [1.0])
            .unwrap_err();

        assert!(err.to_string().contains("embedding dimensions differ"));
    }

    #[test]
    fn extra_reference_samples_improve_best_score() {
        let mut library = library();
        library.add_sample("einstein", [0.7, 0.7]).unwrap();
        let query = Embedding::new([0.7, 0.7]).unwrap();

        let matches = library
            .search(
                &query,
                Some(&ObservationKind::Face),
                &MatchOptions::default(),
            )
            .unwrap();

        assert_eq!(matches[0].reference_id, "einstein");
        assert!(matches[0].score > 0.999);
    }

    #[test]
    fn analyzer_emits_recognized_observation() {
        struct StaticBackend;

        impl RecognitionBackend for StaticBackend {
            fn process_frame(
                &mut self,
                _frame: &VideoFrame<'_>,
            ) -> Result<Vec<RecognitionCandidate>> {
                Ok(vec![RecognitionCandidate::new(
                    ObservationKind::Face,
                    [1.0, 0.0],
                )?
                .region(BoundingBox::new(1, 2, 10, 12)?)
                .track_id("face-1")
                .detector_label("face")
                .detection_score(0.9)])
            }
        }

        let frame = frame(0);
        let mut analyzer = RecognitionVideoAnalyzer::new("identity", StaticBackend, library());
        let observations = analyzer.process_frame(&frame.as_frame()).unwrap();

        assert_eq!(observations.len(), 1);
        assert_eq!(observations[0].label.as_deref(), Some("Albert Einstein"));
        assert_eq!(observations[0].kind, ObservationKind::Face);
        assert_eq!(observations[0].track_id.as_deref(), Some("face-1"));
        assert_eq!(
            observations[0]
                .attributes
                .get("reference_id")
                .map(String::as_str),
            Some("einstein")
        );
    }

    #[test]
    fn temporal_analyzer_waits_for_minimum_hits() {
        struct StaticBackend;

        impl RecognitionBackend for StaticBackend {
            fn process_frame(
                &mut self,
                _frame: &VideoFrame<'_>,
            ) -> Result<Vec<RecognitionCandidate>> {
                Ok(vec![RecognitionCandidate::new(
                    ObservationKind::Face,
                    [1.0, 0.0],
                )?
                .track_id("face-1")])
            }
        }

        let mut analyzer = RecognitionVideoAnalyzer::new("identity", StaticBackend, library())
            .temporal_options(TemporalRecognitionOptions::track_window(2, 5, 0.8).unwrap());

        let first = analyzer.process_frame(&frame(0).as_frame()).unwrap();
        let second = analyzer.process_frame(&frame(1).as_frame()).unwrap();
        let third = analyzer.process_frame(&frame(2).as_frame()).unwrap();

        assert!(first.is_empty());
        assert_eq!(second.len(), 1);
        assert!(third.is_empty());
        assert_eq!(
            second[0]
                .attributes
                .get("temporal_hits")
                .map(String::as_str),
            Some("2")
        );
    }
}
