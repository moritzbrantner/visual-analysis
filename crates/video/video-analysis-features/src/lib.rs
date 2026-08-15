#![doc = include_str!("../README.md")]

pub mod surface;
use std::collections::BTreeMap;

use video_analysis_core::{DetectError, Result};
use video_analysis_dataset::{
    AnalysisDataset, DatasetRecord, FeatureRecord, FeatureValue, ObservationKindRecord,
    ObservationRecord,
};
use video_analysis_transform::{group_by_scene, window_by_time};

/// Trait for feature extractor implementations.
pub trait FeatureExtractor {
    /// Returns name.
    fn name(&self) -> &str;
    /// Returns extract.
    fn extract(&self, dataset: &AnalysisDataset) -> Result<Vec<FeatureRecord>>;
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Variants describing feature scope.
pub enum FeatureScope {
    /// The global variant.
    Global,
    /// The per scene variant.
    PerScene,
    /// The fixed window variant.
    FixedWindow {
        /// The seconds value for this variant.
        seconds: f64,
    },
}

#[derive(Default)]
/// Data type for feature pipeline builder.
pub struct FeaturePipelineBuilder {
    extractors: Vec<Box<dyn FeatureExtractor>>,
}

impl FeaturePipelineBuilder {
    /// Returns extractor.
    pub fn extractor<E: FeatureExtractor + 'static>(mut self, extractor: E) -> Self {
        self.extractors.push(Box::new(extractor));
        self
    }

    /// Returns build.
    pub fn build(self) -> FeaturePipeline {
        FeaturePipeline {
            extractors: self.extractors,
        }
    }
}

/// Data type for feature pipeline.
pub struct FeaturePipeline {
    extractors: Vec<Box<dyn FeatureExtractor>>,
}

impl FeaturePipeline {
    /// Returns builder.
    pub fn builder() -> FeaturePipelineBuilder {
        FeaturePipelineBuilder::default()
    }

    /// Returns extract.
    pub fn extract(&self, dataset: &AnalysisDataset) -> Result<Vec<FeatureRecord>> {
        let mut features = Vec::new();
        for extractor in &self.extractors {
            features.extend(extractor.extract(dataset)?);
        }
        Ok(features)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Data type for scene stats extractor.
pub struct SceneStatsExtractor;

impl FeatureExtractor for SceneStatsExtractor {
    fn name(&self) -> &str {
        "scene_stats"
    }

    fn extract(&self, dataset: &AnalysisDataset) -> Result<Vec<FeatureRecord>> {
        let mut features = Vec::new();
        for group in group_by_scene(dataset) {
            let scene_index = group.scene.scene_index;
            let mut observation_count = 0_i64;
            let mut text_observations = 0_i64;
            let mut face_observations = 0_i64;
            let mut object_observations = 0_i64;
            let mut scene_observations = 0_i64;
            let mut custom_observations = 0_i64;
            let mut events = 0_i64;
            let mut audio_events = 0_i64;
            let mut text_events = 0_i64;

            for record in &group.records {
                match record {
                    DatasetRecord::Observation(observation) => {
                        observation_count += 1;
                        match &observation.kind {
                            ObservationKindRecord::Text => text_observations += 1,
                            ObservationKindRecord::Face => face_observations += 1,
                            ObservationKindRecord::Object => object_observations += 1,
                            ObservationKindRecord::Scene => scene_observations += 1,
                            ObservationKindRecord::Custom(_) => custom_observations += 1,
                        }
                    }
                    DatasetRecord::Event(event) => {
                        events += 1;
                        if event.analyzer.contains("audio") {
                            audio_events += 1;
                        }
                        if event.analyzer.contains("text") {
                            text_events += 1;
                        }
                    }
                    _ => {}
                }
            }

            features.push(
                FeatureRecord::number("scene.duration_seconds", group.scene.duration_seconds)
                    .scope("scene")
                    .scene_index(scene_index),
            );
            features.push(
                FeatureRecord::integer("scene.observation_count", observation_count)
                    .scope("scene")
                    .scene_index(scene_index),
            );
            for (name, value) in [
                ("scene.text_observations", text_observations),
                ("scene.face_observations", face_observations),
                ("scene.object_observations", object_observations),
                ("scene.scene_observations", scene_observations),
                ("scene.custom_observations", custom_observations),
                ("scene.event_count", events),
                ("scene.audio_event_count", audio_events),
                ("scene.text_event_count", text_events),
            ] {
                features.push(
                    FeatureRecord::integer(name, value)
                        .scope("scene")
                        .scene_index(scene_index),
                );
            }
        }
        Ok(features)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Data type for observation label histogram extractor.
pub struct ObservationLabelHistogramExtractor {
    /// The scope value.
    pub scope: FeatureScope,
}

impl Default for ObservationLabelHistogramExtractor {
    fn default() -> Self {
        Self {
            scope: FeatureScope::Global,
        }
    }
}

impl FeatureExtractor for ObservationLabelHistogramExtractor {
    fn name(&self) -> &str {
        "observation_label_histogram"
    }

    fn extract(&self, dataset: &AnalysisDataset) -> Result<Vec<FeatureRecord>> {
        match self.scope {
            FeatureScope::Global => Ok(vec![FeatureRecord::histogram(
                "observation.label_histogram",
                observation_label_histogram(dataset.records.iter()),
            )
            .scope("global")]),
            FeatureScope::PerScene => Ok(group_by_scene(dataset)
                .into_iter()
                .map(|group| {
                    FeatureRecord::histogram(
                        "observation.label_histogram",
                        observation_label_histogram(group.records.iter()),
                    )
                    .scope("scene")
                    .scene_index(group.scene.scene_index)
                })
                .collect()),
            FeatureScope::FixedWindow { seconds } => {
                let windows = window_by_time(&dataset.records, seconds).map_err(|err| {
                    DetectError::InvalidArgument(format!("invalid feature window: {err}"))
                })?;
                Ok(windows
                    .into_iter()
                    .map(|window| {
                        FeatureRecord::histogram(
                            "observation.label_histogram",
                            observation_label_histogram(window.records.iter()),
                        )
                        .scope(format!("window:{}", window.index))
                        .attribute("window_start_seconds", window.start_seconds.to_string())
                        .attribute("window_end_seconds", window.end_seconds.to_string())
                    })
                    .collect())
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Data type for transcript stats extractor.
pub struct TranscriptStatsExtractor;

impl FeatureExtractor for TranscriptStatsExtractor {
    fn name(&self) -> &str {
        "transcript_stats"
    }

    fn extract(&self, dataset: &AnalysisDataset) -> Result<Vec<FeatureRecord>> {
        let mut segments = 0_i64;
        let mut bytes = 0_i64;
        let mut chars = 0_i64;
        let mut words = 0_i64;
        for segment in dataset.records.iter().filter_map(|record| match record {
            DatasetRecord::TextSegment(segment) => Some(segment),
            _ => None,
        }) {
            segments += 1;
            bytes += segment.text.len() as i64;
            chars += segment.text.chars().count() as i64;
            words += segment.text.split_whitespace().count() as i64;
        }
        Ok(vec![
            FeatureRecord::integer("transcript.segment_count", segments).scope("global"),
            FeatureRecord::integer("transcript.byte_count", bytes).scope("global"),
            FeatureRecord::integer("transcript.char_count", chars).scope("global"),
            FeatureRecord::integer("transcript.word_count", words).scope("global"),
        ])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Data type for audio event stats extractor.
pub struct AudioEventStatsExtractor;

impl FeatureExtractor for AudioEventStatsExtractor {
    fn name(&self) -> &str {
        "audio_event_stats"
    }

    fn extract(&self, dataset: &AnalysisDataset) -> Result<Vec<FeatureRecord>> {
        let mut counts = BTreeMap::<String, u64>::new();
        let mut scores = BTreeMap::<String, (f64, u64)>::new();
        for event in dataset.records.iter().filter_map(|record| match record {
            DatasetRecord::Event(event) if event.analyzer.contains("audio") => Some(event),
            _ => None,
        }) {
            *counts.entry(event.label.clone()).or_default() += 1;
            if let Some(score) = event.score {
                if score.is_finite() {
                    let entry = scores.entry(event.label.clone()).or_default();
                    entry.0 += score as f64;
                    entry.1 += 1;
                }
            }
        }

        let mut features =
            vec![FeatureRecord::histogram("audio.event_counts", counts).scope("global")];
        for (label, (sum, count)) in scores {
            if count > 0 {
                features.push(
                    FeatureRecord::number("audio.event_mean_score", sum / count as f64)
                        .scope("global")
                        .attribute("label", label),
                );
            }
        }
        Ok(features)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Data type for track summary extractor.
pub struct TrackSummaryExtractor;

impl FeatureExtractor for TrackSummaryExtractor {
    fn name(&self) -> &str {
        "track_summary"
    }

    fn extract(&self, dataset: &AnalysisDataset) -> Result<Vec<FeatureRecord>> {
        let mut tracks = BTreeMap::<String, TrackAccumulator>::new();
        for observation in dataset.observations() {
            let Some(track_id) = &observation.track_id else {
                continue;
            };
            tracks
                .entry(track_id.clone())
                .or_default()
                .push(observation);
        }

        let mut features = Vec::new();
        for (track_id, track) in tracks {
            features.push(
                FeatureRecord::integer("track.hit_count", track.hits as i64)
                    .scope("track")
                    .track_id(track_id.clone()),
            );
            if let Some(first_frame) = track.first_frame {
                features.push(
                    FeatureRecord::integer("track.first_frame", first_frame as i64)
                        .scope("track")
                        .track_id(track_id.clone()),
                );
            }
            if let Some(last_frame) = track.last_frame {
                features.push(
                    FeatureRecord::integer("track.last_frame", last_frame as i64)
                        .scope("track")
                        .track_id(track_id.clone()),
                );
            }
            if let (Some(first), Some(last)) = (track.first_timestamp, track.last_timestamp) {
                features.push(
                    FeatureRecord::number("track.duration_seconds", (last - first).max(0.0))
                        .scope("track")
                        .track_id(track_id.clone()),
                );
            }
            features.push(
                FeatureRecord::histogram("track.label_histogram", track.labels)
                    .scope("track")
                    .track_id(track_id),
            );
        }
        Ok(features)
    }
}

#[derive(Debug, Default, Clone)]
struct TrackAccumulator {
    hits: u64,
    first_frame: Option<u64>,
    last_frame: Option<u64>,
    first_timestamp: Option<f64>,
    last_timestamp: Option<f64>,
    labels: BTreeMap<String, u64>,
}

impl TrackAccumulator {
    fn push(&mut self, observation: &ObservationRecord) {
        self.hits += 1;
        if let Some(frame) = observation.frame {
            self.first_frame = Some(
                self.first_frame
                    .map_or(frame.frame_index, |value| value.min(frame.frame_index)),
            );
            self.last_frame = Some(
                self.last_frame
                    .map_or(frame.frame_index, |value| value.max(frame.frame_index)),
            );
        }
        if let Some(timestamp) = observation.timestamp {
            self.first_timestamp = Some(
                self.first_timestamp
                    .map_or(timestamp.seconds, |value| value.min(timestamp.seconds)),
            );
            self.last_timestamp = Some(
                self.last_timestamp
                    .map_or(timestamp.seconds, |value| value.max(timestamp.seconds)),
            );
        }
        *self
            .labels
            .entry(
                observation
                    .label
                    .clone()
                    .unwrap_or_else(|| "unlabeled".to_string()),
            )
            .or_default() += 1;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Data type for feature vector mean extractor.
pub struct FeatureVectorMeanExtractor;

impl FeatureExtractor for FeatureVectorMeanExtractor {
    fn name(&self) -> &str {
        "feature_vector_mean"
    }

    fn extract(&self, dataset: &AnalysisDataset) -> Result<Vec<FeatureRecord>> {
        let mut groups = BTreeMap::<(String, Option<String>), VectorAccumulator>::new();
        for feature in dataset.features() {
            let FeatureValue::Vector(values) = &feature.value else {
                continue;
            };
            groups
                .entry((feature.name.clone(), feature.scope.clone()))
                .or_default()
                .push(values);
        }

        Ok(groups
            .into_iter()
            .filter_map(|((name, scope), accumulator)| {
                accumulator.mean().map(|mean| {
                    let mut feature =
                        FeatureRecord::new(format!("{name}.mean"), FeatureValue::Vector(mean));
                    if let Some(scope) = scope {
                        feature = feature.scope(scope);
                    }
                    feature.attribute("aggregation", "mean")
                })
            })
            .collect())
    }
}

#[derive(Debug, Default, Clone)]
struct VectorAccumulator {
    dimensions: Option<usize>,
    count: u64,
    mean: Vec<f64>,
}

impl VectorAccumulator {
    fn push(&mut self, values: &[f32]) {
        if values.iter().any(|value| !value.is_finite()) {
            return;
        }
        match self.dimensions {
            Some(dimensions) if dimensions != values.len() => return,
            None => {
                self.dimensions = Some(values.len());
                self.mean.resize(values.len(), 0.0);
            }
            _ => {}
        }

        self.count += 1;
        for (mean, value) in self.mean.iter_mut().zip(values) {
            let value = *value as f64;
            *mean += (value - *mean) / self.count as f64;
        }
    }

    fn mean(self) -> Option<Vec<f32>> {
        (self.count > 0).then(|| self.mean.into_iter().map(|value| value as f32).collect())
    }
}

fn observation_label_histogram<'a>(
    records: impl Iterator<Item = &'a DatasetRecord>,
) -> BTreeMap<String, u64> {
    let mut histogram = BTreeMap::new();
    for observation in records.filter_map(|record| match record {
        DatasetRecord::Observation(observation) => Some(observation),
        _ => None,
    }) {
        *histogram
            .entry(
                observation
                    .label
                    .clone()
                    .unwrap_or_else(|| "unlabeled".to_string()),
            )
            .or_default() += 1;
    }
    histogram
}

#[cfg(test)]
mod tests {
    use num_rational::Rational64;
    use video_analysis_core::{
        AnalysisEvent, FramePosition, Observation, ObservationKind, OwnedTextSegment, Scene,
    };
    use video_analysis_dataset::{
        AnalysisEventRecord, DatasetMetadata, FeatureRecord, SceneRecord, TextSegmentRecord,
    };

    use super::*;

    fn position(frame_index: u64) -> FramePosition {
        FramePosition::from_frame_index(frame_index, Rational64::new(10, 1))
    }

    fn dataset_with_scene() -> AnalysisDataset {
        let mut dataset = AnalysisDataset::new(DatasetMetadata::default());
        dataset.push(DatasetRecord::Scene(SceneRecord::from_scene(
            0,
            &Scene {
                start: position(0),
                end: position(20),
            },
        )));
        dataset
    }

    #[test]
    fn scene_stats_count_observations_and_events() {
        let mut dataset = dataset_with_scene();
        dataset.push(DatasetRecord::Observation(
            video_analysis_dataset::ObservationRecord::from_observation(
                Observation::new("objects", ObservationKind::Object)
                    .at_frame(position(5))
                    .in_scene(0)
                    .label("person"),
            ),
        ));
        dataset.push(DatasetRecord::Event(AnalysisEventRecord::from_event(
            AnalysisEvent::new("audio_onset", "audio:onset").at_timestamp(position(7).timestamp),
        )));

        let features = SceneStatsExtractor.extract(&dataset).unwrap();
        assert!(features.iter().any(|feature| {
            feature.name == "scene.object_observations" && feature.value == FeatureValue::Integer(1)
        }));
        assert!(features.iter().any(|feature| {
            feature.name == "scene.audio_event_count" && feature.value == FeatureValue::Integer(1)
        }));
    }

    #[test]
    fn label_histogram_uses_unlabeled_bucket() {
        let mut dataset = AnalysisDataset::empty();
        dataset.push(DatasetRecord::Observation(
            video_analysis_dataset::ObservationRecord::from_observation(Observation::new(
                "ocr",
                ObservationKind::Text,
            )),
        ));

        let features = ObservationLabelHistogramExtractor::default()
            .extract(&dataset)
            .unwrap();
        let FeatureValue::Histogram(histogram) = &features[0].value else {
            panic!("expected histogram");
        };
        assert_eq!(histogram["unlabeled"], 1);
    }

    #[test]
    fn transcript_stats_count_text() {
        let segment = OwnedTextSegment::new(0, "hello large world");
        let mut dataset = AnalysisDataset::empty();
        dataset.push(DatasetRecord::TextSegment(TextSegmentRecord::from_segment(
            "text:0",
            &segment.as_segment(),
        )));

        let features = TranscriptStatsExtractor.extract(&dataset).unwrap();
        assert!(features.iter().any(|feature| {
            feature.name == "transcript.word_count" && feature.value == FeatureValue::Integer(3)
        }));
    }

    #[test]
    fn audio_event_stats_count_and_average_scores() {
        let mut dataset = AnalysisDataset::empty();
        dataset.push(DatasetRecord::Event(AnalysisEventRecord::from_event(
            AnalysisEvent::new("audio_pitch", "pitch").score(0.5),
        )));
        dataset.push(DatasetRecord::Event(AnalysisEventRecord::from_event(
            AnalysisEvent::new("audio_pitch", "pitch").score(1.0),
        )));

        let features = AudioEventStatsExtractor.extract(&dataset).unwrap();
        assert!(features.iter().any(|feature| {
            feature.name == "audio.event_mean_score" && feature.value == FeatureValue::Number(0.75)
        }));
    }

    #[test]
    fn track_summaries_group_by_track_id() {
        let mut dataset = AnalysisDataset::empty();
        dataset.push(DatasetRecord::Observation(
            video_analysis_dataset::ObservationRecord::from_observation(
                Observation::new("objects", ObservationKind::Object)
                    .at_frame(position(1))
                    .label("person")
                    .track_id("t1"),
            ),
        ));
        dataset.push(DatasetRecord::Observation(
            video_analysis_dataset::ObservationRecord::from_observation(
                Observation::new("objects", ObservationKind::Object)
                    .at_frame(position(5))
                    .label("person")
                    .track_id("t1"),
            ),
        ));
        dataset.push(DatasetRecord::Observation(
            video_analysis_dataset::ObservationRecord::from_observation(
                Observation::new("objects", ObservationKind::Object).at_frame(position(8)),
            ),
        ));

        let features = TrackSummaryExtractor.extract(&dataset).unwrap();
        assert!(features.iter().any(|feature| {
            feature.name == "track.hit_count"
                && feature.track_id.as_deref() == Some("t1")
                && feature.value == FeatureValue::Integer(2)
        }));
    }

    #[test]
    fn feature_vector_mean_skips_mismatched_dimensions() {
        let mut dataset = AnalysisDataset::empty();
        dataset.push(DatasetRecord::Feature(
            FeatureRecord::new("embedding", FeatureValue::Vector(vec![1.0, 3.0])).scope("global"),
        ));
        dataset.push(DatasetRecord::Feature(
            FeatureRecord::new("embedding", FeatureValue::Vector(vec![3.0, 5.0])).scope("global"),
        ));
        dataset.push(DatasetRecord::Feature(
            FeatureRecord::new("embedding", FeatureValue::Vector(vec![100.0])).scope("global"),
        ));

        let features = FeatureVectorMeanExtractor.extract(&dataset).unwrap();
        assert_eq!(features[0].value, FeatureValue::Vector(vec![2.0, 4.0]));
    }

    #[test]
    fn feature_pipeline_runs_multiple_extractors() {
        let dataset = dataset_with_scene();
        let pipeline = FeaturePipeline::builder()
            .extractor(SceneStatsExtractor)
            .extractor(TranscriptStatsExtractor)
            .build();

        let features = pipeline.extract(&dataset).unwrap();
        assert!(features
            .iter()
            .any(|feature| feature.name == "scene.duration_seconds"));
        assert!(features
            .iter()
            .any(|feature| feature.name == "transcript.segment_count"));
    }
}
