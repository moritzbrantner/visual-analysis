#![doc = include_str!("../README.md")]

pub mod surface;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use video_analysis_dataset::{
    AnalysisDataset, AnalysisEventRecord, DatasetRecord, FeatureRecord, FeatureValue,
    ObservationRecord, SceneRecord,
};

#[derive(Debug, Clone, PartialEq)]
/// Variants describing transform error.
pub enum TransformError {
    /// The invalid window seconds variant.
    InvalidWindowSeconds(f64),
    /// The invalid tolerance seconds variant.
    InvalidToleranceSeconds(f64),
}

impl fmt::Display for TransformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWindowSeconds(value) => {
                write!(f, "window seconds must be finite and positive, got {value}")
            }
            Self::InvalidToleranceSeconds(value) => {
                write!(
                    f,
                    "tolerance seconds must be finite and non-negative, got {value}"
                )
            }
        }
    }
}

impl Error for TransformError {}

/// Type alias for result.
pub type Result<T> = std::result::Result<T, TransformError>;

/// Returns record timestamp seconds.
pub fn record_timestamp_seconds(record: &DatasetRecord) -> Option<f64> {
    match record {
        DatasetRecord::VideoFrame(record) => Some(record.position.timestamp.seconds),
        DatasetRecord::AudioFrame(record) => Some(record.timestamp.seconds),
        DatasetRecord::TextSegment(record) => record.timestamp.map(|timestamp| timestamp.seconds),
        DatasetRecord::Scene(record) => Some(record.start.timestamp.seconds),
        DatasetRecord::Cut(record) => Some(record.position.timestamp.seconds),
        DatasetRecord::Observation(record) => record.timestamp.map(|timestamp| timestamp.seconds),
        DatasetRecord::Event(record) => record.timestamp.map(|timestamp| timestamp.seconds),
        DatasetRecord::Metric(_) => None,
        DatasetRecord::Feature(record) => record.timestamp.map(|timestamp| timestamp.seconds),
        DatasetRecord::Track(record) => record.first_timestamp.map(|timestamp| timestamp.seconds),
        DatasetRecord::Pose2d(record) => record.frame.map(|frame| frame.timestamp.seconds),
        DatasetRecord::Pose3d(record) => record.frame.map(|frame| frame.timestamp.seconds),
    }
}

/// Returns record frame index.
pub fn record_frame_index(record: &DatasetRecord) -> Option<u64> {
    match record {
        DatasetRecord::VideoFrame(record) => Some(record.position.frame_index),
        DatasetRecord::AudioFrame(_) => None,
        DatasetRecord::TextSegment(_) => None,
        DatasetRecord::Scene(record) => Some(record.start.frame_index),
        DatasetRecord::Cut(record) => Some(record.position.frame_index),
        DatasetRecord::Observation(record) => record.frame.map(|frame| frame.frame_index),
        DatasetRecord::Event(_) => None,
        DatasetRecord::Metric(record) => Some(record.frame_index),
        DatasetRecord::Feature(record) => record.frame_index,
        DatasetRecord::Track(record) => record.first_frame,
        DatasetRecord::Pose2d(record) => record.frame.map(|frame| frame.frame_index),
        DatasetRecord::Pose3d(record) => record.frame.map(|frame| frame.frame_index),
    }
}

/// Returns record scene index.
pub fn record_scene_index(record: &DatasetRecord) -> Option<u64> {
    match record {
        DatasetRecord::Scene(record) => Some(record.scene_index),
        DatasetRecord::Observation(record) => record.scene_index,
        DatasetRecord::Feature(record) => record.scene_index,
        _ => None,
    }
}

/// Returns sort by time.
pub fn sort_by_time(records: &mut [DatasetRecord]) {
    records.sort_by(|left, right| {
        match (
            record_timestamp_seconds(left),
            record_timestamp_seconds(right),
        ) {
            (Some(left), Some(right)) => left.total_cmp(&right),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => record_identity_key(left).cmp(&record_identity_key(right)),
        }
    });
}

/// Returns filter records.
pub fn filter_records(
    records: impl IntoIterator<Item = DatasetRecord>,
    mut predicate: impl FnMut(&DatasetRecord) -> bool,
) -> Vec<DatasetRecord> {
    records
        .into_iter()
        .filter(|record| predicate(record))
        .collect()
}

/// Returns filter dataset.
pub fn filter_dataset(
    dataset: &AnalysisDataset,
    mut predicate: impl FnMut(&DatasetRecord) -> bool,
) -> AnalysisDataset {
    AnalysisDataset {
        metadata: dataset.metadata.clone(),
        records: dataset
            .records
            .iter()
            .filter(|record| predicate(record))
            .cloned()
            .collect(),
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for time window.
pub struct TimeWindow {
    /// The index value.
    pub index: u64,
    /// The start seconds value.
    pub start_seconds: f64,
    /// The end seconds value.
    pub end_seconds: f64,
    /// The records value.
    pub records: Vec<DatasetRecord>,
}

/// Returns window by time.
pub fn window_by_time(records: &[DatasetRecord], seconds: f64) -> Result<Vec<TimeWindow>> {
    if seconds <= 0.0 || !seconds.is_finite() {
        return Err(TransformError::InvalidWindowSeconds(seconds));
    }

    let mut windows = BTreeMap::<u64, TimeWindow>::new();
    for record in records {
        let Some(timestamp) = record_timestamp_seconds(record) else {
            continue;
        };
        if timestamp < 0.0 || !timestamp.is_finite() {
            continue;
        }
        let index = (timestamp / seconds).floor() as u64;
        windows
            .entry(index)
            .or_insert_with(|| TimeWindow {
                index,
                start_seconds: index as f64 * seconds,
                end_seconds: (index + 1) as f64 * seconds,
                records: Vec::new(),
            })
            .records
            .push(record.clone());
    }
    Ok(windows.into_values().collect())
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for scene group.
pub struct SceneGroup {
    /// The scene value.
    pub scene: SceneRecord,
    /// The records value.
    pub records: Vec<DatasetRecord>,
}

/// Returns group by scene.
pub fn group_by_scene(dataset: &AnalysisDataset) -> Vec<SceneGroup> {
    let scenes = dataset.scenes().cloned().collect::<Vec<_>>();
    let mut groups = scenes
        .iter()
        .cloned()
        .map(|scene| SceneGroup {
            scene,
            records: Vec::new(),
        })
        .collect::<Vec<_>>();

    for record in &dataset.records {
        if matches!(record, DatasetRecord::Scene(_)) {
            continue;
        }

        let scene_index =
            record_scene_index(record).or_else(|| scene_index_by_bounds(&scenes, record));
        let Some(scene_index) = scene_index else {
            continue;
        };
        if let Some(group) = groups
            .iter_mut()
            .find(|group| group.scene.scene_index == scene_index)
        {
            group.records.push(record.clone());
        }
    }

    groups
}

fn scene_index_by_bounds(scenes: &[SceneRecord], record: &DatasetRecord) -> Option<u64> {
    if let Some(frame_index) = record_frame_index(record) {
        if let Some(scene) = scenes.iter().find(|scene| {
            scene.start.frame_index <= frame_index && frame_index <= scene.end.frame_index
        }) {
            return Some(scene.scene_index);
        }
    }

    let timestamp = record_timestamp_seconds(record)?;
    scenes
        .iter()
        .find(|scene| {
            scene.start.timestamp.seconds <= timestamp && timestamp <= scene.end.timestamp.seconds
        })
        .map(|scene| scene.scene_index)
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Data type for temporal join options.
pub struct TemporalJoinOptions {
    /// The tolerance seconds value.
    pub tolerance_seconds: f64,
}

impl TemporalJoinOptions {
    /// Creates a new value.
    pub fn new(tolerance_seconds: f64) -> Result<Self> {
        if tolerance_seconds < 0.0 || !tolerance_seconds.is_finite() {
            return Err(TransformError::InvalidToleranceSeconds(tolerance_seconds));
        }
        Ok(Self { tolerance_seconds })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Data type for frame join options.
pub struct FrameJoinOptions {
    /// The tolerance frames value.
    pub tolerance_frames: u64,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for joined record pair.
pub struct JoinedRecordPair {
    /// The left index value.
    pub left_index: usize,
    /// The right index value.
    pub right_index: usize,
    /// The delta value.
    pub delta: f64,
    /// The left value.
    pub left: DatasetRecord,
    /// The right value.
    pub right: DatasetRecord,
}

/// Returns join by time.
pub fn join_by_time(
    left: &[DatasetRecord],
    right: &[DatasetRecord],
    options: TemporalJoinOptions,
) -> Vec<JoinedRecordPair> {
    if options.tolerance_seconds < 0.0 || !options.tolerance_seconds.is_finite() {
        return Vec::new();
    }

    let mut joined = Vec::new();
    for (left_index, left_record) in left.iter().enumerate() {
        let Some(left_time) = record_timestamp_seconds(left_record) else {
            continue;
        };
        for (right_index, right_record) in right.iter().enumerate() {
            let Some(right_time) = record_timestamp_seconds(right_record) else {
                continue;
            };
            let delta = right_time - left_time;
            if delta.abs() <= options.tolerance_seconds {
                joined.push(JoinedRecordPair {
                    left_index,
                    right_index,
                    delta,
                    left: left_record.clone(),
                    right: right_record.clone(),
                });
            }
        }
    }
    joined
}

/// Returns join by frame.
pub fn join_by_frame(
    left: &[DatasetRecord],
    right: &[DatasetRecord],
    options: FrameJoinOptions,
) -> Vec<JoinedRecordPair> {
    let mut joined = Vec::new();
    for (left_index, left_record) in left.iter().enumerate() {
        let Some(left_frame) = record_frame_index(left_record) else {
            continue;
        };
        for (right_index, right_record) in right.iter().enumerate() {
            let Some(right_frame) = record_frame_index(right_record) else {
                continue;
            };
            let delta = right_frame as i128 - left_frame as i128;
            if delta.unsigned_abs() <= options.tolerance_frames as u128 {
                joined.push(JoinedRecordPair {
                    left_index,
                    right_index,
                    delta: delta as f64,
                    left: left_record.clone(),
                    right: right_record.clone(),
                });
            }
        }
    }
    joined
}

/// Returns record identity key.
pub fn record_identity_key(record: &DatasetRecord) -> Option<String> {
    match record {
        DatasetRecord::VideoFrame(record) => {
            Some(format!("video:{}:{}", record.stream_id, record.sequence))
        }
        DatasetRecord::AudioFrame(record) => {
            Some(format!("audio:{}:{}", record.stream_id, record.sequence))
        }
        DatasetRecord::TextSegment(record) => Some(format!(
            "text:{}:{}",
            record.stream_id, record.segment_index
        )),
        DatasetRecord::Scene(record) => Some(format!("scene:{}", record.scene_index)),
        DatasetRecord::Cut(record) => Some(format!(
            "cut:{}:{}",
            record.position.frame_index, record.detector
        )),
        DatasetRecord::Observation(record) => Some(observation_identity(record)),
        DatasetRecord::Event(record) => Some(event_identity(record)),
        DatasetRecord::Metric(record) => {
            Some(format!("metric:{}:{}", record.frame_index, record.key))
        }
        DatasetRecord::Feature(record) => Some(format!(
            "feature:{}:{:?}:{:?}:{:?}:{:?}",
            record.name, record.scope, record.timestamp, record.frame_index, record.scene_index
        )),
        DatasetRecord::Track(record) => Some(format!("track:{}", record.track_id)),
        DatasetRecord::Pose2d(record) => Some(format!(
            "pose2d:{}:{:?}:{:?}:{:?}",
            record.analyzer, record.id, record.label, record.frame
        )),
        DatasetRecord::Pose3d(record) => Some(format!(
            "pose3d:{}:{:?}:{:?}:{:?}",
            record.analyzer, record.id, record.label, record.frame
        )),
    }
}

fn observation_identity(record: &ObservationRecord) -> String {
    format!(
        "observation:{}:{:?}:{:?}:{:?}:{:?}:{:?}",
        record.analyzer, record.kind, record.label, record.timestamp, record.frame, record.track_id
    )
}

fn event_identity(record: &AnalysisEventRecord) -> String {
    format!(
        "event:{}:{}:{:?}",
        record.analyzer, record.label, record.timestamp
    )
}

/// Returns dedupe records.
pub fn dedupe_records(records: Vec<DatasetRecord>) -> Vec<DatasetRecord> {
    dedupe_by(records, record_identity_key)
}

/// Returns dedupe by.
pub fn dedupe_by(
    records: Vec<DatasetRecord>,
    mut key_fn: impl FnMut(&DatasetRecord) -> Option<String>,
) -> Vec<DatasetRecord> {
    let mut seen = BTreeSet::new();
    let mut deduped = Vec::new();
    for record in records {
        let Some(key) = key_fn(&record) else {
            deduped.push(record);
            continue;
        };
        if seen.insert(key) {
            deduped.push(record);
        }
    }
    deduped
}

/// Returns merge sorted by time.
pub fn merge_sorted_by_time(datasets: &[AnalysisDataset]) -> AnalysisDataset {
    let mut merged = datasets
        .first()
        .map(|dataset| AnalysisDataset::new(dataset.metadata.clone()))
        .unwrap_or_else(AnalysisDataset::empty);
    for dataset in datasets {
        merged.records.extend(dataset.records.iter().cloned());
    }
    sort_by_time(&mut merged.records);
    merged
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Variants describing aggregation.
pub enum Aggregation {
    /// The count variant.
    Count,
    /// The sum variant.
    Sum,
    /// The min variant.
    Min,
    /// The max variant.
    Max,
    /// The mean variant.
    Mean,
}

/// Returns resample numeric features.
pub fn resample_numeric_features(
    features: &[FeatureRecord],
    interval_seconds: f64,
    aggregation: Aggregation,
) -> Vec<FeatureRecord> {
    if interval_seconds <= 0.0 || !interval_seconds.is_finite() {
        return Vec::new();
    }

    let mut buckets = BTreeMap::<(String, Option<String>, u64), Vec<f64>>::new();
    for feature in features {
        let FeatureValue::Number(value) = feature.value else {
            continue;
        };
        if !value.is_finite() {
            continue;
        }
        let Some(timestamp) = feature.timestamp else {
            continue;
        };
        if timestamp.seconds < 0.0 || !timestamp.seconds.is_finite() {
            continue;
        }
        let bucket_index = (timestamp.seconds / interval_seconds).floor() as u64;
        buckets
            .entry((feature.name.clone(), feature.scope.clone(), bucket_index))
            .or_default()
            .push(value);
    }

    buckets
        .into_iter()
        .filter_map(|((name, scope, bucket_index), values)| {
            aggregate(&values, aggregation).map(|value| {
                let mut feature = FeatureRecord::number(name, value)
                    .scope(format!("window:{bucket_index}"))
                    .attribute("aggregation", aggregation.as_str())
                    .attribute("interval_seconds", interval_seconds.to_string())
                    .attribute(
                        "window_start_seconds",
                        (bucket_index as f64 * interval_seconds).to_string(),
                    )
                    .attribute(
                        "window_end_seconds",
                        ((bucket_index + 1) as f64 * interval_seconds).to_string(),
                    );
                if let Some(scope) = scope {
                    feature = feature.attribute("source_scope", scope);
                }
                feature
            })
        })
        .collect()
}

impl Aggregation {
    /// Borrows this value as a str.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Count => "count",
            Self::Sum => "sum",
            Self::Min => "min",
            Self::Max => "max",
            Self::Mean => "mean",
        }
    }
}

fn aggregate(values: &[f64], aggregation: Aggregation) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(match aggregation {
        Aggregation::Count => values.len() as f64,
        Aggregation::Sum => values.iter().sum(),
        Aggregation::Min => values.iter().copied().reduce(f64::min)?,
        Aggregation::Max => values.iter().copied().reduce(f64::max)?,
        Aggregation::Mean => values.iter().sum::<f64>() / values.len() as f64,
    })
}

#[cfg(test)]
mod tests {
    use num_rational::Rational64;
    use video_analysis_core::{AnalysisEvent, FramePosition, Observation, ObservationKind, Scene};
    use video_analysis_dataset::{
        AnalysisEventRecord, DatasetMetadata, FeatureRecord, SceneRecord, TimestampRecord,
    };

    use super::*;

    fn position(frame_index: u64) -> FramePosition {
        FramePosition::from_frame_index(frame_index, Rational64::new(10, 1))
    }

    fn scene(index: u64, start: u64, end: u64) -> DatasetRecord {
        DatasetRecord::Scene(SceneRecord::from_scene(
            index,
            &Scene {
                start: position(start),
                end: position(end),
            },
        ))
    }

    fn observation(frame: u64, scene_index: Option<u64>, label: &str) -> DatasetRecord {
        let mut observation = Observation::new("objects", ObservationKind::Object)
            .at_frame(position(frame))
            .label(label);
        if let Some(scene_index) = scene_index {
            observation = observation.in_scene(scene_index);
        }
        DatasetRecord::Observation(video_analysis_dataset::ObservationRecord::from_observation(
            observation,
        ))
    }

    #[test]
    fn sorts_mixed_records_by_time() {
        let mut records = vec![observation(20, None, "late"), observation(5, None, "early")];
        sort_by_time(&mut records);
        assert_eq!(record_frame_index(&records[0]), Some(5));
    }

    #[test]
    fn filters_dataset_and_preserves_metadata() {
        let mut dataset = AnalysisDataset::new(DatasetMetadata {
            name: Some("sample".to_string()),
            ..DatasetMetadata::default()
        });
        dataset.push(observation(1, None, "person"));
        dataset.push(scene(0, 0, 10));

        let filtered = filter_dataset(&dataset, |record| matches!(record, DatasetRecord::Scene(_)));
        assert_eq!(filtered.metadata.name.as_deref(), Some("sample"));
        assert_eq!(filtered.records.len(), 1);
    }

    #[test]
    fn groups_by_explicit_scene_index() {
        let mut dataset = AnalysisDataset::empty();
        dataset.push(scene(0, 0, 10));
        dataset.push(scene(1, 10, 20));
        dataset.push(observation(3, Some(1), "person"));

        let groups = group_by_scene(&dataset);
        assert_eq!(groups[1].records.len(), 1);
    }

    #[test]
    fn groups_by_scene_bounds_when_scene_index_is_absent() {
        let mut dataset = AnalysisDataset::empty();
        dataset.push(scene(0, 0, 10));
        dataset.push(scene(1, 11, 20));
        dataset.push(observation(15, None, "car"));

        let groups = group_by_scene(&dataset);
        assert_eq!(groups[1].records.len(), 1);
    }

    #[test]
    fn joins_by_time_and_frame_tolerance() {
        let left = vec![observation(10, None, "person")];
        let right = vec![
            observation(11, None, "person"),
            observation(20, None, "car"),
        ];

        assert_eq!(
            join_by_time(
                &left,
                &right,
                TemporalJoinOptions {
                    tolerance_seconds: 0.11
                }
            )
            .len(),
            1
        );
        assert_eq!(
            join_by_frame(
                &left,
                &right,
                FrameJoinOptions {
                    tolerance_frames: 1
                }
            )
            .len(),
            1
        );
    }

    #[test]
    fn dedupes_records_with_stable_identity_keys() {
        let records = vec![
            observation(10, None, "person"),
            observation(10, None, "person"),
        ];
        assert_eq!(dedupe_records(records).len(), 1);
    }

    #[test]
    fn windows_by_time_and_rejects_invalid_windows() {
        let records = vec![
            observation(1, None, "person"),
            observation(15, None, "person"),
        ];
        let windows = window_by_time(&records, 1.0).unwrap();
        assert_eq!(windows.len(), 2);
        assert!(window_by_time(&records, 0.0).is_err());
    }

    #[test]
    fn resamples_numeric_feature_records() {
        let mut first = FeatureRecord::number("score", 2.0);
        first.timestamp = Some(TimestampRecord {
            pts: 0,
            timebase_num: 1,
            timebase_den: 1,
            seconds: 0.0,
        });
        let mut second = FeatureRecord::number("score", 4.0);
        second.timestamp = Some(TimestampRecord {
            pts: 1,
            timebase_num: 1,
            timebase_den: 1,
            seconds: 1.0,
        });

        let resampled = resample_numeric_features(&[first, second], 5.0, Aggregation::Mean);
        assert_eq!(resampled.len(), 1);
        assert_eq!(resampled[0].value, FeatureValue::Number(3.0));
    }

    #[test]
    fn rejects_invalid_tolerances_through_constructor() {
        assert!(TemporalJoinOptions::new(-1.0).is_err());
    }

    #[test]
    fn event_timestamp_is_discovered() {
        let event = AnalysisEvent::new("audio", "speech").at_timestamp(position(3).timestamp);
        let record = DatasetRecord::Event(AnalysisEventRecord::from_event(event));
        let seconds = record_timestamp_seconds(&record).unwrap();
        assert!((seconds - 0.3).abs() < f64::EPSILON);
    }
}
