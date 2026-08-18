//! Library-owned runtime surface for `video-analysis-transform`.

use std::collections::{BTreeMap, BTreeSet};

use runtime_core::{
    OperationId, PackageSurface, RuntimeCapabilities, SurfaceOperation, SurfaceRequest,
    SurfaceResponse,
};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use video_analysis_dataset::{AnalysisDataset, DatasetRecord, FeatureRecord};

use crate::{group_by_scene, resample_numeric_features, window_by_time, Aggregation};

const DEFAULT_FILTER_LIMIT: usize = 1000;
const MAX_FILTER_LIMIT: usize = 5000;

/// Returns the package surface exposed by every transport wrapper.
pub fn package_surface() -> PackageSurface {
    PackageSurface {
        library: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities: RuntimeCapabilities::pure_rust(),
        operations: vec![
            operation("describe", "Describe package", "Filtering, joins, grouping, and resampling for video-analysis datasets.", serde_json::json!({"includeOperations": true})),
            operation("video.transform.filter", "Filter dataset", "Filters retained dataset records by kind, time, scene, label, and capped result size.", serde_json::json!({"dataset": {"metadata": {"schema_version": 2, "name": null, "source": null, "created_at": null, "attributes": {}}, "records": []}, "kinds": ["observation"], "limit": 1000})),
            operation("video.transform.window", "Window dataset", "Groups records into fixed-duration time windows.", serde_json::json!({"dataset": {"metadata": {"schema_version": 2, "name": null, "source": null, "created_at": null, "attributes": {}}, "records": []}, "seconds": 5.0})),
            operation("video.transform.groupScenes", "Group scenes", "Groups records under scene intervals using explicit scene indexes or frame bounds.", serde_json::json!({"dataset": {"metadata": {"schema_version": 2, "name": null, "source": null, "created_at": null, "attributes": {}}, "records": []}})),
            operation("video.transform.resampleFeatures", "Resample numeric features", "Aggregates numeric feature records into fixed time windows.", serde_json::json!({"dataset": {"metadata": {"schema_version": 2, "name": null, "source": null, "created_at": null, "attributes": {}}, "records": []}, "featureName": "metric", "seconds": 1.0, "aggregation": "mean"})),
        ],
    }
}

fn operation(
    id: &str,
    name: &str,
    description: &str,
    example_request: serde_json::Value,
) -> SurfaceOperation {
    SurfaceOperation {
        id: OperationId::new(id),
        name: name.to_string(),
        description: Some(description.to_string()),
        curation: runtime_core::SurfaceOperationCuration::from_operation_id(id),
        input_schema: serde_json::json!({"type": "object", "additionalProperties": true, "xOperationCategory": runtime_core::operation_category(id)}),
        output_schema: serde_json::json!({"type": "object", "xOperationCategory": runtime_core::operation_category(id)}),
        example_request,
        wasm_supported: true,
        server_supported: true,
    }
}

/// Runs one library-owned operation.
pub fn run_surface_operation(request: SurfaceRequest) -> Result<SurfaceResponse, String> {
    let operation = request.operation.clone();
    let value = match request.operation.as_str() {
        "describe" => describe_value(request.input),
        "video.transform.filter" => filter_value(parse_input(request.input)?)?,
        "video.transform.window" => window_value(parse_input(request.input)?)?,
        "video.transform.groupScenes" => group_scenes_value(parse_input(request.input)?)?,
        "video.transform.resampleFeatures" => resample_features_value(parse_input(request.input)?)?,
        operation => {
            return Err(format!(
                "unsupported operation `{operation}` for {}",
                env!("CARGO_PKG_NAME")
            ))
        }
    };
    Ok(surface_response(operation, value))
}

fn describe_value(input: serde_json::Value) -> serde_json::Value {
    let surface = package_surface();
    serde_json::json!({
        "library": surface.library,
        "version": surface.version,
        "operationCount": surface.operations.len(),
        "operations": surface.operations.iter().map(|operation| operation.id.as_str()).collect::<Vec<_>>(),
        "input": input
    })
}

fn surface_response(operation: OperationId, value: serde_json::Value) -> SurfaceResponse {
    SurfaceResponse {
        operation,
        value,
        diagnostics: Vec::new(),
        artifacts: Vec::new(),
    }
}

fn parse_input<T: DeserializeOwned>(input: serde_json::Value) -> Result<T, String> {
    serde_json::from_value(input).map_err(|error| format!("invalid request: {error}"))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FilterRequest {
    dataset: AnalysisDataset,
    #[serde(default)]
    kinds: Vec<String>,
    start_seconds: Option<f64>,
    end_seconds: Option<f64>,
    #[serde(default)]
    scene_indexes: Vec<u64>,
    #[serde(default)]
    labels: Vec<String>,
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WindowRequest {
    dataset: AnalysisDataset,
    seconds: f64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DatasetInput {
    dataset: AnalysisDataset,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ResampleFeaturesRequest {
    dataset: AnalysisDataset,
    feature_name: String,
    seconds: f64,
    aggregation: String,
}

fn filter_value(request: FilterRequest) -> Result<serde_json::Value, String> {
    let limit = request.limit.unwrap_or(DEFAULT_FILTER_LIMIT);
    if limit > MAX_FILTER_LIMIT {
        return Err(format!("limit must be <= {MAX_FILTER_LIMIT}"));
    }
    if let Some(start) = request.start_seconds {
        if !start.is_finite() {
            return Err("startSeconds must be finite".to_string());
        }
    }
    if let Some(end) = request.end_seconds {
        if !end.is_finite() {
            return Err("endSeconds must be finite".to_string());
        }
    }

    let kinds = request.kinds.into_iter().collect::<BTreeSet<_>>();
    let scene_indexes = request.scene_indexes.into_iter().collect::<BTreeSet<_>>();
    let labels = request.labels.into_iter().collect::<BTreeSet<_>>();
    let input_record_count = request.dataset.records.len();
    let filtered = request
        .dataset
        .records
        .iter()
        .filter(|record| {
            (kinds.is_empty() || kinds.contains(record.kind()))
                && request.start_seconds.is_none_or(|start| {
                    record_timestamp_seconds(record).is_some_and(|value| value >= start)
                })
                && request.end_seconds.is_none_or(|end| {
                    record_timestamp_seconds(record).is_some_and(|value| value <= end)
                })
                && (scene_indexes.is_empty()
                    || record_scene_index(record)
                        .is_some_and(|index| scene_indexes.contains(&index)))
                && (labels.is_empty()
                    || record_label(record).is_some_and(|label| labels.contains(label)))
        })
        .cloned()
        .collect::<Vec<_>>();

    let output_records = filtered.iter().take(limit).cloned().collect::<Vec<_>>();
    let mut dataset = request.dataset;
    dataset.records = output_records;
    Ok(serde_json::json!({
        "dataset": dataset,
        "inputRecordCount": input_record_count,
        "outputRecordCount": dataset.records.len(),
        "totalMatching": filtered.len(),
        "truncated": filtered.len() > dataset.records.len()
    }))
}

fn window_value(request: WindowRequest) -> Result<serde_json::Value, String> {
    let windows = window_by_time(&request.dataset.records, request.seconds)
        .map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "windows": windows
            .iter()
            .map(|window| serde_json::json!({
                "index": window.index,
                "startSeconds": window.start_seconds,
                "endSeconds": window.end_seconds,
                "recordCount": window.records.len(),
                "recordCounts": record_counts(&window.records)
            }))
            .collect::<Vec<_>>()
    }))
}

fn group_scenes_value(request: DatasetInput) -> Result<serde_json::Value, String> {
    let groups = group_by_scene(&request.dataset);
    Ok(serde_json::json!({
        "groups": groups
            .iter()
            .map(|group| serde_json::json!({
                "sceneIndex": group.scene.scene_index,
                "startFrame": group.scene.start.frame_index,
                "endFrame": group.scene.end.frame_index,
                "recordCount": group.records.len(),
                "recordCounts": record_counts(&group.records)
            }))
            .collect::<Vec<_>>()
    }))
}

fn resample_features_value(request: ResampleFeaturesRequest) -> Result<serde_json::Value, String> {
    let aggregation = parse_aggregation(&request.aggregation)?;
    let features = request
        .dataset
        .features()
        .filter(|feature| feature.name == request.feature_name)
        .cloned()
        .collect::<Vec<FeatureRecord>>();
    let resampled = resample_numeric_features(&features, request.seconds, aggregation);
    Ok(serde_json::json!({
        "features": resampled,
        "featureCount": resampled.len(),
        "sourceFeatureCount": features.len(),
        "aggregation": aggregation.as_str()
    }))
}

fn parse_aggregation(value: &str) -> Result<Aggregation, String> {
    match value {
        "count" => Ok(Aggregation::Count),
        "sum" => Ok(Aggregation::Sum),
        "min" => Ok(Aggregation::Min),
        "max" => Ok(Aggregation::Max),
        "mean" => Ok(Aggregation::Mean),
        other => Err(format!("unsupported aggregation `{other}`")),
    }
}

fn record_counts(records: &[DatasetRecord]) -> BTreeMap<String, u64> {
    let mut counts = BTreeMap::<String, u64>::new();
    for record in records {
        *counts.entry(record.kind().to_string()).or_default() += 1;
    }
    counts
}

fn record_timestamp_seconds(record: &DatasetRecord) -> Option<f64> {
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

fn record_scene_index(record: &DatasetRecord) -> Option<u64> {
    match record {
        DatasetRecord::Scene(record) => Some(record.scene_index),
        DatasetRecord::Observation(record) => record.scene_index,
        DatasetRecord::Feature(record) => record.scene_index,
        _ => None,
    }
}

fn record_label(record: &DatasetRecord) -> Option<&String> {
    match record {
        DatasetRecord::Observation(record) => record.label.as_ref(),
        DatasetRecord::Event(record) => Some(&record.label),
        DatasetRecord::Track(record) => record.label.as_ref(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use video_analysis_dataset::{
        AnalysisEventRecord, DatasetMetadata, FeatureRecord, FeatureValue, FramePositionRecord,
        ObservationKindRecord, ObservationRecord, SceneRecord, TimestampRecord,
    };

    #[test]
    fn window_rejects_invalid_seconds() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.transform.window"),
            input: serde_json::json!({"dataset": AnalysisDataset::empty(), "seconds": 0.0}),
        })
        .expect_err("invalid seconds");
        assert!(error.contains("window seconds"));
    }

    #[test]
    fn filter_by_kind_and_time_range() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.transform.filter"),
            input: serde_json::json!({
                "dataset": sample_dataset(),
                "kinds": ["event"],
                "startSeconds": 0.5,
                "endSeconds": 1.5
            }),
        })
        .expect("filter");

        assert_eq!(response.value["outputRecordCount"], 1);
        assert_eq!(
            response.value["dataset"]["records"][0]["event"]["label"],
            "speech"
        );
    }

    #[test]
    fn group_scenes_assigns_records_by_scene_and_frame_bounds() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.transform.groupScenes"),
            input: serde_json::json!({"dataset": sample_dataset()}),
        })
        .expect("group scenes");

        assert_eq!(response.value["groups"][0]["sceneIndex"], 0);
        assert_eq!(
            response.value["groups"][0]["recordCounts"]["observation"],
            1
        );
    }

    #[test]
    fn resampling_rejects_unsupported_aggregation() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.transform.resampleFeatures"),
            input: serde_json::json!({
                "dataset": sample_dataset(),
                "featureName": "metric",
                "seconds": 1.0,
                "aggregation": "median"
            }),
        })
        .expect_err("unsupported aggregation");
        assert!(error.contains("unsupported aggregation"));
    }

    fn sample_dataset() -> AnalysisDataset {
        let mut dataset = AnalysisDataset::new(DatasetMetadata::default());
        dataset.push(DatasetRecord::Scene(SceneRecord {
            scene_index: 0,
            start: frame(0, 0.0),
            end: frame(30, 1.0),
            duration_seconds: 1.0,
        }));
        dataset.push(DatasetRecord::Observation(ObservationRecord {
            timestamp: Some(timestamp(0.4)),
            frame: Some(frame(12, 0.4)),
            scene_index: Some(0),
            analyzer: "test".to_string(),
            kind: ObservationKindRecord::Object,
            label: Some("car".to_string()),
            text: None,
            score: Some(0.9),
            region: None,
            track_id: None,
            attributes: Default::default(),
        }));
        dataset.push(DatasetRecord::Event(AnalysisEventRecord {
            timestamp: Some(timestamp(1.0)),
            analyzer: "audio".to_string(),
            label: "speech".to_string(),
            score: Some(0.8),
        }));
        dataset.push(DatasetRecord::Feature(
            FeatureRecord::new("metric", FeatureValue::Number(2.0)).timestamp_raw(timestamp(0.2)),
        ));
        dataset
    }

    trait FeatureRecordTestExt {
        fn timestamp_raw(self, timestamp: TimestampRecord) -> Self;
    }

    impl FeatureRecordTestExt for FeatureRecord {
        fn timestamp_raw(mut self, timestamp: TimestampRecord) -> Self {
            self.timestamp = Some(timestamp);
            self
        }
    }

    fn frame(frame_index: u64, seconds: f64) -> FramePositionRecord {
        FramePositionRecord {
            frame_index,
            timestamp: timestamp(seconds),
        }
    }

    fn timestamp(seconds: f64) -> TimestampRecord {
        TimestampRecord {
            pts: (seconds * 30.0) as i64,
            timebase_num: 1,
            timebase_den: 30,
            seconds,
        }
    }
}
