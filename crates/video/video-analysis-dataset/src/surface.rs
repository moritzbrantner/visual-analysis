//! Library-owned runtime surface for `video-analysis-dataset`.

use std::collections::{BTreeMap, BTreeSet};

use runtime_core::{
    OperationId, PackageSurface, RuntimeCapabilities, SurfaceOperation, SurfaceRequest,
    SurfaceResponse,
};
use serde::de::DeserializeOwned;
use serde::Deserialize;

use crate::{AnalysisDataset, DatasetRecord};

const DEFAULT_RECORD_LIMIT: usize = 100;
const MAX_RECORD_LIMIT: usize = 1000;

/// Returns the package surface exposed by every transport wrapper.
pub fn package_surface() -> PackageSurface {
    PackageSurface {
        library: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities: RuntimeCapabilities::pure_rust(),
        operations: vec![
            operation(
                "describe",
                "Describe package",
                "Serializable retained analysis records for video-analysis.",
                serde_json::json!({"type": "object", "additionalProperties": true}),
                serde_json::json!({"type": "object", "required": ["library", "version", "operationCount"]}),
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "video.dataset.summary",
                "Dataset summary",
                "Summarizes retained dataset metadata, record counts, time range, and frame range.",
                serde_json::json!({
                    "type": "object",
                    "required": ["dataset"],
                    "properties": { "dataset": { "type": "object" } }
                }),
                serde_json::json!({
                    "type": "object",
                    "required": ["schemaVersion", "recordCount", "recordCounts", "attributeCount"]
                }),
                serde_json::json!({"dataset": {"metadata": {"schema_version": 2, "name": null, "source": null, "created_at": null, "attributes": {}}, "records": []}}),
            ),
            operation(
                "video.dataset.recordsByKind",
                "Records by kind",
                "Returns a capped record preview filtered by dataset record kind.",
                serde_json::json!({
                    "type": "object",
                    "required": ["dataset"],
                    "properties": {
                        "dataset": { "type": "object" },
                        "kinds": { "type": "array", "items": { "type": "string" } },
                        "limit": { "type": "integer", "minimum": 0, "maximum": 1000 }
                    }
                }),
                serde_json::json!({
                    "type": "object",
                    "required": ["records", "returned", "totalMatching", "truncated"]
                }),
                serde_json::json!({
                    "dataset": {"metadata": {"schema_version": 2, "name": null, "source": null, "created_at": null, "attributes": {}}, "records": []},
                    "kinds": ["scene", "observation"],
                    "limit": 100
                }),
            ),
        ],
    }
}

fn operation(
    id: &str,
    name: &str,
    description: &str,
    input_schema: serde_json::Value,
    output_schema: serde_json::Value,
    example_request: serde_json::Value,
) -> SurfaceOperation {
    SurfaceOperation {
        id: OperationId::new(id),
        name: name.to_string(),
        description: Some(description.to_string()),
        curation: runtime_core::SurfaceOperationCuration::from_operation_id(id),
        input_schema,
        output_schema,
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
        "video.dataset.summary" => {
            let input: DatasetInput = parse_input(request.input)?;
            dataset_summary_value(&input.dataset)
        }
        "video.dataset.recordsByKind" => {
            let input: RecordsByKindRequest = parse_input(request.input)?;
            records_by_kind_value(input)?
        }
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
struct DatasetInput {
    dataset: AnalysisDataset,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RecordsByKindRequest {
    dataset: AnalysisDataset,
    #[serde(default)]
    kinds: Vec<String>,
    limit: Option<usize>,
}

fn dataset_summary_value(dataset: &AnalysisDataset) -> serde_json::Value {
    let mut record_counts = BTreeMap::<String, u64>::new();
    let mut min_seconds: Option<f64> = None;
    let mut max_seconds: Option<f64> = None;
    let mut min_frame: Option<u64> = None;
    let mut max_frame: Option<u64> = None;

    for record in &dataset.records {
        *record_counts.entry(record.kind().to_string()).or_default() += 1;
        for seconds in [
            record_timestamp_seconds(record),
            record_end_timestamp_seconds(record),
        ]
        .into_iter()
        .flatten()
        .filter(|value| value.is_finite())
        {
            min_seconds = Some(min_seconds.map_or(seconds, |current| current.min(seconds)));
            max_seconds = Some(max_seconds.map_or(seconds, |current| current.max(seconds)));
        }
        for frame in [record_frame_index(record), record_end_frame_index(record)]
            .into_iter()
            .flatten()
        {
            min_frame = Some(min_frame.map_or(frame, |current| current.min(frame)));
            max_frame = Some(max_frame.map_or(frame, |current| current.max(frame)));
        }
    }

    serde_json::json!({
        "schemaVersion": dataset.metadata.schema_version,
        "name": dataset.metadata.name,
        "recordCount": dataset.records.len(),
        "recordCounts": record_counts,
        "timeRangeSeconds": match (min_seconds, max_seconds) {
            (Some(start), Some(end)) => Some(serde_json::json!({"start": start, "end": end})),
            _ => None,
        },
        "frameRange": match (min_frame, max_frame) {
            (Some(start), Some(end)) => Some(serde_json::json!({"start": start, "end": end})),
            _ => None,
        },
        "attributeCount": dataset.metadata.attributes.len()
    })
}

fn records_by_kind_value(input: RecordsByKindRequest) -> Result<serde_json::Value, String> {
    let limit = input.limit.unwrap_or(DEFAULT_RECORD_LIMIT);
    if limit > MAX_RECORD_LIMIT {
        return Err(format!("limit must be <= {MAX_RECORD_LIMIT}"));
    }
    let kinds = input.kinds.into_iter().collect::<BTreeSet<_>>();
    let matches = input
        .dataset
        .records
        .iter()
        .filter(|record| kinds.is_empty() || kinds.contains(record.kind()))
        .collect::<Vec<_>>();
    let records = matches
        .iter()
        .take(limit)
        .map(|record| serde_json::to_value(record).map_err(|error| error.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(serde_json::json!({
        "records": records,
        "returned": records.len(),
        "totalMatching": matches.len(),
        "truncated": matches.len() > records.len()
    }))
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

fn record_end_timestamp_seconds(record: &DatasetRecord) -> Option<f64> {
    match record {
        DatasetRecord::Scene(record) => Some(record.end.timestamp.seconds),
        DatasetRecord::Track(record) => record.last_timestamp.map(|timestamp| timestamp.seconds),
        _ => None,
    }
}

fn record_frame_index(record: &DatasetRecord) -> Option<u64> {
    match record {
        DatasetRecord::VideoFrame(record) => Some(record.position.frame_index),
        DatasetRecord::Scene(record) => Some(record.start.frame_index),
        DatasetRecord::Cut(record) => Some(record.position.frame_index),
        DatasetRecord::Observation(record) => record.frame.map(|frame| frame.frame_index),
        DatasetRecord::Metric(record) => Some(record.frame_index),
        DatasetRecord::Feature(record) => record.frame_index,
        DatasetRecord::Track(record) => record.first_frame,
        DatasetRecord::Pose2d(record) => record.frame.map(|frame| frame.frame_index),
        DatasetRecord::Pose3d(record) => record.frame.map(|frame| frame.frame_index),
        DatasetRecord::AudioFrame(_) | DatasetRecord::TextSegment(_) | DatasetRecord::Event(_) => {
            None
        }
    }
}

fn record_end_frame_index(record: &DatasetRecord) -> Option<u64> {
    match record {
        DatasetRecord::Scene(record) => Some(record.end.frame_index),
        DatasetRecord::Track(record) => record.last_frame,
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AnalysisDataset, DatasetMetadata, DatasetRecord, FramePositionRecord, SceneRecord,
        TimestampRecord,
    };

    #[test]
    fn package_surface_lists_dataset_operations() {
        let ids = package_surface()
            .operations
            .into_iter()
            .map(|operation| operation.id.0)
            .collect::<Vec<_>>();
        assert!(ids.contains(&"video.dataset.summary".to_string()));
        assert!(ids.contains(&"video.dataset.recordsByKind".to_string()));
    }

    #[test]
    fn empty_dataset_summary_reports_zero_counts() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.dataset.summary"),
            input: serde_json::json!({"dataset": AnalysisDataset::empty()}),
        })
        .expect("summary");

        assert_eq!(response.value["recordCount"], 0);
        assert_eq!(response.value["recordCounts"], serde_json::json!({}));
        assert!(response.value["timeRangeSeconds"].is_null());
    }

    #[test]
    fn mixed_dataset_summary_counts_kinds() {
        let dataset = sample_dataset(2);
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.dataset.summary"),
            input: serde_json::json!({"dataset": dataset}),
        })
        .expect("summary");

        assert_eq!(response.value["recordCounts"]["scene"], 2);
        assert_eq!(response.value["frameRange"]["start"], 0);
        assert_eq!(response.value["frameRange"]["end"], 15);
    }

    #[test]
    fn records_by_kind_respects_limit_and_truncation() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.dataset.recordsByKind"),
            input: serde_json::json!({
                "dataset": sample_dataset(3),
                "kinds": ["scene"],
                "limit": 2
            }),
        })
        .expect("records by kind");

        assert_eq!(response.value["returned"], 2);
        assert_eq!(response.value["totalMatching"], 3);
        assert_eq!(response.value["truncated"], true);
    }

    fn sample_dataset(scene_count: u64) -> AnalysisDataset {
        let mut dataset = AnalysisDataset::new(DatasetMetadata {
            schema_version: 1,
            name: Some("sample".to_string()),
            source: None,
            created_at: None,
            attributes: Default::default(),
        });
        for index in 0..scene_count {
            dataset.push(DatasetRecord::Scene(SceneRecord {
                scene_index: index,
                start: frame(index * 10),
                end: frame(index * 10 + 5),
                duration_seconds: 5.0,
            }));
        }
        dataset
    }

    fn frame(frame_index: u64) -> FramePositionRecord {
        FramePositionRecord {
            frame_index,
            timestamp: TimestampRecord {
                pts: frame_index as i64,
                timebase_num: 1,
                timebase_den: 30,
                seconds: frame_index as f64 / 30.0,
            },
        }
    }
}
