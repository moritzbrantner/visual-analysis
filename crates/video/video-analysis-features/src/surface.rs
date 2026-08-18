//! Library-owned runtime surface for `video-analysis-features`.

use serde::de::DeserializeOwned;
use std::collections::BTreeMap;

use runtime_core::{
    OperationId, PackageSurface, RuntimeCapabilities, SurfaceOperation, SurfaceRequest,
    SurfaceResponse,
};
use serde::Deserialize;
use video_analysis_dataset::{AnalysisDataset, DatasetRecord, FeatureRecord, FeatureValue};

use crate::{
    AudioEventStatsExtractor, FeaturePipeline, FeatureVectorMeanExtractor,
    ObservationLabelHistogramExtractor, SceneStatsExtractor, TrackSummaryExtractor,
    TranscriptStatsExtractor,
};

const DEFAULT_EXTRACTORS: &[&str] = &[
    "scene_stats",
    "observation_label_histogram",
    "transcript_stats",
    "audio_event_stats",
    "track_summary",
    "feature_vector_mean",
];

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
                "Feature extraction over retained video-analysis datasets.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "video.features.extract",
                "Extract features",
                "Runs selected deterministic feature extractors over a retained dataset.",
                serde_json::json!({
                    "dataset": {"metadata": {"schema_version": 2, "name": null, "source": null, "created_at": null, "attributes": {}}, "records": []},
                    "extractors": DEFAULT_EXTRACTORS
                }),
            ),
            operation(
                "video.features.timelineSummary",
                "Summarize feature timeline",
                "Summarizes retained dataset record counts and timestamp coverage before feature extraction.",
                serde_json::json!({
                    "dataset": {"metadata": {"schema_version": 2, "name": null, "source": null, "created_at": null, "attributes": {}}, "records": []}
                }),
            ),
            operation(
                "video.features.aggregate",
                "Aggregate extracted features",
                "Runs selected deterministic feature extractors and summarizes output feature names, scopes, and value kinds.",
                serde_json::json!({
                    "dataset": {"metadata": {"schema_version": 2, "name": null, "source": null, "created_at": null, "attributes": {}}, "records": []},
                    "extractors": DEFAULT_EXTRACTORS
                }),
            ),
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
        "video.features.extract" => extract_value(parse_input(request.input)?)?,
        "video.features.timelineSummary" => timeline_summary_value(parse_input(request.input)?),
        "video.features.aggregate" => aggregate_value(parse_input(request.input)?)?,
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
struct ExtractRequest {
    dataset: AnalysisDataset,
    #[serde(default)]
    extractors: Vec<String>,
}

fn extract_value(request: ExtractRequest) -> Result<serde_json::Value, String> {
    let (features, extractor_names) = extract_features(request.dataset, request.extractors)?;
    Ok(serde_json::json!({
        "features": features,
        "featureCount": features.len(),
        "extractors": extractor_names
    }))
}

fn extract_features(
    dataset: AnalysisDataset,
    extractors: Vec<String>,
) -> Result<(Vec<FeatureRecord>, Vec<String>), String> {
    let extractor_names = if extractors.is_empty() {
        DEFAULT_EXTRACTORS
            .iter()
            .map(|value| (*value).to_string())
            .collect::<Vec<_>>()
    } else {
        extractors
    };

    let mut builder = FeaturePipeline::builder();
    for name in &extractor_names {
        builder = match name.as_str() {
            "scene_stats" => builder.extractor(SceneStatsExtractor),
            "observation_label_histogram" => {
                builder.extractor(ObservationLabelHistogramExtractor::default())
            }
            "transcript_stats" => builder.extractor(TranscriptStatsExtractor),
            "audio_event_stats" => builder.extractor(AudioEventStatsExtractor),
            "track_summary" => builder.extractor(TrackSummaryExtractor),
            "feature_vector_mean" => builder.extractor(FeatureVectorMeanExtractor),
            other => return Err(format!("unknown feature extractor `{other}`")),
        };
    }

    let features = builder
        .build()
        .extract(&dataset)
        .map_err(|error| error.to_string())?;
    Ok((features, extractor_names))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DatasetInput {
    dataset: AnalysisDataset,
}

fn timeline_summary_value(request: DatasetInput) -> serde_json::Value {
    let mut record_counts = BTreeMap::<String, u64>::new();
    let mut first_timestamp = None::<f64>;
    let mut last_timestamp = None::<f64>;
    for record in &request.dataset.records {
        *record_counts.entry(record.kind().to_string()).or_default() += 1;
        if let Some(timestamp) = record_timestamp_seconds(record) {
            first_timestamp = Some(first_timestamp.map_or(timestamp, |value| value.min(timestamp)));
            last_timestamp = Some(last_timestamp.map_or(timestamp, |value| value.max(timestamp)));
        }
    }

    serde_json::json!({
        "recordCount": request.dataset.records.len(),
        "recordCounts": record_counts,
        "firstTimestampSeconds": first_timestamp,
        "lastTimestampSeconds": last_timestamp,
        "durationSeconds": match (first_timestamp, last_timestamp) {
            (Some(first), Some(last)) => Some((last - first).max(0.0)),
            _ => None,
        }
    })
}

fn aggregate_value(request: ExtractRequest) -> Result<serde_json::Value, String> {
    let (features, extractor_names) = extract_features(request.dataset, request.extractors)?;
    let mut value_kinds = BTreeMap::<String, u64>::new();
    let mut scopes = BTreeMap::<String, u64>::new();
    let mut names = BTreeMap::<String, u64>::new();

    for feature in &features {
        *value_kinds
            .entry(feature_value_kind(&feature.value).to_string())
            .or_default() += 1;
        *scopes
            .entry(
                feature
                    .scope
                    .clone()
                    .unwrap_or_else(|| "unspecified".to_string()),
            )
            .or_default() += 1;
        *names.entry(feature.name.clone()).or_default() += 1;
    }

    Ok(serde_json::json!({
        "featureCount": features.len(),
        "extractors": extractor_names,
        "featureNames": names,
        "scopes": scopes,
        "valueKinds": value_kinds
    }))
}

fn feature_value_kind(value: &FeatureValue) -> &'static str {
    match value {
        FeatureValue::Number(_) => "number",
        FeatureValue::Integer(_) => "integer",
        FeatureValue::Boolean(_) => "boolean",
        FeatureValue::Text(_) => "text",
        FeatureValue::Vector(_) => "vector",
        FeatureValue::Histogram(_) => "histogram",
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use video_analysis_dataset::AnalysisDataset;

    #[test]
    fn default_extractors_return_stable_names() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.features.extract"),
            input: serde_json::json!({"dataset": AnalysisDataset::empty()}),
        })
        .expect("features");

        assert_eq!(
            response.value["extractors"],
            serde_json::json!(DEFAULT_EXTRACTORS)
        );
        assert!(response.value["featureCount"].as_u64().unwrap() > 0);
    }

    #[test]
    fn unknown_extractor_returns_typed_error() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.features.extract"),
            input: serde_json::json!({
                "dataset": AnalysisDataset::empty(),
                "extractors": ["missing"]
            }),
        })
        .expect_err("unknown extractor");
        assert!(error.contains("unknown feature extractor"));
    }

    #[test]
    fn package_surface_lists_feature_operation() {
        let ids = package_surface()
            .operations
            .into_iter()
            .map(|operation| operation.id.0)
            .collect::<Vec<_>>();
        assert!(ids.contains(&"video.features.extract".to_string()));
    }
}
