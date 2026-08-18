//! Library-owned runtime surface for `video-analysis-storage`.

use runtime_core::{
    OperationId, PackageSurface, RuntimeCapabilities, SurfaceOperation, SurfaceRequest,
    SurfaceResponse,
};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use video_analysis_dataset::AnalysisDataset;

use crate::build_manifest;

const DEFAULT_JSONL_LIMIT: usize = 20;
const MAX_JSONL_LIMIT: usize = 100;

/// Returns the package surface exposed by every transport wrapper.
pub fn package_surface() -> PackageSurface {
    PackageSurface {
        library: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities: RuntimeCapabilities::pure_rust(),
        operations: vec![
            operation("describe", "Describe package", "Dataset persistence for video-analysis.", serde_json::json!({"includeOperations": true})),
            operation("video.storage.manifestPlan", "Manifest plan", "Builds a dataset manifest preview without writing files.", serde_json::json!({"dataset": {"metadata": {"schema_version": 2, "name": null, "source": null, "created_at": null, "attributes": {}}, "records": []}, "recordsPath": "records.jsonl"})),
            operation("video.storage.jsonlPreview", "JSONL preview", "Serializes a capped preview of dataset records as JSON lines without writing files.", serde_json::json!({"dataset": {"metadata": {"schema_version": 2, "name": null, "source": null, "created_at": null, "attributes": {}}, "records": []}, "limit": 20})),
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
        "video.storage.manifestPlan" => manifest_plan_value(parse_input(request.input)?)?,
        "video.storage.jsonlPreview" => jsonl_preview_value(parse_input(request.input)?)?,
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
struct ManifestPlanRequest {
    dataset: AnalysisDataset,
    records_path: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct JsonlPreviewRequest {
    dataset: AnalysisDataset,
    limit: Option<usize>,
}

fn manifest_plan_value(request: ManifestPlanRequest) -> Result<serde_json::Value, String> {
    serde_json::to_value(build_manifest(&request.dataset, request.records_path))
        .map_err(|error| error.to_string())
}

fn jsonl_preview_value(request: JsonlPreviewRequest) -> Result<serde_json::Value, String> {
    let limit = request.limit.unwrap_or(DEFAULT_JSONL_LIMIT);
    if limit > MAX_JSONL_LIMIT {
        return Err(format!("limit must be <= {MAX_JSONL_LIMIT}"));
    }
    let total_records = request.dataset.records.len();
    let lines = request
        .dataset
        .records
        .iter()
        .take(limit)
        .map(|record| serde_json::to_string(record).map_err(|error| error.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(serde_json::json!({
        "lines": lines,
        "returned": lines.len(),
        "totalRecords": total_records,
        "truncated": total_records > lines.len()
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use video_analysis_dataset::{AnalysisDataset, DatasetRecord, MetricRecord};

    #[test]
    fn manifest_plan_includes_record_counts() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.storage.manifestPlan"),
            input: serde_json::json!({
                "dataset": sample_dataset(2),
                "recordsPath": "records.jsonl"
            }),
        })
        .expect("manifest plan");

        assert_eq!(response.value["record_count"], 2);
        assert_eq!(response.value["record_counts"]["metric"], 2);
    }

    #[test]
    fn jsonl_preview_caps_output_and_reports_truncation() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.storage.jsonlPreview"),
            input: serde_json::json!({"dataset": sample_dataset(3), "limit": 2}),
        })
        .expect("jsonl preview");

        assert_eq!(response.value["returned"], 2);
        assert_eq!(response.value["totalRecords"], 3);
        assert_eq!(response.value["truncated"], true);
    }

    #[test]
    fn malformed_dataset_returns_invalid_request() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.storage.jsonlPreview"),
            input: serde_json::json!({"dataset": {"records": "nope"}}),
        })
        .expect_err("invalid request");

        assert!(error.contains("invalid request"));
    }

    fn sample_dataset(count: u64) -> AnalysisDataset {
        let mut dataset = AnalysisDataset::empty();
        for index in 0..count {
            dataset.push(DatasetRecord::Metric(MetricRecord::new(
                index,
                "score",
                index as f64,
            )));
        }
        dataset
    }
}
