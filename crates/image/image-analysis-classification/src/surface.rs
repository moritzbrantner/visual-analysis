//! Library-owned runtime surface for `image-analysis-classification`.

use std::collections::BTreeMap;

use runtime_core::{
    describe_surface_response, primary_workflow_operation, structured_operation_response,
    OperationId, PackageSurface, RuntimeCapabilities, RuntimeRequirement, SurfaceError,
    SurfaceExecutionMode, SurfaceExecutionPlan, SurfaceModelExecutionPreference, SurfaceOperation,
    SurfaceRequest, SurfaceResponse, SurfaceRuntimeContext, SurfaceSideEffect,
};
use serde::Deserialize;

use crate::{image_classification_catalog, parse_task, schema_summary, ImageClassification};

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
                "Image classification catalogs, schemas, and imported prediction validation.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "image.classification.classify",
                "Classify image",
                "Server-only local ONNX classification workflow. Backend construction lives in this task crate and uses runtime-onnx for session execution.",
                serde_json::json!({"image": sample_image_json(), "model": "vit-base-patch16-224-onnx", "limit": 5, "minScore": 0.0}),
            ),
            operation(
                "image.classification.models",
                "Classification models",
                "Lists deterministic image classification catalog entries without running inference.",
                serde_json::json!({"task": "classify"}),
            ),
            operation(
                "image.classification.schema",
                "Classification schema",
                "Returns image classification task and preset schema metadata.",
                serde_json::json!({}),
            ),
            operation(
                "image.classification.imported",
                "Imported classifications",
                "Validates caller-supplied labels and scores into normalized classification values.",
                serde_json::json!({"labels": [{"label": "landscape", "score": 0.92, "attributes": {"source": "fixture"}}]}),
            ),
            operation(
                "image.classification.topLabels",
                "Top labels",
                "Ranks normalized imported classification labels by score.",
                serde_json::json!({"labels": [{"label": "cat", "score": 0.91}, {"label": "dog", "score": 0.42}], "limit": 2}),
            ),
            operation(
                "image.classification.thresholdLabels",
                "Threshold labels",
                "Splits normalized imported classification labels by a minimum score threshold.",
                serde_json::json!({"labels": [{"label": "cat", "score": 0.91}, {"label": "dog", "score": 0.42}], "minScore": 0.5}),
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
    if id == "image.classification.classify" {
        let mut operation = primary_workflow_operation(
            id,
            name,
            description,
            example_request,
            Some(model_execution_plan(id)),
            &[
                "moenarch-image-analysis-core",
                "moenarch-model-runtime",
                "moenarch-runtime-onnx",
            ],
        );
        operation.wasm_supported = false;
        return operation;
    }
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

fn model_execution_plan(operation: &str) -> SurfaceExecutionPlan {
    SurfaceExecutionPlan {
        operation: OperationId::new(operation),
        mode: SurfaceExecutionMode::InMemory,
        side_effects: vec![
            SurfaceSideEffect::ReadsFiles,
            SurfaceSideEffect::WritesFiles,
            SurfaceSideEffect::Network,
        ],
        cancellable: false,
        progress_unit: Some("files".to_string()),
        expected_artifacts: Vec::new(),
        requirements: vec![
            RuntimeRequirement {
                name: "onnxruntime".to_string(),
                description: Some("ONNX Runtime CPU execution via runtime-onnx".to_string()),
                required: true,
            },
            RuntimeRequirement {
                name: "local-filesystem".to_string(),
                description: Some("Readable and writable .model-runtime bundle root".to_string()),
                required: true,
            },
            RuntimeRequirement {
                name: "hugging-face-access".to_string(),
                description: Some("Network access for first-run model downloads".to_string()),
                required: true,
            },
        ],
        max_recommended_input_bytes: Some(16_777_216),
    }
}

/// Runs one library-owned operation.
pub fn run_surface_operation(request: SurfaceRequest) -> Result<SurfaceResponse, String> {
    run_surface_operation_with_context(
        request,
        &SurfaceRuntimeContext::compatibility_no_side_effects(),
    )
}

/// Runs one library-owned operation with explicit runtime/storage permissions.
pub fn run_surface_operation_with_context(
    request: SurfaceRequest,
    context: &SurfaceRuntimeContext,
) -> Result<SurfaceResponse, String> {
    let surface = package_surface();
    let operation = request.operation.clone();
    let value = match request.operation.as_str() {
        "describe" => return Ok(describe_surface_response(&surface, request)),
        "image.classification.classify" => classify_value(parse_input(request.input)?, context)?,
        "image.classification.models" => models_value(parse_input(request.input)?)?,
        "image.classification.schema" => schema_summary(),
        "image.classification.imported" => imported_value(parse_input(request.input)?)?,
        "image.classification.topLabels" => top_labels_value(parse_input(request.input)?)?,
        "image.classification.thresholdLabels" => {
            threshold_labels_value(parse_input(request.input)?)?
        }
        operation => {
            return Err(format!(
                "unsupported operation `{operation}` for {}",
                env!("CARGO_PKG_NAME")
            ));
        }
    };
    Ok(structured_operation_response(&surface, operation, value))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelsRequest {
    task: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ImportedRequest {
    #[serde(default, alias = "items", alias = "classifications")]
    labels: Vec<ImportedClassification>,
    limit: Option<usize>,
    min_score: Option<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ClassifyRequest {
    image: Option<serde_json::Value>,
    model: Option<String>,
    min_score: Option<f32>,
    limit: Option<usize>,
    auto_download: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ImportedClassification {
    label: String,
    score: Option<f32>,
    #[serde(default)]
    attributes: BTreeMap<String, String>,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct NormalizedClassification {
    label: String,
    score: Option<f32>,
    attributes: BTreeMap<String, String>,
}

fn models_value(request: ModelsRequest) -> Result<serde_json::Value, String> {
    let task = request
        .task
        .as_deref()
        .map(|task| {
            parse_task(task).ok_or_else(|| format!("unsupported classification task `{task}`"))
        })
        .transpose()?;
    Ok(serde_json::json!({
        "models": image_classification_catalog(task)
    }))
}

fn classify_value(
    request: ClassifyRequest,
    context: &SurfaceRuntimeContext,
) -> Result<serde_json::Value, String> {
    if request.image.is_none() {
        return Err(SurfaceError::invalid_request(
            Some("image.classification.classify"),
            "image.classification.classify requires an image payload",
        )
        .to_error_string());
    }
    if let Some(limit) = request.limit {
        if limit == 0 {
            return Err(SurfaceError::invalid_request(
                Some("image.classification.classify"),
                "classification label limit must be greater than zero",
            )
            .to_error_string());
        }
    }
    let min_score = request.min_score.unwrap_or(0.0);
    if !min_score.is_finite() {
        return Err(SurfaceError::invalid_request(
            Some("image.classification.classify"),
            "classification minScore must be finite",
        )
        .to_error_string());
    }
    if request.auto_download.unwrap_or(context.model.auto_setup)
        && context.model.preference == SurfaceModelExecutionPreference::ModelFirst
        && !context.allows_model_auto_setup()
    {
        return Err(SurfaceError::permission_denied(
            Some("image.classification.classify"),
            "model_auto_setup",
            "model auto-setup requires native runtime, reads, writes, network, and modelRoot context",
        )
        .to_error_string());
    }
    let setup_allowed = context.allows_model_auto_setup();
    let model_root = context.storage.model_root.clone();
    Ok(serde_json::json!({
        "executed": false,
        "serverOnly": true,
        "wasmSupported": false,
        "serverSupported": true,
        "labels": [],
        "topLabel": null,
        "count": 0,
        "modelId": request.model.unwrap_or_else(|| "Xenova/vit-base-patch16-224".to_string()),
        "bundlePath": model_root,
        "runtime": "onnx",
        "modelExecutionPreference": &context.model.preference,
        "autoDownload": request.auto_download.unwrap_or(context.model.auto_setup),
        "setup": {
            "attempted": setup_allowed,
            "allowed": setup_allowed,
            "mode": if setup_allowed { "context-declared" } else { "plan-only" },
            "requires": ["modelRoot", "network", "writes", "runtime-onnx"]
        },
        "contracts": {
            "image": "moenarch-image-analysis-core::ImageView",
            "bundle": "moenarch-model-runtime::HuggingFaceModelSpec",
            "runtime": "moenarch-runtime-onnx"
        },
        "diagnostics": ["ONNX classification is model-first. This package-surface call returns a typed execution plan unless a native adapter supplies a capable model runtime context."]
    }))
}

fn sample_image_json() -> serde_json::Value {
    serde_json::json!({
        "width": 2,
        "height": 2,
        "pixelFormat": "rgb24",
        "stride": null,
        "data": [255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]
    })
}

fn imported_value(request: ImportedRequest) -> Result<serde_json::Value, String> {
    let classifications =
        normalize_classifications("image.classification.imported", request.labels)?;
    Ok(serde_json::json!({
        "count": classifications.len(),
        "classifications": classifications
    }))
}

fn top_labels_value(request: ImportedRequest) -> Result<serde_json::Value, String> {
    let mut classifications =
        normalize_classifications("image.classification.topLabels", request.labels)?;
    if let Some(limit) = request.limit {
        if limit == 0 {
            return Err("classification label limit must be greater than zero".to_string());
        }
    }
    classifications.sort_by(|left, right| compare_scores_desc(left.score, right.score));
    if let Some(limit) = request.limit {
        classifications.truncate(limit);
    }
    Ok(serde_json::json!({
        "count": classifications.len(),
        "topLabel": classifications.first(),
        "classifications": classifications
    }))
}

fn threshold_labels_value(request: ImportedRequest) -> Result<serde_json::Value, String> {
    let threshold = request.min_score.unwrap_or(0.5);
    if !threshold.is_finite() {
        return Err("classification minScore must be finite".to_string());
    }
    if let Some(limit) = request.limit {
        if limit == 0 {
            return Err("classification label limit must be greater than zero".to_string());
        }
    }
    let mut classifications =
        normalize_classifications("image.classification.thresholdLabels", request.labels)?;
    classifications.sort_by(|left, right| compare_scores_desc(left.score, right.score));
    let mut accepted = classifications
        .iter()
        .filter(|classification| classification.score.is_some_and(|score| score >= threshold))
        .cloned()
        .collect::<Vec<_>>();
    if let Some(limit) = request.limit {
        accepted.truncate(limit);
    }
    let rejected = classifications
        .into_iter()
        .filter(|classification| !classification.score.is_some_and(|score| score >= threshold))
        .collect::<Vec<_>>();
    Ok(serde_json::json!({
        "threshold": threshold,
        "count": accepted.len(),
        "rejectedCount": rejected.len(),
        "topLabel": accepted.first(),
        "accepted": accepted,
        "rejected": rejected
    }))
}

fn normalize_classifications(
    operation: &str,
    labels: Vec<ImportedClassification>,
) -> Result<Vec<NormalizedClassification>, String> {
    runtime_core::require_non_empty(operation, "labels", &labels)?;
    labels
        .into_iter()
        .map(|item| {
            if item.label.trim().is_empty() {
                return Err("classification label must not be empty".to_string());
            }
            if let Some(score) = item.score {
                if !score.is_finite() {
                    return Err("classification score must be finite".to_string());
                }
            }
            let mut classification = ImageClassification::new(item.label.trim().to_string());
            if let Some(score) = item.score {
                classification = classification.score(score);
            }
            for (key, value) in item.attributes {
                classification = classification.attribute(key, value);
            }
            Ok(NormalizedClassification {
                label: classification.label,
                score: classification.score,
                attributes: classification.attributes,
            })
        })
        .collect()
}

fn compare_scores_desc(left: Option<f32>, right: Option<f32>) -> std::cmp::Ordering {
    match (left, right) {
        (Some(left), Some(right)) => right.total_cmp(&left),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    }
}

fn parse_input<T: for<'de> Deserialize<'de>>(input: serde_json::Value) -> Result<T, String> {
    serde_json::from_value(input).map_err(|error| format!("invalid request: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_surface_lists_classification_operations() {
        let ids = package_surface()
            .operations
            .into_iter()
            .map(|operation| operation.id.0)
            .collect::<Vec<_>>();
        assert!(ids.contains(&"describe".to_string()));
        assert!(ids.contains(&"image.classification.classify".to_string()));
        assert!(ids.contains(&"image.classification.models".to_string()));
        assert!(ids.contains(&"image.classification.imported".to_string()));
        assert!(ids.contains(&"image.classification.topLabels".to_string()));
        assert!(ids.contains(&"image.classification.thresholdLabels".to_string()));
    }

    #[test]
    fn classify_operation_is_server_only_and_structured() {
        let surface = package_surface();
        let operation = surface
            .operations
            .iter()
            .find(|operation| operation.id.as_str() == "image.classification.classify")
            .expect("classify operation");
        assert!(!operation.wasm_supported);
        assert!(operation.server_supported);
        assert!(!operation.input_schema["xExecutionPlan"].is_null());
        assert_eq!(operation.input_schema["additionalProperties"], false);
        assert_eq!(operation.input_schema["xOperationCategory"], "workflow");
        assert_eq!(
            operation.input_schema["xLowerContractProof"]["crates"][0],
            "moenarch-image-analysis-core"
        );

        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.classification.classify"),
            input: operation.example_request.clone(),
        })
        .expect("classify");
        assert_eq!(response.value["executed"], false);
        assert_eq!(response.value["runtime"], "onnx");
        assert_eq!(response.value["setup"]["mode"], "plan-only");
        assert_eq!(
            response.value["contracts"]["bundle"],
            "moenarch-model-runtime::HuggingFaceModelSpec"
        );
    }

    #[test]
    fn classify_denies_auto_setup_without_context_permissions() {
        let error = run_surface_operation_with_context(
            SurfaceRequest {
                operation: OperationId::new("image.classification.classify"),
                input: serde_json::json!({
                    "image": sample_image_json(),
                    "autoDownload": true
                }),
            },
            &SurfaceRuntimeContext {
                runtime: runtime_core::SurfaceRuntimeKind::NativeCli,
                side_effects: runtime_core::SurfaceSideEffectPolicy::none(),
                storage: runtime_core::SurfaceStorageContext::default(),
                model: runtime_core::SurfaceModelContext {
                    auto_setup: true,
                    preference: SurfaceModelExecutionPreference::ModelFirst,
                },
            },
        )
        .expect_err("auto setup denied");
        let parsed = runtime_core::parse_surface_error(&error).expect("typed surface error");
        assert_eq!(parsed.code, "permission_denied");
        assert_eq!(parsed.details["permission"], "model_auto_setup");
    }

    #[test]
    fn classify_accepts_capable_fake_bundle_context() {
        let response = run_surface_operation_with_context(
            SurfaceRequest {
                operation: OperationId::new("image.classification.classify"),
                input: serde_json::json!({
                    "image": sample_image_json(),
                    "model": "vit-base-patch16-224-onnx"
                }),
            },
            &SurfaceRuntimeContext {
                runtime: runtime_core::SurfaceRuntimeKind::NativeServer,
                side_effects: runtime_core::SurfaceSideEffectPolicy {
                    allow_reads: true,
                    allow_writes: true,
                    allow_network: true,
                    allow_external_process: false,
                    max_download_bytes: Some(64 * 1024 * 1024),
                },
                storage: runtime_core::SurfaceStorageContext {
                    cache_root: Some("memory://cache".to_string()),
                    artifact_root: None,
                    model_root: Some("memory://models".to_string()),
                },
                model: runtime_core::SurfaceModelContext {
                    auto_setup: true,
                    preference: SurfaceModelExecutionPreference::ModelFirst,
                },
            },
        )
        .expect("capable context");
        assert_eq!(response.value["bundlePath"], "memory://models");
        assert_eq!(response.value["setup"]["mode"], "context-declared");
        assert_eq!(response.value["setup"]["attempted"], true);
    }

    #[test]
    fn imported_operation_normalizes_labels() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.classification.imported"),
            input: serde_json::json!({"labels": [{"label": " landscape ", "score": 0.9}]}),
        })
        .expect("imported");
        assert_eq!(response.value["count"], 1);
        assert_eq!(response.value["classifications"][0]["label"], "landscape");
    }

    #[test]
    fn imported_operation_rejects_empty_label() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.classification.imported"),
            input: serde_json::json!({"labels": [{"label": ""}]}),
        })
        .expect_err("empty label");
        assert!(error.contains("label"));
    }

    #[test]
    fn top_labels_operation_returns_highest_score() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.classification.topLabels"),
            input: serde_json::json!({"labels": [
                {"label": "low", "score": 0.1},
                {"label": "high", "score": 0.9}
            ]}),
        })
        .expect("top labels");
        assert_eq!(response.value["topLabel"]["label"], "high");
    }

    #[test]
    fn ranking_operations_reject_empty_labels() {
        let top_error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.classification.topLabels"),
            input: serde_json::json!({"labels": []}),
        })
        .expect_err("empty labels");
        assert!(top_error.contains("labels"));

        let threshold_error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.classification.thresholdLabels"),
            input: serde_json::json!({"labels": []}),
        })
        .expect_err("empty labels");
        assert!(threshold_error.contains("labels"));
    }
}
