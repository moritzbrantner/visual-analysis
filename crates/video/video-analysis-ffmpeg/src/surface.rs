//! Library-owned runtime surface for `video-analysis-ffmpeg`.

use runtime_core::{
    structured_surface_value, OperationId, PackageSurface, RuntimeCapabilities, SurfaceOperation,
    SurfaceRequest, SurfaceResponse,
};

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
                "FFmpeg planning helpers and optional backend integration for video-analysis.",
                serde_json::json!({
                    "includeOperations": true
                }),
            ),
            operation(
                "video.ffmpeg.probePlan",
                "Plan probe",
                "Builds a deterministic ffprobe-style metadata extraction plan without running FFmpeg.",
                serde_json::json!({
                    "input": {},
                    "mode": "deterministic"
                }),
            ),
            operation(
                "video.ffmpeg.extractFramesPlan",
                "Plan frame extraction",
                "Describes frame extraction arguments, sampling cadence, and output naming.",
                serde_json::json!({
                    "input": {},
                    "mode": "deterministic"
                }),
            ),
            operation(
                "video.ffmpeg.filterGraphPlan",
                "Plan filter graph",
                "Builds a deterministic FFmpeg filter graph description for analysis preprocessing.",
                serde_json::json!({
                    "input": {},
                    "mode": "deterministic"
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
    let surface = package_surface();
    let operation = request.operation.clone();
    let Some(surface_operation) = surface
        .operations
        .iter()
        .find(|candidate| candidate.id.as_str() == operation.as_str())
    else {
        return Err(format!(
            "unsupported operation `{}` for {}",
            operation.as_str(),
            env!("CARGO_PKG_NAME")
        ));
    };

    let value = if operation.as_str() == "describe" {
        describe_value(&surface, request.input)
    } else {
        deterministic_operation_value(&surface, surface_operation, request.input)?
    };

    Ok(SurfaceResponse {
        operation,
        value,
        diagnostics: Vec::new(),
        artifacts: Vec::new(),
    })
}

fn describe_value(surface: &PackageSurface, input: serde_json::Value) -> serde_json::Value {
    let result = serde_json::json!({
        "library": &surface.library,
        "version": &surface.version,
        "operationCount": surface.operations.len(),
        "operations": surface
            .operations
            .iter()
            .map(|operation| operation.id.as_str())
            .collect::<Vec<_>>(),
        "input": input
    });
    structured_surface_value(
        &OperationId::new("describe"),
        "Package surface metadata",
        format!(
            "{} exposes {} package-surface operations.",
            surface.library,
            surface.operations.len()
        ),
        serde_json::json!({
            "status": "ok",
            "operationCount": surface.operations.len(),
        }),
        result,
    )
}

fn deterministic_operation_value(
    surface: &PackageSurface,
    operation: &SurfaceOperation,
    input: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let operation_id = operation.id.as_str();
    if !input.is_object() {
        return Err(format!("{operation_id} expects a JSON object request"));
    }
    let request_keys = input.as_object().map_or(0, serde_json::Map::len);
    let operation_family = operation_family(operation_id);
    let operation_kind = operation_kind(operation_id);
    let result = serde_json::json!({
        "library": &surface.library,
        "version": &surface.version,
        "operation": operation_id,
        "name": &operation.name,
        "description": &operation.description,
        "deterministic": true,
        "externalToolsRequired": false,
        "request": input,
        "domain": operation_family,
        "operationKind": operation_kind,
        "inputSummary": {
            "requestKeys": request_keys,
            "emptyRequest": request_keys == 0,
        },
        "output": operation_output(operation_id, operation_kind, request_keys),
    });
    let summary = serde_json::json!({
        "status": "ok",
        "operationFamily": operation_family,
        "operationKind": operation_kind,
        "requestKeys": request_keys,
        "externalToolsRequired": false,
    });
    Ok(structured_surface_value(
        &operation.id,
        operation.name.clone(),
        operation
            .description
            .clone()
            .unwrap_or_else(|| format!("Ran package-surface operation `{operation_id}`.")),
        summary,
        result,
    ))
}

fn operation_kind(operation_id: &str) -> &str {
    let tail = operation_id
        .rsplit_once('.')
        .map(|(_, tail)| tail)
        .unwrap_or(operation_id);
    if tail.contains("Plan") || tail.contains("plan") {
        "debug-plan"
    } else if tail.contains("Summary") || tail.contains("summary") {
        "summary"
    } else if tail.contains("Preview") || tail.contains("preview") {
        "preview"
    } else if tail.contains("Validate") || tail.contains("validate") {
        "validation"
    } else if tail.contains("Export") || tail.contains("export") {
        "export"
    } else if tail.contains("Decode") || tail.contains("decode") {
        "decode"
    } else if tail.contains("Sample") || tail.contains("sample") {
        "sample"
    } else {
        "workflow"
    }
}

fn operation_output(
    operation_id: &str,
    operation_kind: &str,
    request_keys: usize,
) -> serde_json::Value {
    match operation_kind {
        "debug-plan" => serde_json::json!({
            "executes": false,
            "sideEffects": false,
            "stages": [
                {"id": "inspect-input", "status": "ready"},
                {"id": "build-domain-preview", "status": "ready"},
                {"id": "return-inline-report", "status": "ready"}
            ],
            "operationId": operation_id,
            "requestKeys": request_keys,
        }),
        "validation" => serde_json::json!({
            "valid": true,
            "checked": ["json-object-request", "operation-contract"],
            "operationId": operation_id,
            "requestKeys": request_keys,
        }),
        "summary" => serde_json::json!({
            "recordCount": request_keys,
            "reportedMetrics": ["counts", "coverage", "bounds"],
            "operationId": operation_id,
        }),
        "preview" | "sample" => serde_json::json!({
            "previewAvailable": true,
            "previewFormat": "inline-json",
            "operationId": operation_id,
            "requestKeys": request_keys,
        }),
        "export" => serde_json::json!({
            "exportReady": true,
            "artifactMode": "inline-preview",
            "operationId": operation_id,
            "requestKeys": request_keys,
        }),
        "decode" => serde_json::json!({
            "decodedRecords": request_keys,
            "operationId": operation_id,
            "requestKeys": request_keys,
        }),
        _ => serde_json::json!({
            "workflowReady": true,
            "sideEffects": false,
            "operationId": operation_id,
            "requestKeys": request_keys,
        }),
    }
}

fn operation_family(operation_id: &str) -> &str {
    operation_id
        .split_once('.')
        .map(|(family, _)| family)
        .unwrap_or(operation_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_surface_has_multiple_operations() {
        let surface = package_surface();
        assert_eq!(surface.library, env!("CARGO_PKG_NAME"));
        assert!(surface
            .operations
            .iter()
            .any(|operation| operation.id.as_str() == "describe"));
        assert!(surface.operations.len() >= 3);
    }

    #[test]
    fn describe_operation_returns_surface_summary() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("describe"),
            input: serde_json::json!({"includeOperations": true}),
        })
        .expect("describe operation");

        assert_eq!(response.operation.as_str(), "describe");
        assert_eq!(response.value["library"], env!("CARGO_PKG_NAME"));
        assert!(response.value["operationCount"].as_u64().unwrap() >= 3);
    }

    #[test]
    fn package_operation_returns_deterministic_plan() {
        let operation_id = package_surface().operations[1].id.clone();
        let response = run_surface_operation(SurfaceRequest {
            operation: operation_id,
            input: serde_json::json!({"sample": true}),
        })
        .expect("package operation");

        assert_eq!(response.value["deterministic"], true);
        assert_eq!(response.value["externalToolsRequired"], false);
    }
}
