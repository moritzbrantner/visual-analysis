//! Runtime metadata for the registry-backed detector compatibility package.

use runtime_core::{
    OperationId, PackageSurface, RuntimeCapabilities, SurfaceOperation, SurfaceRequest,
    SurfaceResponse,
};

/// Returns the package surface retained by the visual compatibility wrappers.
pub fn package_surface() -> PackageSurface {
    let operations = [
        ("describe", "Describe package"),
        ("video.detectors.registry", "Summarize canonical detector registry"),
        (
            "video.detectors.flashFilter",
            "Describe canonical minimum-length policy",
        ),
        (
            "video.detectors.compositePlan",
            "Describe the visual composition adapter",
        ),
    ]
    .into_iter()
    .map(|(id, name)| SurfaceOperation {
        id: OperationId::new(id),
        name: name.to_string(),
        description: Some("Compatibility metadata backed by scenedetect-core =0.1.0.".to_string()),
        curation: runtime_core::SurfaceOperationCuration::from_operation_id(id),
        input_schema: serde_json::json!({"type": "object", "additionalProperties": true}),
        output_schema: serde_json::json!({"type": "object"}),
        example_request: serde_json::json!({}),
        wasm_supported: true,
        server_supported: true,
    })
    .collect();
    PackageSurface {
        library: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities: RuntimeCapabilities::pure_rust(),
        operations,
    }
}

/// Runs a structural compatibility operation without owning detector algorithms.
pub fn run_surface_operation(request: SurfaceRequest) -> Result<SurfaceResponse, String> {
    if !package_surface()
        .operations
        .iter()
        .any(|operation| operation.id == request.operation)
    {
        return Err(format!(
            "unsupported operation `{}`",
            request.operation.as_str()
        ));
    }
    Ok(SurfaceResponse {
        operation: request.operation,
        value: serde_json::json!({
            "canonicalOwner": "scenedetect-core",
            "canonicalVersion": "0.1.0",
            "input": request.input,
        }),
        diagnostics: Vec::new(),
        artifacts: Vec::new(),
    })
}
