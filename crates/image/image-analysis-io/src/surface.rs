//! Library-owned runtime surface for `image-analysis-io`.

use runtime_core::{
    describe_surface_response, structured_operation_response, OperationId, PackageSurface,
    RuntimeCapabilities, SurfaceOperation, SurfaceRequest, SurfaceResponse,
};
use serde::Deserialize;

use crate::ImageFileFormat;

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
                "Deterministic image I/O format inference and read/write planning.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "image.io.supportedFormats",
                "Supported image formats",
                "Lists image file formats supported by the image I/O crate.",
                serde_json::json!({}),
            ),
            operation(
                "image.io.inferFormat",
                "Infer image format",
                "Infers an image file format from a path extension without reading the file.",
                serde_json::json!({"path": "example.png"}),
            ),
            operation(
                "image.io.plan",
                "Plan image I/O",
                "Plans a read or write operation by inferring format and support status without touching the filesystem.",
                serde_json::json!({"path": "example.webp", "operation": "read"}),
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
    let value = match request.operation.as_str() {
        "describe" => return Ok(describe_surface_response(&surface, request)),
        "image.io.supportedFormats" => supported_formats_value(),
        "image.io.inferFormat" => infer_format_value(parse_input(request.input)?)?,
        "image.io.plan" => plan_value(parse_input(request.input)?)?,
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
struct PathRequest {
    path: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PlanRequest {
    path: String,
    operation: String,
}

fn supported_formats_value() -> serde_json::Value {
    serde_json::json!({
        "formats": ["png", "jpeg", "webp"],
        "extensions": ["png", "jpg", "jpeg", "webp"]
    })
}

fn infer_format_value(request: PathRequest) -> Result<serde_json::Value, String> {
    let format = ImageFileFormat::from_path(&request.path).map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "path": request.path,
        "format": format_name(format),
        "supported": true
    }))
}

fn plan_value(request: PlanRequest) -> Result<serde_json::Value, String> {
    if !matches!(request.operation.as_str(), "read" | "write") {
        return Err(format!(
            "unsupported image I/O operation `{}`",
            request.operation
        ));
    }
    let format = ImageFileFormat::from_path(&request.path).map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "path": request.path,
        "operation": request.operation,
        "format": format_name(format),
        "supported": true,
        "willReadOrWriteFile": false
    }))
}

fn format_name(format: ImageFileFormat) -> &'static str {
    match format {
        ImageFileFormat::Png => "png",
        ImageFileFormat::Jpeg => "jpeg",
        ImageFileFormat::WebP => "webp",
    }
}

fn parse_input<T: for<'de> Deserialize<'de>>(input: serde_json::Value) -> Result<T, String> {
    serde_json::from_value(input).map_err(|error| format!("invalid request: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_surface_lists_io_operations() {
        let ids = package_surface()
            .operations
            .into_iter()
            .map(|operation| operation.id.0)
            .collect::<Vec<_>>();
        assert!(ids.contains(&"image.io.supportedFormats".to_string()));
        assert!(ids.contains(&"image.io.inferFormat".to_string()));
        assert!(ids.contains(&"image.io.plan".to_string()));
    }

    #[test]
    fn infer_format_returns_jpeg_for_jpg_path() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.io.inferFormat"),
            input: serde_json::json!({"path": "photo.jpg"}),
        })
        .expect("infer format");
        assert_eq!(response.value["format"], "jpeg");
    }

    #[test]
    fn plan_rejects_unknown_operation() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.io.plan"),
            input: serde_json::json!({"path": "photo.png", "operation": "delete"}),
        })
        .expect_err("unsupported operation");
        assert!(error.contains("unsupported image I/O operation"));
    }
}
