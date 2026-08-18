//! Library-owned runtime surface for `image-analysis-processing`.

use runtime_core::{
    describe_surface_response, structured_operation_response, surface_operation, PackageSurface,
    RuntimeCapabilities, SurfaceRequest, SurfaceResponse,
};

use crate::operations::{
    apply_value, composite_value, hash_value, pipeline_value, sample_image_json,
};

/// Returns the package surface exposed by every transport wrapper.
pub fn package_surface() -> PackageSurface {
    PackageSurface {
        library: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities: RuntimeCapabilities::pure_rust(),
        operations: vec![
            surface_operation(
                "describe",
                "Describe package",
                "CPU image processing primitives for video-analysis.",
                serde_json::json!({"includeOperations": true}),
            ),
            surface_operation(
                "image.processing.apply",
                "Apply operation",
                "Applies one deterministic CPU image operation and returns a capped output preview.",
                serde_json::json!({"image": sample_image_json(), "operation": {"type": "grayscale"}}),
            ),
            surface_operation(
                "image.processing.pipeline",
                "Apply pipeline",
                "Applies an ordered deterministic CPU image operation pipeline and returns a capped output preview.",
                serde_json::json!({"image": sample_image_json(), "operations": [{"type": "flipHorizontal"}, {"type": "blueNoise", "amount": 8, "seed": 42}, {"type": "brightnessContrast", "brightness": 12, "contrast": 1.1}]}),
            ),
            surface_operation(
                "image.processing.composite",
                "Composite images",
                "Composites an overlay image onto a base image with opacity, blend mode, and an optional gray mask.",
                serde_json::json!({"base": sample_image_json(), "overlay": sample_image_json(), "x": 0, "y": 0, "opacity": 0.5, "blendMode": "normal"}),
            ),
            surface_operation(
                "image.processing.hash",
                "Perceptual hash",
                "Computes a deterministic luma perceptual hash for an in-memory image.",
                serde_json::json!({"image": sample_image_json(), "hashSize": 2}),
            ),
        ],
    }
}

/// Runs one library-owned operation.
pub fn run_surface_operation(request: SurfaceRequest) -> Result<SurfaceResponse, String> {
    let surface = package_surface();
    let operation = request.operation.clone();
    let value = match request.operation.as_str() {
        "describe" => return Ok(describe_surface_response(&surface, request)),
        "image.processing.apply" => apply_value(request.input)?,
        "image.processing.pipeline" => pipeline_value(request.input)?,
        "image.processing.composite" => composite_value(request.input)?,
        "image.processing.hash" => hash_value(request.input)?,
        operation => {
            return Err(format!(
                "unsupported operation `{operation}` for {}",
                env!("CARGO_PKG_NAME")
            ))
        }
    };
    Ok(structured_operation_response(&surface, operation, value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_core::OperationId;

    #[test]
    fn grayscale_and_invert_return_expected_preview() {
        let grayscale = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.processing.apply"),
            input: serde_json::json!({"image": sample_image_json(), "operation": {"type": "grayscale"}, "previewLimit": 1}),
        })
        .expect("grayscale");
        assert_eq!(grayscale.value["pixelFormat"], "gray8");

        let invert = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.processing.apply"),
            input: serde_json::json!({"image": sample_image_json(), "operation": {"type": "invert"}, "previewLimit": 3}),
        })
        .expect("invert");
        assert_eq!(
            invert.value["dataPreview"],
            serde_json::json!([0, 255, 255])
        );
    }

    #[test]
    fn pipeline_and_composite_surfaces_work() {
        let pipeline = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.processing.pipeline"),
            input: serde_json::json!({
                "image": sample_image_json(),
                "operations": [
                    {"type": "flipHorizontal"},
                    {"type": "brightnessContrast", "brightness": 10, "contrast": 1.0}
                ],
                "previewLimit": 3
            }),
        })
        .expect("pipeline");
        assert_eq!(pipeline.value["pixelFormat"], "rgb24");

        let composite = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.processing.composite"),
            input: serde_json::json!({
                "base": sample_image_json(),
                "overlay": sample_image_json(),
                "x": -1,
                "y": 0,
                "opacity": 0.5,
                "blendMode": "normal",
                "previewLimit": 3
            }),
        })
        .expect("composite");
        assert_eq!(composite.value["width"], 2);
    }

    #[test]
    fn crop_rejects_out_of_bounds_region() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.processing.apply"),
            input: serde_json::json!({"image": sample_image_json(), "operation": {"type": "crop", "x": 1, "y": 1, "width": 4, "height": 4}}),
        })
        .expect_err("bad crop");
        assert!(error.contains("crop") || error.contains("region"));
    }

    #[test]
    fn noise_operations_return_seeded_previews() {
        let blue = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.processing.apply"),
            input: serde_json::json!({
                "image": sample_image_json(),
                "operation": {"type": "blueNoise", "amount": 12, "seed": 7},
                "previewLimit": 12
            }),
        })
        .expect("blue noise");
        assert_eq!(blue.value["pixelFormat"], "rgb24");
        assert_ne!(
            blue.value["dataPreview"],
            serde_json::json!([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])
        );

        let poisson = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.processing.apply"),
            input: serde_json::json!({
                "image": sample_image_json(),
                "operation": {"type": "poissonNoise", "scale": 32.0, "seed": 9},
                "previewLimit": 12
            }),
        })
        .expect("poisson noise");
        assert_eq!(poisson.value["pixelFormat"], "rgb24");
        assert_eq!(poisson.value["dataLength"], 12);
    }

    #[test]
    fn hash_rejects_invalid_size() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.processing.hash"),
            input: serde_json::json!({"image": sample_image_json(), "hashSize": 0}),
        })
        .expect_err("bad hash");
        assert!(error.contains("hashSize"));
    }
}
