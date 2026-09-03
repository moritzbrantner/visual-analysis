//! Library-owned runtime surface for `image-analysis-embeddings`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use image_analysis_core::contracts::{sample_image_json, ImagePayload};
use image_analysis_detection::{FaceBox, FaceDetection};
use model_runtime::ModelBundleResolveOptions;
use runtime_core::{
    describe_surface_response, structured_operation_response, OperationId, PackageSurface,
    RuntimeCapabilities, RuntimeRequirement, SurfaceExecutionMode, SurfaceExecutionPlan,
    SurfaceOperation, SurfaceRequest, SurfaceResponse, SurfaceSideEffect,
};
use serde::Deserialize;
use video_analysis_core::BoundingBox;

use crate::{
    image_embedding_catalog, parse_task, schema_summary, FaceEmbedderBackend, FaceEmbedding,
    FaceEmbeddingPreset, ImageEmbedding, OnnxFaceEmbedder,
};

const DEFAULT_FACE_EMBEDDING_MODEL: &str = "opencv-sface-onnx";

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
                "Image and face embedding catalogs, schemas, imported vector validation, and native SFace execution.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "image.embeddings.models",
                "Embedding models",
                "Lists deterministic image and face embedding catalog entries without running embedding models.",
                serde_json::json!({"task": "face-embed"}),
            ),
            operation(
                "image.embeddings.faceEmbed",
                "Embed face",
                "Runs OpenCV SFace ONNX embedding for a whole image or an optional pixel-space face region. Model downloads remain opt-in through autoDownload.",
                serde_json::json!({"image": sample_image_json(), "model": DEFAULT_FACE_EMBEDDING_MODEL, "region": {"x": 0, "y": 0, "width": 2, "height": 2}, "autoDownload": false}),
            ),
            operation(
                "image.embeddings.schema",
                "Embedding schema",
                "Returns image and face embedding task and preset schema metadata.",
                serde_json::json!({}),
            ),
            operation(
                "image.embeddings.validate",
                "Validate embeddings",
                "Validates imported image or face embedding vectors and optional face regions.",
                serde_json::json!({"kind": "face", "vector": [0.1, 0.2, 0.3], "region": {"x": 1, "y": 2, "width": 3, "height": 4}}),
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
    let mut operation = SurfaceOperation {
        id: OperationId::new(id),
        name: name.to_string(),
        description: Some(description.to_string()),
        curation: runtime_core::SurfaceOperationCuration::from_operation_id(id),
        input_schema: serde_json::json!({"type": "object", "additionalProperties": true, "xOperationCategory": runtime_core::operation_category(id)}),
        output_schema: serde_json::json!({"type": "object", "xOperationCategory": runtime_core::operation_category(id)}),
        example_request,
        wasm_supported: true,
        server_supported: true,
    };
    if id == "image.embeddings.faceEmbed" {
        let plan = model_execution_plan(id);
        operation.input_schema["xExecutionPlan"] =
            runtime_core::surface_execution_plan_value(&plan);
        operation.output_schema["xExecutionPlan"] =
            runtime_core::surface_execution_plan_value(&plan);
        operation.wasm_supported = false;
    }
    operation
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
        progress_unit: Some("faces".to_string()),
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
                description: Some(
                    "Network access only when autoDownload is explicitly true".to_string(),
                ),
                required: false,
            },
        ],
        max_recommended_input_bytes: Some(16_777_216),
    }
}

/// Runs one library-owned operation.
pub fn run_surface_operation(request: SurfaceRequest) -> Result<SurfaceResponse, String> {
    let surface = package_surface();
    let operation = request.operation.clone();
    let value = match request.operation.as_str() {
        "describe" => return Ok(describe_surface_response(&surface, request)),
        "image.embeddings.models" => models_value(parse_input(request.input)?)?,
        "image.embeddings.faceEmbed" => face_embed_value(parse_input(request.input)?)?,
        "image.embeddings.schema" => schema_summary(),
        "image.embeddings.validate" => validate_value(parse_input(request.input)?)?,
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
struct FaceEmbedRequest {
    #[serde(default)]
    image: Option<ImagePayload>,
    #[serde(default)]
    image_path: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    auto_download: Option<bool>,
    #[serde(default)]
    model_root: Option<String>,
    #[serde(default)]
    region: Option<BoxRequest>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ValidateRequest {
    #[serde(default = "default_kind")]
    kind: String,
    vector: Vec<f32>,
    #[serde(default)]
    region: Option<BoxRequest>,
    #[serde(default)]
    attributes: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BoxRequest {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

fn models_value(request: ModelsRequest) -> Result<serde_json::Value, String> {
    let task = request
        .task
        .as_deref()
        .map(|task| parse_task(task).ok_or_else(|| format!("unsupported embedding task `{task}`")))
        .transpose()?;
    let mut models = serde_json::to_value(image_embedding_catalog(task))
        .map_err(|error| format!("failed to serialize embedding model catalog: {error}"))?;
    if let Some(entries) = models.as_array_mut() {
        for entry in entries {
            if entry["id"] == DEFAULT_FACE_EMBEDDING_MODEL {
                entry["supported"] = serde_json::json!(cfg!(feature = "onnx"));
                entry["note"] = serde_json::json!(if cfg!(feature = "onnx") {
                    "Native SFace execution is available through image.embeddings.faceEmbed."
                } else {
                    "Enable the onnx feature to execute SFace natively."
                });
            }
        }
    }
    Ok(serde_json::json!({"models": models}))
}

fn face_embed_value(request: FaceEmbedRequest) -> Result<serde_json::Value, String> {
    validate_image_source(
        "image.embeddings.faceEmbed",
        request.image.as_ref(),
        request.image_path.as_deref(),
    )?;
    let model = request
        .model
        .unwrap_or_else(|| DEFAULT_FACE_EMBEDDING_MODEL.to_string());
    if model != DEFAULT_FACE_EMBEDDING_MODEL {
        return Err(format!(
            "unsupported face embedding model `{model}`; expected `{DEFAULT_FACE_EMBEDDING_MODEL}`"
        ));
    }

    let auto_download = request.auto_download.unwrap_or(false);
    let spec = FaceEmbeddingPreset::OpenCvSFace.model_spec();
    let mut bundle_options = ModelBundleResolveOptions {
        auto_download,
        download_progress: auto_download,
        ..ModelBundleResolveOptions::default()
    };
    if let Some(model_root) = request.model_root.as_deref() {
        bundle_options.bundle_root = PathBuf::from(model_root);
    }
    let bundle_path = bundle_options.bundle_root.display().to_string();
    let bundle = model_runtime::resolve_or_download_bundle(&spec, &bundle_options)
        .map_err(|error| error.to_string())?;
    let mut embedder = OnnxFaceEmbedder::from_bundle(bundle).map_err(|error| error.to_string())?;

    let embedding = if let Some(image) = request.image {
        let view = image.view().map_err(|error| error.to_string())?;
        let detection = request
            .region
            .map(|region| face_detection_for_region(region, view.width, view.height))
            .transpose()?;
        embedder
            .embed_face(&view, detection.as_ref())
            .map_err(|error| error.to_string())?
    } else {
        let image_path = request.image_path.expect("validated imagePath");
        let image = image_analysis_io::read_image(&image_path)
            .map_err(|error| format!("failed to read face embedding image `{image_path}`: {error}"))?;
        let view = image.as_view();
        let detection = request
            .region
            .map(|region| face_detection_for_region(region, view.width, view.height))
            .transpose()?;
        embedder
            .embed_face(&view, detection.as_ref())
            .map_err(|error| error.to_string())?
    };

    Ok(serde_json::json!({
        "executed": true,
        "nativeOnly": true,
        "cliSupported": true,
        "wasmSupported": false,
        "serverSupported": true,
        "kind": "face",
        "modelId": DEFAULT_FACE_EMBEDDING_MODEL,
        "bundlePath": bundle_path,
        "runtime": "onnx",
        "autoDownload": auto_download,
        "dimensions": embedding.vector.len(),
        "vector": embedding.vector,
        "region": embedding.region.map(|region| serde_json::json!({
            "x": region.x,
            "y": region.y,
            "width": region.width,
            "height": region.height
        })),
        "attributes": embedding.attributes
    }))
}

fn face_detection_for_region(
    region: BoxRequest,
    image_width: u32,
    image_height: u32,
) -> Result<FaceDetection, String> {
    let region = region.bounding_box()?;
    if image_width == 0 || image_height == 0 {
        return Err("face embedding image dimensions must be non-zero".to_string());
    }
    if region.x.saturating_add(region.width) > image_width
        || region.y.saturating_add(region.height) > image_height
    {
        return Err("face embedding region must stay inside the image bounds".to_string());
    }
    let bbox = FaceBox::new(
        region.x as f32 / image_width as f32,
        region.y as f32 / image_height as f32,
        region.width as f32 / image_width as f32,
        region.height as f32 / image_height as f32,
    )
    .map_err(|error| error.to_string())?;
    FaceDetection::new(bbox, 1.0).map_err(|error| error.to_string())
}

fn validate_image_source(
    operation: &str,
    image: Option<&ImagePayload>,
    image_path: Option<&str>,
) -> Result<(), String> {
    if image.is_some() && image_path.is_some() {
        return Err(format!(
            "{operation} accepts either image or imagePath, not both"
        ));
    }
    if image.is_none() && image_path.is_none() {
        return Err(format!("{operation} requires image or imagePath"));
    }
    Ok(())
}

fn validate_value(request: ValidateRequest) -> Result<serde_json::Value, String> {
    let finite = request.vector.iter().all(|value| value.is_finite());
    match request.kind.as_str() {
        "image" | "image_embedding" | "embed" => {
            let mut embedding =
                ImageEmbedding::new(request.vector).map_err(|error| error.to_string())?;
            for (key, value) in request.attributes {
                embedding = embedding.attribute(key, value);
            }
            Ok(serde_json::json!({
                "kind": "image",
                "dimensions": embedding.vector.len(),
                "finite": finite,
                "region": null,
                "attributes": embedding.attributes
            }))
        }
        "face" | "face_embedding" | "face-embed" => {
            let mut embedding =
                FaceEmbedding::new(request.vector).map_err(|error| error.to_string())?;
            if let Some(region) = request.region {
                embedding = embedding.region(region.bounding_box()?);
            }
            for (key, value) in request.attributes {
                embedding = embedding.attribute(key, value);
            }
            Ok(serde_json::json!({
                "kind": "face",
                "dimensions": embedding.vector.len(),
                "finite": finite,
                "region": embedding.region.map(|region| serde_json::json!({
                    "x": region.x,
                    "y": region.y,
                    "width": region.width,
                    "height": region.height
                })),
                "attributes": embedding.attributes
            }))
        }
        other => Err(format!("unsupported embedding kind `{other}`")),
    }
}

impl BoxRequest {
    fn bounding_box(self) -> Result<BoundingBox, String> {
        BoundingBox::new(self.x, self.y, self.width, self.height).map_err(|error| error.to_string())
    }
}

fn parse_input<T: for<'de> Deserialize<'de>>(input: serde_json::Value) -> Result<T, String> {
    serde_json::from_value(input).map_err(|error| format!("invalid request: {error}"))
}

fn default_kind() -> String {
    "image".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_surface_lists_embedding_operations() {
        let ids = package_surface()
            .operations
            .into_iter()
            .map(|operation| operation.id.0)
            .collect::<Vec<_>>();
        assert!(ids.contains(&"image.embeddings.models".to_string()));
        assert!(ids.contains(&"image.embeddings.faceEmbed".to_string()));
        assert!(ids.contains(&"image.embeddings.validate".to_string()));
    }

    #[test]
    fn face_embed_requires_exactly_one_image_source() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.embeddings.faceEmbed"),
            input: serde_json::json!({"model": DEFAULT_FACE_EMBEDDING_MODEL}),
        })
        .expect_err("missing image");
        assert!(error.contains("requires image or imagePath"));
    }

    #[test]
    fn face_embed_rejects_unsupported_model_before_bundle_resolution() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.embeddings.faceEmbed"),
            input: serde_json::json!({
                "image": sample_image_json(),
                "model": "other-face-model",
                "autoDownload": false
            }),
        })
        .expect_err("unsupported model");
        assert!(error.contains("unsupported face embedding model"));
    }

    #[test]
    fn face_region_converts_to_normalized_detection() {
        let detection = face_detection_for_region(
            BoxRequest {
                x: 50,
                y: 25,
                width: 100,
                height: 50,
            },
            200,
            100,
        )
        .expect("detection");
        assert_eq!(detection.bbox.x, 0.25);
        assert_eq!(detection.bbox.y, 0.25);
        assert_eq!(detection.bbox.width, 0.5);
        assert_eq!(detection.bbox.height, 0.5);
    }

    #[test]
    fn face_region_rejects_out_of_bounds_crop() {
        let error = face_detection_for_region(
            BoxRequest {
                x: 90,
                y: 0,
                width: 20,
                height: 10,
            },
            100,
            100,
        )
        .expect_err("out of bounds");
        assert!(error.contains("inside the image bounds"));
    }

    #[test]
    fn validates_face_embedding_region() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.embeddings.validate"),
            input: serde_json::json!({
                "kind": "face",
                "vector": [0.1, 0.2],
                "region": {"x": 1, "y": 2, "width": 3, "height": 4}
            }),
        })
        .expect("validate");
        assert_eq!(response.value["dimensions"], 2);
        assert_eq!(response.value["region"]["width"], 3);
    }

    #[test]
    fn rejects_empty_embedding_values() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.embeddings.validate"),
            input: serde_json::json!({"vector": []}),
        })
        .expect_err("invalid vector");
        assert!(error.contains("must not be empty"));
    }
}
