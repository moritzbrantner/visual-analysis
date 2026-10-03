//! Library-owned runtime surface for `image-analysis-detection`.

use std::path::PathBuf;

use image_analysis_core::contracts::{sample_image_json, ImagePayload};
use model_runtime::{ModelBundleResolveOptions, ModelPreset};
use runtime_core::{
    describe_surface_response, structured_operation_response, OperationId, PackageSurface,
    RuntimeCapabilities, RuntimeRequirement, SurfaceExecutionMode, SurfaceExecutionPlan,
    SurfaceOperation, SurfaceRequest, SurfaceResponse, SurfaceSideEffect,
};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use video_analysis_core::BoundingBox;

use crate::{
    ColorBlobDetectionOptions, ColorBlobDetector, FaceDetection, FaceDetectionPreset,
    FaceDetectorBackend, ImageDetection, OnnxFaceDetector, OnnxObjectDetector, TargetColor,
};

const DEFAULT_DETECTION_MODEL: &str = "xenova-detr-resnet-50-onnx";
const DEFAULT_FACE_DETECTION_MODEL: &str = "opencv-yunet-onnx";

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
                "Canonical image detection types, native DETR/YuNet execution, and mask-proposal adapters for video-analysis.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "image.detection.detect",
                "Detect objects",
                "Runs local ONNX DETR object detection against an in-memory image payload or local image path. Model downloads remain opt-in through autoDownload.",
                serde_json::json!({"image": sample_image_json(), "model": DEFAULT_DETECTION_MODEL, "autoDownload": false, "minScore": 0.5, "limit": 20}),
            ),
            operation(
                "image.detection.detectFaces",
                "Detect faces",
                "Runs local OpenCV YuNet ONNX face detection against an in-memory image payload or local image path. Model downloads remain opt-in through autoDownload.",
                serde_json::json!({"image": sample_image_json(), "model": DEFAULT_FACE_DETECTION_MODEL, "autoDownload": false, "limit": 20}),
            ),
            operation(
                "image.detection.colorBlob",
                "Color blob detection",
                "Detects connected red-dominant color blobs in an in-memory image.",
                serde_json::json!({"image": sample_image_json(), "targetColor": "red", "minAreaPixels": 1, "mergeAdjacent": true}),
            ),
            operation(
                "image.detection.models",
                "Detection models",
                "Returns detection model catalog and deterministic fallbacks without running detectors.",
                serde_json::json!({}),
            ),
            operation(
                "image.detection.boxSummary",
                "Detection box summary",
                "Validates imported bounding boxes and detections, then returns aggregate score and union bounds.",
                serde_json::json!({"detections": [{"label": "face", "score": 0.9, "region": {"x": 1, "y": 2, "width": 3, "height": 4}}]}),
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
    if matches!(id, "image.detection.detect" | "image.detection.detectFaces") {
        let plan = model_execution_plan(id);
        operation.input_schema["xExecutionPlan"] =
            runtime_core::surface_execution_plan_value(&plan);
        operation.output_schema["xExecutionPlan"] =
            runtime_core::surface_execution_plan_value(&plan);
        operation.wasm_supported = false;
    }
    if let Some(contract) = landscape_contract(id) {
        runtime_core::attach_landscape_contract(&mut operation, contract);
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
        progress_unit: Some("detections".to_string()),
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

fn landscape_contract(id: &str) -> Option<runtime_core::landscape::LandscapeOperationContract> {
    match id {
        "image.detection.detect" | "image.detection.detectFaces" | "image.detection.colorBlob" => {
            let function = match id {
                "image.detection.detect" => "image.detection.detect",
                "image.detection.detectFaces" => "image.detection.detectFaces",
                _ => "image.detection.detectColorBlob",
            };
            Some(runtime_core::landscape::LandscapeOperationContract::new(
                runtime_core::landscape::LandscapeFunction::new(function, env!("CARGO_PKG_NAME"))
                    .input(runtime_core::landscape::LandscapePort::new(
                        "image",
                        runtime_core::landscape::well_known::image_image(),
                    ))
                    .input(runtime_core::landscape::LandscapePort::new(
                        "request",
                        runtime_core::landscape::well_known::image_detection_request(),
                    ))
                    .output(
                        runtime_core::landscape::LandscapePort::new(
                            "detections",
                            runtime_core::landscape::well_known::vision_detection(),
                        )
                        .many(),
                    ),
            ))
        }
        _ => None,
    }
}

/// Runs one library-owned operation.
pub fn run_surface_operation(request: SurfaceRequest) -> Result<SurfaceResponse, String> {
    let surface = package_surface();
    let operation = request.operation.clone();
    let value = match request.operation.as_str() {
        "describe" => return Ok(describe_surface_response(&surface, request)),
        "image.detection.detect" => detect_value(parse_input(request.input)?)?,
        "image.detection.detectFaces" => detect_faces_value(parse_input(request.input)?)?,
        "image.detection.colorBlob" => color_blob_value(parse_input(request.input)?)?,
        "image.detection.models" => models_value(),
        "image.detection.boxSummary" => box_summary_value(parse_input(request.input)?)?,
        operation => {
            return Err(format!(
                "unsupported operation `{operation}` for {}",
                env!("CARGO_PKG_NAME")
            ))
        }
    };
    Ok(structured_operation_response(&surface, operation, value))
}

fn parse_input<T: DeserializeOwned>(input: serde_json::Value) -> Result<T, String> {
    serde_json::from_value(input).map_err(|error| format!("invalid request: {error}"))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DetectRequest {
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
    min_score: Option<f32>,
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FaceDetectRequest {
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
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ColorBlobRequest {
    image: ImagePayload,
    target_color: String,
    min_area_pixels: Option<usize>,
    merge_adjacent: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BoxSummaryRequest {
    #[serde(default)]
    detections: Vec<DetectionPayload>,
    #[serde(default)]
    boxes: Vec<BoxPayload>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DetectionPayload {
    label: Option<String>,
    score: Option<f32>,
    #[serde(alias = "box")]
    region: BoxPayload,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BoxPayload {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

fn detect_value(request: DetectRequest) -> Result<serde_json::Value, String> {
    validate_image_source(
        "image.detection.detect",
        request.image.as_ref(),
        request.image_path.as_deref(),
    )?;

    let model = request
        .model
        .unwrap_or_else(|| DEFAULT_DETECTION_MODEL.to_string());
    if model != DEFAULT_DETECTION_MODEL {
        return Err(format!(
            "unsupported object detection model `{model}`; expected `{DEFAULT_DETECTION_MODEL}`"
        ));
    }

    let min_score = request.min_score.unwrap_or(0.5);
    if !min_score.is_finite() || !(0.0..=1.0).contains(&min_score) {
        return Err("detection minScore must be finite and between 0 and 1".to_string());
    }
    let limit = request.limit.unwrap_or(100);
    if limit == 0 {
        return Err("detection limit must be greater than zero".to_string());
    }

    let auto_download = request.auto_download.unwrap_or(false);
    let spec = ModelPreset::XenovaDetrResnet50Onnx.spec();
    let (bundle, bundle_path) = resolve_bundle(
        &spec,
        auto_download,
        request.model_root.as_deref(),
    )?;
    let mut detector = OnnxObjectDetector::from_bundle(bundle).map_err(|error| error.to_string())?;
    detector.score_threshold = min_score;

    let mut detections = if let Some(image) = request.image {
        let view = image.view().map_err(|error| error.to_string())?;
        detector
            .detect_image(&view)
            .map_err(|error| error.to_string())?
    } else {
        let image_path = request.image_path.expect("validated imagePath");
        let image = image_analysis_io::read_image(&image_path)
            .map_err(|error| format!("failed to read detection image `{image_path}`: {error}"))?;
        detector
            .detect_image(&image.as_view())
            .map_err(|error| error.to_string())?
    };
    detections.sort_by(|left, right| {
        right
            .score
            .unwrap_or(f32::NEG_INFINITY)
            .total_cmp(&left.score.unwrap_or(f32::NEG_INFINITY))
    });
    detections.truncate(limit);

    Ok(serde_json::json!({
        "executed": true,
        "nativeOnly": true,
        "cliSupported": true,
        "wasmSupported": false,
        "serverSupported": true,
        "count": detections.len(),
        "detections": detections.iter().map(detection_json).collect::<Vec<_>>(),
        "modelId": DEFAULT_DETECTION_MODEL,
        "bundlePath": bundle_path,
        "runtime": "onnx",
        "autoDownload": auto_download,
        "minScore": min_score,
        "limit": limit
    }))
}

fn detect_faces_value(request: FaceDetectRequest) -> Result<serde_json::Value, String> {
    validate_image_source(
        "image.detection.detectFaces",
        request.image.as_ref(),
        request.image_path.as_deref(),
    )?;

    let model = request
        .model
        .unwrap_or_else(|| DEFAULT_FACE_DETECTION_MODEL.to_string());
    if model != DEFAULT_FACE_DETECTION_MODEL {
        return Err(format!(
            "unsupported face detection model `{model}`; expected `{DEFAULT_FACE_DETECTION_MODEL}`"
        ));
    }
    let limit = request.limit.unwrap_or(100);
    if limit == 0 {
        return Err("face detection limit must be greater than zero".to_string());
    }

    let auto_download = request.auto_download.unwrap_or(false);
    let spec = FaceDetectionPreset::OpenCvYuNet.model_spec();
    let (bundle, bundle_path) = resolve_bundle(
        &spec,
        auto_download,
        request.model_root.as_deref(),
    )?;
    let mut detector = OnnxFaceDetector::from_bundle(bundle).map_err(|error| error.to_string())?;

    let (mut detections, width, height) = if let Some(image) = request.image {
        let view = image.view().map_err(|error| error.to_string())?;
        let width = view.width;
        let height = view.height;
        let detections = detector
            .detect_faces(&view)
            .map_err(|error| error.to_string())?;
        (detections, width, height)
    } else {
        let image_path = request.image_path.expect("validated imagePath");
        let image = image_analysis_io::read_image(&image_path)
            .map_err(|error| format!("failed to read face detection image `{image_path}`: {error}"))?;
        let view = image.as_view();
        let width = view.width;
        let height = view.height;
        let detections = detector
            .detect_faces(&view)
            .map_err(|error| error.to_string())?;
        (detections, width, height)
    };
    detections.sort_by(|left, right| right.confidence.total_cmp(&left.confidence));
    detections.truncate(limit);

    Ok(serde_json::json!({
        "executed": true,
        "nativeOnly": true,
        "cliSupported": true,
        "wasmSupported": false,
        "serverSupported": true,
        "count": detections.len(),
        "detections": detections
            .iter()
            .map(|detection| face_detection_json(detection, width, height))
            .collect::<Vec<_>>(),
        "imageSize": {"width": width, "height": height},
        "modelId": DEFAULT_FACE_DETECTION_MODEL,
        "bundlePath": bundle_path,
        "runtime": "onnx",
        "autoDownload": auto_download,
        "limit": limit
    }))
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

fn resolve_bundle(
    spec: &model_runtime::HuggingFaceModelSpec,
    auto_download: bool,
    model_root: Option<&str>,
) -> Result<(model_runtime::ModelBundle, String), String> {
    let mut bundle_options = ModelBundleResolveOptions {
        auto_download,
        download_progress: auto_download,
        ..ModelBundleResolveOptions::default()
    };
    if let Some(model_root) = model_root {
        bundle_options.bundle_root = PathBuf::from(model_root);
    }
    let bundle_path = bundle_options.bundle_root.display().to_string();
    let bundle = model_runtime::resolve_or_download_bundle(spec, &bundle_options)
        .map_err(|error| error.to_string())?;
    Ok((bundle, bundle_path))
}

fn detection_json(detection: &ImageDetection) -> serde_json::Value {
    serde_json::json!({
        "label": &detection.label,
        "score": detection.score,
        "region": {
            "x": detection.region.x,
            "y": detection.region.y,
            "width": detection.region.width,
            "height": detection.region.height
        },
        "attributes": &detection.attributes
    })
}

fn face_detection_json(
    detection: &FaceDetection,
    image_width: u32,
    image_height: u32,
) -> serde_json::Value {
    let width = image_width as f32;
    let height = image_height as f32;
    let x = (detection.bbox.x * width).floor().clamp(0.0, width) as u32;
    let y = (detection.bbox.y * height).floor().clamp(0.0, height) as u32;
    let max_x = ((detection.bbox.x + detection.bbox.width) * width)
        .ceil()
        .clamp(0.0, width) as u32;
    let max_y = ((detection.bbox.y + detection.bbox.height) * height)
        .ceil()
        .clamp(0.0, height) as u32;
    let pixel_width = max_x.saturating_sub(x);
    let pixel_height = max_y.saturating_sub(y);

    serde_json::json!({
        "label": "face",
        "score": detection.confidence,
        "region": {
            "x": x,
            "y": y,
            "width": pixel_width,
            "height": pixel_height
        },
        "normalizedRegion": {
            "x": detection.bbox.x,
            "y": detection.bbox.y,
            "width": detection.bbox.width,
            "height": detection.bbox.height
        },
        "landmarks": detection.landmarks.as_ref().map(|landmarks| &landmarks.points),
        "attributes": &detection.attributes
    })
}

fn color_blob_value(request: ColorBlobRequest) -> Result<serde_json::Value, String> {
    let target = match request.target_color.as_str() {
        "red" => TargetColor::Red,
        other => return Err(format!("unsupported target color `{other}`")),
    };
    let mut options =
        ColorBlobDetectionOptions::default().min_area_pixels(request.min_area_pixels.unwrap_or(24));
    options.target = target;
    options.morph_open_3x3 = request.merge_adjacent.unwrap_or(true);
    let image = request.image.view().map_err(|error| error.to_string())?;
    let mut detector = ColorBlobDetector::new(options).map_err(|error| error.to_string())?;
    let detections = detector
        .detect_image(&image)
        .map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "count": detections.len(),
        "detections": detections.iter().map(detection_json).collect::<Vec<_>>()
    }))
}

fn models_value() -> serde_json::Value {
    let detr_spec = ModelPreset::XenovaDetrResnet50Onnx.spec();
    let face_spec = FaceDetectionPreset::OpenCvYuNet.model_spec();
    serde_json::json!({
        "models": [
            {
                "id": DEFAULT_DETECTION_MODEL,
                "task": detr_spec.task.as_protocol_str(),
                "repoId": detr_spec.repo_id_value(),
                "revision": detr_spec.revision_value(),
                "name": detr_spec.name,
                "files": detr_spec.files,
                "supported": cfg!(feature = "onnx"),
                "operation": "image.detection.detect"
            },
            {
                "id": DEFAULT_FACE_DETECTION_MODEL,
                "task": face_spec.task.as_protocol_str(),
                "repoId": face_spec.repo_id_value(),
                "revision": face_spec.revision_value(),
                "name": face_spec.name,
                "files": face_spec.files,
                "supported": cfg!(feature = "onnx"),
                "operation": "image.detection.detectFaces"
            }
        ],
        "deterministicFallbacks": [{
            "id": "color-blob-red",
            "operation": "image.detection.colorBlob",
            "supported": true
        }]
    })
}

fn box_summary_value(request: BoxSummaryRequest) -> Result<serde_json::Value, String> {
    let mut entries = Vec::new();
    for detection in request.detections {
        if let Some(score) = detection.score {
            if !score.is_finite() {
                return Err("detection score must be finite".to_string());
            }
        }
        let region = detection.region.bounding_box()?;
        entries.push((detection.label, detection.score, region));
    }
    for region in request.boxes {
        entries.push((None, None, region.bounding_box()?));
    }
    let union = union_bounds(entries.iter().map(|(_, _, region)| *region));
    let scored = entries
        .iter()
        .filter_map(|(_, score, _)| *score)
        .collect::<Vec<_>>();
    let average_score = if scored.is_empty() {
        None
    } else {
        Some(scored.iter().sum::<f32>() / scored.len() as f32)
    };
    let labels = entries
        .iter()
        .filter_map(|(label, _, _)| label.as_deref())
        .collect::<Vec<_>>();
    Ok(serde_json::json!({
        "count": entries.len(),
        "scoredCount": scored.len(),
        "averageScore": average_score,
        "labels": labels,
        "unionBounds": union.map(|region| serde_json::json!({
            "x": region.x,
            "y": region.y,
            "width": region.width,
            "height": region.height
        }))
    }))
}

impl BoxPayload {
    fn bounding_box(self) -> Result<BoundingBox, String> {
        BoundingBox::new(self.x, self.y, self.width, self.height).map_err(|error| error.to_string())
    }
}

fn union_bounds(regions: impl IntoIterator<Item = BoundingBox>) -> Option<BoundingBox> {
    let mut regions = regions.into_iter();
    let first = regions.next()?;
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (
        first.x,
        first.y,
        first.x + first.width,
        first.y + first.height,
    );
    for region in regions {
        min_x = min_x.min(region.x);
        min_y = min_y.min(region.y);
        max_x = max_x.max(region.x + region.width);
        max_y = max_y.max(region.y + region.height);
    }
    BoundingBox::new(min_x, min_y, max_x - min_x, max_y - min_y).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_surface_lists_detection_operations() {
        let ids = package_surface()
            .operations
            .into_iter()
            .map(|operation| operation.id.0)
            .collect::<Vec<_>>();
        assert!(ids.contains(&"image.detection.detect".to_string()));
        assert!(ids.contains(&"image.detection.detectFaces".to_string()));
        assert!(ids.contains(&"image.detection.colorBlob".to_string()));
        assert!(ids.contains(&"image.detection.models".to_string()));
    }

    #[test]
    fn detect_requires_exactly_one_image_source() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.detection.detect"),
            input: serde_json::json!({"model": DEFAULT_DETECTION_MODEL}),
        })
        .expect_err("missing image");
        assert!(error.contains("requires image or imagePath"));
    }

    #[test]
    fn face_detect_requires_exactly_one_image_source() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.detection.detectFaces"),
            input: serde_json::json!({"model": DEFAULT_FACE_DETECTION_MODEL}),
        })
        .expect_err("missing image");
        assert!(error.contains("requires image or imagePath"));
    }

    #[test]
    fn detect_rejects_unsupported_model_before_bundle_resolution() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.detection.detect"),
            input: serde_json::json!({
                "image": sample_image_json(),
                "model": "yolos-tiny",
                "autoDownload": false
            }),
        })
        .expect_err("unsupported model");
        assert!(error.contains("unsupported object detection model"));
    }

    #[test]
    fn face_detect_rejects_unsupported_model_before_bundle_resolution() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.detection.detectFaces"),
            input: serde_json::json!({
                "image": sample_image_json(),
                "model": "other-face-model",
                "autoDownload": false
            }),
        })
        .expect_err("unsupported model");
        assert!(error.contains("unsupported face detection model"));
    }

    #[test]
    fn detect_rejects_invalid_threshold_before_bundle_resolution() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.detection.detect"),
            input: serde_json::json!({
                "image": sample_image_json(),
                "minScore": 1.5,
                "autoDownload": false
            }),
        })
        .expect_err("invalid threshold");
        assert!(error.contains("between 0 and 1"));
    }

    #[test]
    fn face_detect_rejects_zero_limit_before_bundle_resolution() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.detection.detectFaces"),
            input: serde_json::json!({
                "image": sample_image_json(),
                "limit": 0,
                "autoDownload": false
            }),
        })
        .expect_err("invalid limit");
        assert!(error.contains("greater than zero"));
    }

    #[test]
    fn face_json_exposes_pixel_and_normalized_regions() {
        let detection = FaceDetection::new(
            crate::FaceBox::new(0.25, 0.5, 0.5, 0.25).expect("box"),
            0.95,
        )
        .expect("detection");
        let value = face_detection_json(&detection, 200, 100);
        assert_eq!(value["region"]["x"], 50);
        assert_eq!(value["region"]["y"], 50);
        assert_eq!(value["region"]["width"], 100);
        assert_eq!(value["region"]["height"], 25);
        assert_eq!(value["normalizedRegion"]["x"], 0.25);
    }

    #[test]
    fn detects_red_block() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.detection.colorBlob"),
            input: serde_json::json!({"image": sample_image_json(), "targetColor": "red", "minAreaPixels": 1, "mergeAdjacent": false}),
        })
        .expect("color blob");
        assert_eq!(response.value["count"], 1);
    }

    #[test]
    fn empty_colors_produce_zero_detections() {
        let mut image = sample_image_json();
        image["data"] = serde_json::json!([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.detection.colorBlob"),
            input: serde_json::json!({"image": image, "targetColor": "red", "minAreaPixels": 1}),
        })
        .expect("color blob");
        assert_eq!(response.value["count"], 0);
    }

    #[test]
    fn unsupported_target_color_errors() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.detection.colorBlob"),
            input: serde_json::json!({"image": sample_image_json(), "targetColor": "blue"}),
        })
        .expect_err("unsupported color");
        assert!(error.contains("unsupported target color"));
    }

    #[test]
    fn models_operation_reports_detection_catalog() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.detection.models"),
            input: serde_json::json!({}),
        })
        .expect("models");
        assert_eq!(response.value["models"][0]["id"], DEFAULT_DETECTION_MODEL);
        assert_eq!(response.value["models"][1]["id"], DEFAULT_FACE_DETECTION_MODEL);
        assert_eq!(
            response.value["models"][1]["operation"],
            "image.detection.detectFaces"
        );
    }

    #[test]
    fn box_summary_returns_union_bounds() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.detection.boxSummary"),
            input: serde_json::json!({
                "detections": [
                    {"label": "a", "score": 0.5, "region": {"x": 1, "y": 1, "width": 2, "height": 2}},
                    {"label": "b", "score": 1.0, "region": {"x": 4, "y": 2, "width": 2, "height": 1}}
                ]
            }),
        })
        .expect("box summary");
        assert_eq!(response.value["count"], 2);
        assert_eq!(response.value["unionBounds"]["width"], 5);
    }

    #[test]
    fn box_summary_rejects_empty_boxes() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.detection.boxSummary"),
            input: serde_json::json!({"boxes": [{"x": 0, "y": 0, "width": 0, "height": 1}]}),
        })
        .expect_err("invalid box");
        assert!(error.contains("bounding box"));
    }
}
