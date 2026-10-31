//! ONNX model bundle validation and the options derived from a bundle.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use image_analysis_processing::{
    image_model_preprocessing_from_config, validate_image_model_preprocessing,
    ImageModelPreprocessing,
};
use model_runtime::ModelTask;
use video_analysis_core::{DetectError, Result};

use crate::{BoxFormat, OnnxFaceDetectionOptions, OnnxObjectDetectionOptions};

#[derive(Debug, Clone, PartialEq)]
/// Data type for ONNX vision bundle info.
pub struct OnnxVisionBundleInfo {
    /// The config path value.
    pub config_path: PathBuf,
    /// The preprocessor config path value.
    pub preprocessor_config_path: Option<PathBuf>,
    /// The model path value.
    pub model_path: PathBuf,
    /// The labels value.
    pub labels: BTreeMap<i64, String>,
}

/// Validates ONNX vision bundle.
pub fn validate_onnx_vision_bundle(
    bundle: &model_runtime::ModelBundle,
) -> Result<OnnxVisionBundleInfo> {
    match bundle.manifest.task {
        ModelTask::ObjectDetection | ModelTask::FaceDetection => {}
        ModelTask::Custom(ref task) if task == "face_detection" => {}
        ref task => {
            return Err(DetectError::InvalidArgument(format!(
            "ONNX detection bundle {} task must be object_detection or face_detection, got {:?}",
            bundle_descriptor(bundle),
            task
        )))
        }
    }
    let config_path = required_bundle_file(bundle, "config.json")?;
    let preprocessor_config_path = bundle.file_path("preprocessor_config.json");
    let onnx_files = bundle_files_with_extension(bundle, "onnx");
    let model_path = match onnx_files.as_slice() {
        [path] => path.clone(),
        [] => {
            return Err(DetectError::InvalidArgument(format!(
                "ONNX detection bundle {} must contain exactly one `.onnx` model file selected by extension, found 0",
                bundle_descriptor(bundle)
            )))
        }
        files => {
            return Err(DetectError::InvalidArgument(format!(
                "ONNX detection bundle {} must contain exactly one `.onnx` model file selected by extension, found {}",
                bundle_descriptor(bundle),
                files.len()
            )))
        }
    };
    let config = read_json(&config_path)?;
    Ok(OnnxVisionBundleInfo {
        config_path,
        preprocessor_config_path,
        model_path,
        labels: label_map_from_config(&config)?,
    })
}

/// Returns object detection options from bundle.
pub fn object_detection_options_from_bundle(
    bundle: &model_runtime::ModelBundle,
) -> Result<OnnxObjectDetectionOptions> {
    if bundle.manifest.task != ModelTask::ObjectDetection {
        return Err(DetectError::InvalidArgument(format!(
            "ONNX object detection bundle task must be object_detection, got {:?}",
            bundle.manifest.task
        )));
    }
    let info = validate_onnx_vision_bundle(bundle)?;
    let config = read_json(&info.config_path)?;
    let preprocessing = if let Some(path) = &info.preprocessor_config_path {
        image_model_preprocessing_from_config(&read_json(path)?)?
    } else {
        ImageModelPreprocessing::default()
    };
    validate_image_model_preprocessing(&preprocessing)?;
    Ok(OnnxObjectDetectionOptions {
        preprocessing,
        score_threshold: 0.0,
        labels: info.labels,
        box_format: box_format_from_config(&config),
    })
}

/// Compatibility alias for earlier helper name.
pub fn options_from_bundle(
    bundle: &model_runtime::ModelBundle,
) -> Result<OnnxObjectDetectionOptions> {
    object_detection_options_from_bundle(bundle)
}

/// Returns face detection options from bundle.
pub fn face_detection_options_from_bundle(
    bundle: &model_runtime::ModelBundle,
) -> Result<OnnxFaceDetectionOptions> {
    let info = validate_onnx_face_detection_bundle(bundle)?;
    let preprocessing = if let Some(path) = &info.preprocessor_config_path {
        image_model_preprocessing_from_config(&read_json(path)?)?
    } else {
        OnnxFaceDetectionOptions::default().preprocessing
    };
    validate_image_model_preprocessing(&preprocessing)?;
    Ok(OnnxFaceDetectionOptions {
        preprocessing,
        ..OnnxFaceDetectionOptions::default()
    })
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OnnxFaceDetectionBundleInfo {
    preprocessor_config_path: Option<PathBuf>,
    pub(crate) model_path: PathBuf,
}

pub(crate) fn validate_onnx_face_detection_bundle(
    bundle: &model_runtime::ModelBundle,
) -> Result<OnnxFaceDetectionBundleInfo> {
    match &bundle.manifest.task {
        ModelTask::FaceDetection => {}
        ModelTask::Custom(task) if task == "face_detection" => {}
        _ => {
            return Err(DetectError::InvalidArgument(format!(
                "ONNX face detection bundle {} task must be face_detection, got {:?}",
                bundle_descriptor(bundle),
                bundle.manifest.task
            )))
        }
    }
    let onnx_files = bundle_files_with_extension(bundle, "onnx");
    let model_path = match onnx_files.as_slice() {
        [path] => path.clone(),
        [] => {
            return Err(DetectError::InvalidArgument(format!(
                "ONNX face detection bundle {} must contain exactly one `.onnx` model file selected by extension, found 0",
                bundle_descriptor(bundle)
            )))
        }
        files => {
            return Err(DetectError::InvalidArgument(format!(
                "ONNX face detection bundle {} must contain exactly one `.onnx` model file selected by extension, found {}",
                bundle_descriptor(bundle),
                files.len()
            )))
        }
    };
    Ok(OnnxFaceDetectionBundleInfo {
        preprocessor_config_path: bundle.file_path("preprocessor_config.json"),
        model_path,
    })
}

fn box_format_from_config(config: &serde_json::Value) -> BoxFormat {
    let value = config
        .get("video_analysis")
        .and_then(|value| value.get("box_format"))
        .or_else(|| config.get("box_format"))
        .and_then(serde_json::Value::as_str);
    match value {
        Some("xyxy_absolute") => BoxFormat::XyxyAbsolute,
        Some("xyxy_normalized") => BoxFormat::XyxyNormalized,
        Some("cxcywh_absolute") => BoxFormat::CxcywhAbsolute,
        _ => BoxFormat::CxcywhNormalized,
    }
}

pub(crate) fn validate_threshold(value: f32) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(DetectError::InvalidArgument(
            "ONNX score threshold must be finite and in the range 0..=1".to_string(),
        ));
    }
    Ok(())
}

fn required_bundle_file(bundle: &model_runtime::ModelBundle, remote_path: &str) -> Result<PathBuf> {
    bundle.file_path(remote_path).ok_or_else(|| {
        DetectError::InvalidArgument(format!(
            "model bundle {} is missing required file `{remote_path}`",
            bundle_descriptor(bundle)
        ))
    })
}

fn bundle_descriptor(bundle: &model_runtime::ModelBundle) -> String {
    format!(
        "`{}` (task {:?})",
        bundle.manifest.name, bundle.manifest.task
    )
}

fn bundle_files_with_extension(
    bundle: &model_runtime::ModelBundle,
    extension: &str,
) -> Vec<PathBuf> {
    bundle
        .manifest
        .files
        .values()
        .filter(|file| {
            Path::new(&file.remote_path)
                .extension()
                .and_then(|value| value.to_str())
                == Some(extension)
        })
        .map(|file| bundle.root.join(&file.local_path))
        .collect()
}

fn read_json(path: &Path) -> Result<serde_json::Value> {
    let data = std::fs::read(path)?;
    serde_json::from_slice(&data).map_err(|err| {
        DetectError::Source(format!("failed to parse JSON `{}`: {err}", path.display()))
    })
}

fn label_map_from_config(config: &serde_json::Value) -> Result<BTreeMap<i64, String>> {
    let Some(id2label) = config
        .get("id2label")
        .and_then(serde_json::Value::as_object)
    else {
        return Ok(BTreeMap::new());
    };
    let mut labels = BTreeMap::new();
    for (key, value) in id2label {
        let id = key.parse::<i64>().map_err(|err| {
            DetectError::InvalidArgument(format!("invalid id2label key `{key}`: {err}"))
        })?;
        let label = value.as_str().ok_or_else(|| {
            DetectError::InvalidArgument(format!("id2label value for `{key}` must be a string"))
        })?;
        labels.insert(id, label.to_string());
    }
    Ok(labels)
}

#[cfg(feature = "onnx")]
pub(crate) fn runtime_onnx_error(error: runtime_onnx::OnnxRuntimeError) -> DetectError {
    match error {
        runtime_onnx::OnnxRuntimeError::InvalidArgument(message)
        | runtime_onnx::OnnxRuntimeError::InvalidTensorShape(message) => {
            DetectError::InvalidArgument(message)
        }
        runtime_onnx::OnnxRuntimeError::Io(error) => DetectError::Io(error),
        other => DetectError::Source(other.to_string()),
    }
}
