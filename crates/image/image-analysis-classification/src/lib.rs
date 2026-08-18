#![doc = include_str!("../README.md")]

pub mod surface;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use image_analysis_core::ImageView;
use image_analysis_processing::{
    image_model_preprocessing_from_config, preprocess_image_for_model,
    validate_image_model_preprocessing, ImageModelPreprocessing, ImageModelTensor,
};
use model_runtime::{HuggingFaceModelSpec, ModelRuntimeBackend, ModelTask};
use serde::{Deserialize, Serialize};
use video_analysis_core::DetectError;
use video_analysis_core::Result;

/// Classification task family for still images.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageClassificationTask {
    /// Whole-image classification.
    ImageClassification,
}

impl ImageClassificationTask {
    /// Returns all classification task variants.
    pub const ALL: &'static [Self] = &[Self::ImageClassification];

    /// Returns the stable API path segment for this task.
    pub fn path_segment(self) -> &'static str {
        match self {
            Self::ImageClassification => "classify",
        }
    }
}

/// Runtime families for image classification execution and postprocessing.
pub type ImageClassificationRuntime = ModelRuntimeBackend;

/// Classification model metadata for UI and CLI discovery.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageClassificationCatalogEntry {
    /// Stable local entry id.
    pub id: String,
    /// Upstream model id or strategy id.
    pub model_id: String,
    /// Task family.
    pub task: ImageClassificationTask,
    /// Preferred runtime.
    pub runtime: ImageClassificationRuntime,
    /// Whether the runtime is available in the default contributor build.
    pub supported: bool,
    /// Optional fallback entry id.
    pub fallback: Option<String>,
    /// Human-readable note for unsupported or fallback-only paths.
    pub note: Option<String>,
}

/// Variants describing image classification preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageClassificationPreset {
    /// The ViT base patch16 224 variant.
    #[default]
    VitBasePatch16_224,
}

impl ImageClassificationPreset {
    /// Returns the model spec for this preset.
    pub fn model_spec(self) -> HuggingFaceModelSpec {
        match self {
            Self::VitBasePatch16_224 => HuggingFaceModelSpec::new(
                "Xenova/vit-base-patch16-224",
                ModelTask::ImageClassification,
            )
            .name("vit-base-patch16-224")
            .file("config.json")
            .file("preprocessor_config.json")
            .first_available_file(["onnx/model_quantized.onnx", "onnx/model.onnx"]),
        }
    }
}

/// One image classification prediction.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageClassification {
    /// Label assigned to this value.
    pub label: String,
    /// Score assigned to this value.
    pub score: Option<f32>,
    /// Adapter-specific attributes.
    pub attributes: BTreeMap<String, String>,
}

impl ImageClassification {
    /// Creates a new classification prediction.
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            score: None,
            attributes: BTreeMap::new(),
        }
    }

    /// Sets the score.
    pub fn score(mut self, value: f32) -> Self {
        self.score = Some(value);
        self
    }

    /// Adds an adapter-specific attribute.
    pub fn attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }
}

/// Backend trait for image classification.
pub trait ImageClassifierBackend {
    /// Classifies one image.
    fn classify_image(&mut self, image: &ImageView<'_>) -> Result<Vec<ImageClassification>>;
}

/// Returns classification catalog entries.
pub fn image_classification_catalog(
    task: Option<ImageClassificationTask>,
) -> Vec<ImageClassificationCatalogEntry> {
    let entries = vec![catalog_entry(
        "vit-base-patch16-224",
        ImageClassificationTask::ImageClassification,
        ImageClassificationPreset::VitBasePatch16_224.model_spec(),
        ModelRuntimeBackend::Onnx,
        cfg!(feature = "local-onnx"),
        Some("imported-classification"),
        Some("Server-only ONNX execution uses runtime-onnx inside image-analysis-classification; imported labels remain the WASM fallback."),
    )];

    entries
        .into_iter()
        .filter(|entry| task.is_none_or(|task| entry.task == task))
        .collect()
}

/// Compatibility alias for callers migrating off aggregate task catalog names.
pub fn model_catalog(
    task: Option<ImageClassificationTask>,
) -> Vec<ImageClassificationCatalogEntry> {
    image_classification_catalog(task)
}

/// Parses a classification task string.
pub fn parse_task(input: &str) -> Option<ImageClassificationTask> {
    ImageClassificationTask::ALL
        .iter()
        .copied()
        .find(|task| task.path_segment() == input)
        .or(match input {
            "image_classification" | "classification" => {
                Some(ImageClassificationTask::ImageClassification)
            }
            _ => None,
        })
}

/// Returns preset ids registered for image classification.
pub fn registered_image_classification_presets() -> Vec<String> {
    image_classification_catalog(None)
        .into_iter()
        .map(|entry| entry.id)
        .collect()
}

/// Returns a schema-oriented summary.
pub fn schema_summary() -> serde_json::Value {
    serde_json::json!({
        "tasks": ImageClassificationTask::ALL
            .iter()
            .map(|task| task.path_segment())
            .collect::<Vec<_>>(),
        "models": image_classification_catalog(None),
        "registeredPresets": registered_image_classification_presets()
    })
}

fn catalog_entry(
    id: &str,
    task: ImageClassificationTask,
    spec: HuggingFaceModelSpec,
    runtime: ModelRuntimeBackend,
    supported: bool,
    fallback: Option<&str>,
    note: Option<&str>,
) -> ImageClassificationCatalogEntry {
    ImageClassificationCatalogEntry {
        id: id.to_string(),
        model_id: spec.repo_id_value().unwrap_or(id).to_string(),
        task,
        runtime,
        supported,
        fallback: fallback.map(str::to_string),
        note: note.map(str::to_string),
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for ONNX image classification options.
pub struct OnnxImageClassificationOptions {
    /// The preprocessing value.
    pub preprocessing: ImageModelPreprocessing,
    /// The score threshold value.
    pub score_threshold: f32,
    /// The labels value.
    pub labels: BTreeMap<i64, String>,
    /// The top k value.
    pub top_k: usize,
}

impl Default for OnnxImageClassificationOptions {
    fn default() -> Self {
        Self {
            preprocessing: ImageModelPreprocessing::default(),
            score_threshold: 0.0,
            labels: BTreeMap::new(),
            top_k: 5,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for ONNX image classification output.
pub struct OnnxImageClassificationOutput {
    /// The class identifiers value.
    pub class_ids: Vec<i64>,
    /// The scores value.
    pub scores: Vec<f32>,
}

/// Trait for ONNX image classification runner implementations.
pub trait OnnxImageClassificationRunner {
    /// Runs image classification.
    fn run_image_classification(
        &mut self,
        input: &ImageModelTensor,
    ) -> Result<OnnxImageClassificationOutput>;
}

#[derive(Debug, Clone, Default)]
/// Data type for unavailable ONNX image classification runner.
pub struct UnavailableOnnxImageClassificationRunner;

impl OnnxImageClassificationRunner for UnavailableOnnxImageClassificationRunner {
    fn run_image_classification(
        &mut self,
        _input: &ImageModelTensor,
    ) -> Result<OnnxImageClassificationOutput> {
        Err(DetectError::Source(
            "native ONNX image classification execution is unavailable; construct with a runner or enable `local-onnx`"
                .to_string(),
        ))
    }
}

#[cfg(feature = "local-onnx")]
impl OnnxImageClassificationRunner for runtime_onnx::OnnxSession {
    fn run_image_classification(
        &mut self,
        input: &ImageModelTensor,
    ) -> Result<OnnxImageClassificationOutput> {
        use runtime_onnx::{OnnxRunner, OnnxTensor, OnnxTensorValue};

        let metadata = self.metadata().map_err(runtime_onnx_error)?;
        let input_name = runtime_onnx::input_name(&metadata, 0)
            .map_err(runtime_onnx_error)?
            .to_string();
        let outputs = self
            .run(vec![runtime_onnx::OnnxNamedTensor {
                name: input_name,
                tensor: OnnxTensorValue::F32(
                    OnnxTensor::new(
                        vec![
                            1,
                            input.channels,
                            input.height as usize,
                            input.width as usize,
                        ],
                        input.values.clone(),
                    )
                    .map_err(runtime_onnx_error)?,
                ),
            }])
            .map_err(runtime_onnx_error)?;
        let logits = outputs
            .iter()
            .find_map(|output| {
                runtime_onnx::f32_output_by_name_or_index(&outputs, &output.name, 0)
                    .ok()
                    .and_then(|tensor| classification_logits_from_tensor(tensor).ok())
            })
            .ok_or_else(|| {
                DetectError::InvalidArgument(
                    "ONNX image classification model did not return a logits tensor".to_string(),
                )
            })?;
        let probabilities = softmax(&logits);
        let mut ranked = probabilities
            .into_iter()
            .enumerate()
            .map(|(index, score)| (index as i64, score))
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
        Ok(OnnxImageClassificationOutput {
            class_ids: ranked.iter().map(|(class_id, _)| *class_id).collect(),
            scores: ranked.into_iter().map(|(_, score)| score).collect(),
        })
    }
}

#[derive(Debug)]
/// Data type for ONNX image classifier.
pub struct OnnxImageClassifier<R = UnavailableOnnxImageClassificationRunner> {
    options: OnnxImageClassificationOptions,
    runner: R,
}

#[cfg(not(feature = "local-onnx"))]
impl OnnxImageClassifier<UnavailableOnnxImageClassificationRunner> {
    /// Builds this value from bundle.
    pub fn from_bundle(bundle: model_runtime::ModelBundle) -> Result<Self> {
        Ok(Self {
            options: classification_options_from_bundle(&bundle)?,
            runner: UnavailableOnnxImageClassificationRunner,
        })
    }
}

#[cfg(feature = "local-onnx")]
impl OnnxImageClassifier<runtime_onnx::OnnxSession> {
    /// Builds this value from bundle.
    pub fn from_bundle(bundle: model_runtime::ModelBundle) -> Result<Self> {
        let info = validate_onnx_classification_bundle(&bundle)?;
        Ok(Self {
            options: classification_options_from_bundle(&bundle)?,
            runner: runtime_onnx::OnnxSession::from_file(info.model_path)
                .map_err(runtime_onnx_error)?,
        })
    }
}

impl<R: OnnxImageClassificationRunner> OnnxImageClassifier<R> {
    /// Builds this value from runner.
    pub fn from_runner(bundle: model_runtime::ModelBundle, runner: R) -> Result<Self> {
        Ok(Self {
            options: classification_options_from_bundle(&bundle)?,
            runner,
        })
    }

    /// Returns this value with options.
    pub fn with_options(options: OnnxImageClassificationOptions, runner: R) -> Result<Self> {
        validate_image_model_preprocessing(&options.preprocessing)?;
        validate_threshold(options.score_threshold)?;
        if options.top_k == 0 {
            return Err(DetectError::InvalidArgument(
                "ONNX classification top_k must be greater than zero".to_string(),
            ));
        }
        Ok(Self { options, runner })
    }

    /// Returns options.
    pub fn options(&self) -> &OnnxImageClassificationOptions {
        &self.options
    }
}

impl<R: OnnxImageClassificationRunner> ImageClassifierBackend for OnnxImageClassifier<R> {
    fn classify_image(&mut self, image: &ImageView<'_>) -> Result<Vec<ImageClassification>> {
        let input = preprocess_image_for_model(image, &self.options.preprocessing)?;
        let output = self.runner.run_image_classification(&input)?;
        if output.class_ids.len() != output.scores.len() {
            return Err(DetectError::InvalidArgument(
                "ONNX image classification outputs must have the same number of class ids and scores"
                    .to_string(),
            ));
        }
        let mut labels = output
            .class_ids
            .into_iter()
            .zip(output.scores)
            .filter(|(_, score)| score.is_finite() && *score >= self.options.score_threshold)
            .map(|(class_id, score)| {
                ImageClassification::new(
                    self.options
                        .labels
                        .get(&class_id)
                        .cloned()
                        .unwrap_or_else(|| format!("LABEL_{class_id}")),
                )
                .score(score)
            })
            .collect::<Vec<_>>();
        labels.sort_by(|left, right| {
            right
                .score
                .unwrap_or(0.0)
                .total_cmp(&left.score.unwrap_or(0.0))
        });
        labels.truncate(self.options.top_k);
        Ok(labels)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct OnnxClassificationBundleInfo {
    config_path: PathBuf,
    preprocessor_config_path: Option<PathBuf>,
    model_path: PathBuf,
    labels: BTreeMap<i64, String>,
}

fn validate_onnx_classification_bundle(
    bundle: &model_runtime::ModelBundle,
) -> Result<OnnxClassificationBundleInfo> {
    if bundle.manifest.task != ModelTask::ImageClassification {
        return Err(DetectError::InvalidArgument(format!(
            "ONNX image classification bundle task must be image_classification, got {:?}",
            bundle.manifest.task
        )));
    }
    let config_path = required_bundle_file(bundle, "config.json")?;
    let preprocessor_config_path = bundle.file_path("preprocessor_config.json");
    let model_path = first_bundle_file_with_extension(bundle, "onnx").ok_or_else(|| {
        DetectError::InvalidArgument(
            "ONNX image classification bundle must contain an `.onnx` model file".to_string(),
        )
    })?;
    let config = read_json(&config_path)?;
    Ok(OnnxClassificationBundleInfo {
        config_path,
        preprocessor_config_path,
        model_path,
        labels: label_map_from_config(&config)?,
    })
}

/// Returns classification options from bundle.
pub fn classification_options_from_bundle(
    bundle: &model_runtime::ModelBundle,
) -> Result<OnnxImageClassificationOptions> {
    let info = validate_onnx_classification_bundle(bundle)?;
    let preprocessing = if let Some(path) = &info.preprocessor_config_path {
        image_model_preprocessing_from_config(&read_json(path)?)?
    } else {
        ImageModelPreprocessing::default()
    };
    validate_image_model_preprocessing(&preprocessing)?;
    Ok(OnnxImageClassificationOptions {
        preprocessing,
        score_threshold: 0.0,
        labels: info.labels,
        top_k: 5,
    })
}

fn validate_threshold(value: f32) -> Result<()> {
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
            "model bundle `{}` is missing required file `{remote_path}`",
            bundle.manifest.name
        ))
    })
}

fn first_bundle_file_with_extension(
    bundle: &model_runtime::ModelBundle,
    extension: &str,
) -> Option<PathBuf> {
    bundle
        .manifest
        .files
        .values()
        .find(|file| {
            Path::new(&file.remote_path)
                .extension()
                .and_then(|value| value.to_str())
                == Some(extension)
        })
        .map(|file| bundle.root.join(&file.local_path))
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

#[cfg(feature = "local-onnx")]
fn classification_logits_from_tensor(tensor: &runtime_onnx::OnnxF32Tensor) -> Result<Vec<f32>> {
    if tensor.values.iter().any(|value| !value.is_finite()) {
        return Err(DetectError::InvalidArgument(
            "ONNX image classification logits must be finite".to_string(),
        ));
    }
    match tensor.shape.as_slice() {
        [classes] if *classes == tensor.values.len() => Ok(tensor.values.clone()),
        [1, classes] if *classes == tensor.values.len() => Ok(tensor.values.clone()),
        _ => Err(DetectError::InvalidArgument(format!(
            "unsupported ONNX image classification output shape `{:?}`",
            tensor.shape
        ))),
    }
}

#[cfg(feature = "local-onnx")]
fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities = logits
        .iter()
        .map(|value| (value - max).exp())
        .collect::<Vec<_>>();
    let total = probabilities.iter().sum::<f32>();
    if total <= f32::EPSILON || !total.is_finite() {
        return vec![0.0; logits.len()];
    }
    for probability in &mut probabilities {
        *probability /= total;
    }
    probabilities
}

#[cfg(feature = "local-onnx")]
fn runtime_onnx_error(error: runtime_onnx::OnnxRuntimeError) -> DetectError {
    match error {
        runtime_onnx::OnnxRuntimeError::InvalidArgument(message)
        | runtime_onnx::OnnxRuntimeError::InvalidTensorShape(message) => {
            DetectError::InvalidArgument(message)
        }
        runtime_onnx::OnnxRuntimeError::Io(error) => DetectError::Io(error),
        other => DetectError::Source(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubClassifier;

    impl ImageClassifierBackend for StubClassifier {
        fn classify_image(&mut self, _image: &ImageView<'_>) -> Result<Vec<ImageClassification>> {
            Ok(vec![ImageClassification::new("demo").score(0.9)])
        }
    }

    #[test]
    fn classification_catalog_reports_preset() {
        let catalog = image_classification_catalog(None);
        assert!(catalog
            .iter()
            .any(|entry| entry.id == "vit-base-patch16-224"));
        assert_eq!(
            parse_task("classify"),
            Some(ImageClassificationTask::ImageClassification)
        );
    }

    #[test]
    fn classifier_trait_returns_predictions() {
        let image = image_analysis_core::OwnedImage::new_rgb(1, 1, vec![0, 0, 0]).unwrap();
        let predictions = StubClassifier
            .classify_image(&image.as_view())
            .expect("classify");
        assert_eq!(predictions[0].label, "demo");
    }
}
