#![doc = include_str!("../README.md")]

pub mod surface;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use image_analysis_core::ImageView;
use image_analysis_processing::{
    image_model_preprocessing_from_config, preprocess_image_for_model,
    validate_image_model_preprocessing, ImageModelPreprocessing,
};
use model_runtime::{HuggingFaceModelSpec, ModelRuntimeBackend, ModelTask};
use serde::{Deserialize, Serialize};
use video_analysis_core::{DetectError, Result};

/// Captioning task family for still images.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageCaptioningTask {
    /// Image caption generation.
    ImageCaptioning,
}

impl ImageCaptioningTask {
    /// Returns all captioning task variants.
    pub const ALL: &'static [Self] = &[Self::ImageCaptioning];

    /// Returns the stable API path segment for this task.
    pub fn path_segment(self) -> &'static str {
        match self {
            Self::ImageCaptioning => "caption",
        }
    }
}

/// Runtime families for image captioning execution and postprocessing.
pub type ImageCaptioningRuntime = ModelRuntimeBackend;

/// Captioning model metadata for UI and CLI discovery.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageCaptioningCatalogEntry {
    /// Stable local entry id.
    pub id: String,
    /// Upstream model id or strategy id.
    pub model_id: String,
    /// Task family.
    pub task: ImageCaptioningTask,
    /// Preferred runtime.
    pub runtime: ImageCaptioningRuntime,
    /// Whether the runtime is available in the default contributor build.
    pub supported: bool,
    /// Optional fallback entry id.
    pub fallback: Option<String>,
    /// Human-readable note for unsupported or fallback-only paths.
    pub note: Option<String>,
}

/// Variants describing image caption preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageCaptionPreset {
    /// The ViT-GPT2 ONNX captioning variant.
    #[default]
    VitGpt2,
    /// The BLIP base variant.
    BlipBase,
}

impl ImageCaptionPreset {
    /// Returns the model spec for this preset.
    pub fn model_spec(self) -> HuggingFaceModelSpec {
        match self {
            Self::VitGpt2 => HuggingFaceModelSpec::new(
                "Xenova/vit-gpt2-image-captioning",
                ModelTask::Custom("image_captioning".to_string()),
            )
            .name("vit-gpt2-image-captioning")
            .file("config.json")
            .file("generation_config.json")
            .file("preprocessor_config.json")
            .file("tokenizer.json")
            .file("tokenizer_config.json")
            .file("vocab.json")
            .file("merges.txt")
            .first_available_file([
                "onnx/encoder_model_quantized.onnx",
                "onnx/encoder_model.onnx",
            ])
            .first_available_file([
                "onnx/decoder_model_quantized.onnx",
                "onnx/decoder_model.onnx",
            ]),
            Self::BlipBase => HuggingFaceModelSpec::new(
                "Salesforce/blip-image-captioning-base",
                ModelTask::Custom("image_captioning".to_string()),
            )
            .name("blip-image-captioning-base")
            .file("config.json")
            .file("preprocessor_config.json")
            .first_available_file(["model.safetensors", "pytorch_model.bin"]),
        }
    }
}

/// One generated image caption.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageCaption {
    /// Caption text.
    pub text: String,
    /// Optional score.
    pub score: Option<f32>,
    /// Adapter-specific attributes.
    pub attributes: BTreeMap<String, String>,
}

impl ImageCaption {
    /// Creates a new caption.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
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

/// Backend trait for image captioning.
pub trait ImageCaptionerBackend {
    /// Captions one image.
    fn caption_image(&mut self, image: &ImageView<'_>) -> Result<Vec<ImageCaption>>;
}

/// Returns captioning catalog entries.
pub fn image_captioning_catalog(
    task: Option<ImageCaptioningTask>,
) -> Vec<ImageCaptioningCatalogEntry> {
    let entries = vec![
        catalog_entry(
            "vit-gpt2-image-captioning",
            ImageCaptioningTask::ImageCaptioning,
            ImageCaptionPreset::VitGpt2.model_spec(),
            ModelRuntimeBackend::Onnx,
            cfg!(feature = "local-onnx"),
            Some("imported-caption"),
            Some("Server-only ONNX execution uses runtime-onnx inside image-analysis-captioning; imported captions remain the WASM fallback."),
        ),
        catalog_entry(
            "blip-image-captioning-base",
            ImageCaptioningTask::ImageCaptioning,
            ImageCaptionPreset::BlipBase.model_spec(),
            ModelRuntimeBackend::Onnx,
            false,
            Some("imported-caption"),
            Some("Reference-only until BLIP generation is implemented."),
        ),
    ];

    entries
        .into_iter()
        .filter(|entry| task.is_none_or(|task| entry.task == task))
        .collect()
}

/// Compatibility alias for callers migrating off aggregate task catalog names.
pub fn model_catalog(task: Option<ImageCaptioningTask>) -> Vec<ImageCaptioningCatalogEntry> {
    image_captioning_catalog(task)
}

/// Parses a captioning task string.
pub fn parse_task(input: &str) -> Option<ImageCaptioningTask> {
    ImageCaptioningTask::ALL
        .iter()
        .copied()
        .find(|task| task.path_segment() == input)
        .or(match input {
            "image_captioning" | "captioning" => Some(ImageCaptioningTask::ImageCaptioning),
            _ => None,
        })
}

/// Returns a schema-oriented summary.
pub fn schema_summary() -> serde_json::Value {
    serde_json::json!({
        "tasks": ImageCaptioningTask::ALL
            .iter()
            .map(|task| task.path_segment())
            .collect::<Vec<_>>(),
        "models": image_captioning_catalog(None)
    })
}

fn catalog_entry(
    id: &str,
    task: ImageCaptioningTask,
    spec: HuggingFaceModelSpec,
    runtime: ModelRuntimeBackend,
    supported: bool,
    fallback: Option<&str>,
    note: Option<&str>,
) -> ImageCaptioningCatalogEntry {
    ImageCaptioningCatalogEntry {
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
/// Data type for ONNX image captioning options.
pub struct OnnxImageCaptioningOptions {
    /// The preprocessing value.
    pub preprocessing: ImageModelPreprocessing,
    /// Maximum generated token count.
    pub max_new_tokens: usize,
    /// Decoder start token id.
    pub decoder_start_token_id: Option<u32>,
    /// End-of-sequence token id.
    pub eos_token_id: Option<u32>,
    /// Padding token id.
    pub pad_token_id: Option<u32>,
}

impl Default for OnnxImageCaptioningOptions {
    fn default() -> Self {
        Self {
            preprocessing: ImageModelPreprocessing {
                input_width: 224,
                input_height: 224,
                rescale_factor: 1.0 / 255.0,
                mean: [0.5, 0.5, 0.5],
                std: [0.5, 0.5, 0.5],
                channel_order: image_analysis_processing::ChannelOrder::Rgb,
            },
            max_new_tokens: 32,
            decoder_start_token_id: None,
            eos_token_id: None,
            pad_token_id: None,
        }
    }
}

#[derive(Debug)]
/// Data type for ONNX image captioner.
pub struct OnnxImageCaptioner {
    options: OnnxImageCaptioningOptions,
    #[cfg(not(feature = "local-onnx"))]
    encoder_path: PathBuf,
    #[cfg(not(feature = "local-onnx"))]
    decoder_path: PathBuf,
    #[cfg(not(feature = "local-onnx"))]
    tokenizer_path: PathBuf,
    #[cfg(feature = "local-onnx")]
    encoder: runtime_onnx::OnnxSession,
    #[cfg(feature = "local-onnx")]
    decoder: runtime_onnx::OnnxSession,
    #[cfg(feature = "local-onnx")]
    tokenizer: tokenizers::Tokenizer,
}

impl OnnxImageCaptioner {
    /// Builds this value from bundle.
    pub fn from_bundle(bundle: model_runtime::ModelBundle) -> Result<Self> {
        let info = validate_onnx_captioning_bundle(&bundle)?;
        #[cfg(not(feature = "local-onnx"))]
        let options = captioning_options_from_bundle(&bundle)?;
        #[cfg(feature = "local-onnx")]
        let mut options = captioning_options_from_bundle(&bundle)?;
        #[cfg(feature = "local-onnx")]
        {
            let tokenizer =
                tokenizers::Tokenizer::from_file(&info.tokenizer_path).map_err(|err| {
                    DetectError::Source(format!(
                        "failed to load tokenizer `{}`: {err}",
                        info.tokenizer_path.display()
                    ))
                })?;
            resolve_captioning_token_ids(&mut options, &tokenizer, &info)?;
            validate_captioning_options(&options)?;
            return Ok(Self {
                options,
                encoder: runtime_onnx::OnnxSession::from_file(&info.encoder_path)
                    .map_err(runtime_onnx_error)?,
                decoder: runtime_onnx::OnnxSession::from_file(&info.decoder_path)
                    .map_err(runtime_onnx_error)?,
                tokenizer,
            });
        }

        #[cfg(not(feature = "local-onnx"))]
        Ok(Self {
            options,
            encoder_path: info.encoder_path,
            decoder_path: info.decoder_path,
            tokenizer_path: info.tokenizer_path,
        })
    }

    /// Returns options.
    pub fn options(&self) -> &OnnxImageCaptioningOptions {
        &self.options
    }
}

impl ImageCaptionerBackend for OnnxImageCaptioner {
    fn caption_image(&mut self, image: &ImageView<'_>) -> Result<Vec<ImageCaption>> {
        let input = preprocess_image_for_model(image, &self.options.preprocessing)?;
        #[cfg(feature = "local-onnx")]
        {
            validate_captioning_options(&self.options)?;
            let encoder_hidden_states = run_caption_encoder(&mut self.encoder, &input)?;
            let decoder_start_token_id = self.options.decoder_start_token_id.ok_or_else(|| {
                DetectError::InvalidArgument(
                    "ONNX image captioning requires decoder_start_token_id".to_string(),
                )
            })?;
            let eos_token_id = self.options.eos_token_id.ok_or_else(|| {
                DetectError::InvalidArgument(
                    "ONNX image captioning requires eos_token_id".to_string(),
                )
            })?;
            let generated = greedy_decode_token_ids(
                decoder_start_token_id,
                eos_token_id,
                self.options.max_new_tokens,
                |input_ids| {
                    run_caption_decoder(&mut self.decoder, input_ids, &encoder_hidden_states)
                },
            )?;
            let text = self
                .tokenizer
                .decode(&generated, true)
                .map_err(|err| DetectError::Source(format!("failed to decode caption: {err}")))?
                .trim()
                .to_string();
            validate_decoded_caption_text(&text)?;
            Ok(vec![ImageCaption::new(text)])
        }

        #[cfg(not(feature = "local-onnx"))]
        {
            let _ = input;
            let _ = (&self.encoder_path, &self.decoder_path, &self.tokenizer_path);
            Err(DetectError::Source(
                "native ONNX image caption decoding is not available in this build path"
                    .to_string(),
            ))
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct OnnxCaptioningBundleInfo {
    config_path: PathBuf,
    preprocessor_config_path: Option<PathBuf>,
    tokenizer_config_path: Option<PathBuf>,
    tokenizer_path: PathBuf,
    encoder_path: PathBuf,
    decoder_path: PathBuf,
}

fn validate_onnx_captioning_bundle(
    bundle: &model_runtime::ModelBundle,
) -> Result<OnnxCaptioningBundleInfo> {
    match &bundle.manifest.task {
        ModelTask::Custom(task) if task == "image_captioning" => {}
        task => {
            return Err(DetectError::InvalidArgument(format!(
                "ONNX image captioning bundle {} task must be image_captioning, got {:?}",
                bundle_descriptor(bundle),
                task
            )))
        }
    }
    let config_path = required_bundle_file(bundle, "config.json")?;
    let preprocessor_config_path = bundle.file_path("preprocessor_config.json");
    let tokenizer_config_path = bundle.file_path("tokenizer_config.json");
    let tokenizer_path = required_bundle_file(bundle, "tokenizer.json")?;
    let encoder_path = bundle
        .manifest
        .files
        .values()
        .find(|file| file.remote_path.ends_with(".onnx") && file.remote_path.contains("encoder"))
        .map(|file| bundle.root.join(&file.local_path))
        .ok_or_else(|| {
            DetectError::InvalidArgument(format!(
                "ONNX image captioning bundle {} must contain an encoder `.onnx` model selected by remote path containing `encoder`",
                bundle_descriptor(bundle)
            ))
        })?;
    let decoder_path = bundle
        .manifest
        .files
        .values()
        .find(|file| file.remote_path.ends_with(".onnx") && file.remote_path.contains("decoder"))
        .map(|file| bundle.root.join(&file.local_path))
        .ok_or_else(|| {
            DetectError::InvalidArgument(format!(
                "ONNX image captioning bundle {} must contain a decoder `.onnx` model selected by remote path containing `decoder`",
                bundle_descriptor(bundle)
            ))
        })?;
    Ok(OnnxCaptioningBundleInfo {
        config_path,
        preprocessor_config_path,
        tokenizer_config_path,
        tokenizer_path,
        encoder_path,
        decoder_path,
    })
}

/// Returns ONNX image captioning options from a model-runtime bundle.
pub fn captioning_options_from_bundle(
    bundle: &model_runtime::ModelBundle,
) -> Result<OnnxImageCaptioningOptions> {
    let info = validate_onnx_captioning_bundle(bundle)?;
    let config = read_json(&info.config_path)?;
    let preprocessing = if let Some(path) = &info.preprocessor_config_path {
        image_model_preprocessing_from_config(&read_json(path)?)?
    } else {
        OnnxImageCaptioningOptions::default().preprocessing
    };
    validate_image_model_preprocessing(&preprocessing)?;
    let options = OnnxImageCaptioningOptions {
        preprocessing,
        max_new_tokens: config
            .get("max_new_tokens")
            .or_else(|| config.get("max_length"))
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(32),
        decoder_start_token_id: config_token_id(&config, "decoder_start_token_id"),
        eos_token_id: config_token_id(&config, "eos_token_id"),
        pad_token_id: config_token_id(&config, "pad_token_id"),
    };
    validate_captioning_max_tokens(options.max_new_tokens)?;
    Ok(options)
}

fn config_token_id(config: &serde_json::Value, key: &str) -> Option<u32> {
    config
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
}

#[cfg(any(feature = "local-onnx", test))]
fn validate_captioning_options(options: &OnnxImageCaptioningOptions) -> Result<()> {
    validate_image_model_preprocessing(&options.preprocessing)?;
    validate_captioning_max_tokens(options.max_new_tokens)?;
    if options.decoder_start_token_id.is_none() {
        return Err(DetectError::InvalidArgument(
            "ONNX image captioning requires decoder_start_token_id".to_string(),
        ));
    }
    if options.eos_token_id.is_none() {
        return Err(DetectError::InvalidArgument(
            "ONNX image captioning requires eos_token_id".to_string(),
        ));
    }
    Ok(())
}

fn validate_captioning_max_tokens(max_new_tokens: usize) -> Result<()> {
    if max_new_tokens == 0 {
        return Err(DetectError::InvalidArgument(
            "ONNX image captioning max_new_tokens must be greater than zero".to_string(),
        ));
    }
    Ok(())
}

#[cfg(any(feature = "local-onnx", test))]
fn greedy_decode_token_ids<F>(
    decoder_start_token_id: u32,
    eos_token_id: u32,
    max_new_tokens: usize,
    mut next_logits: F,
) -> Result<Vec<u32>>
where
    F: FnMut(&[u32]) -> Result<Vec<f32>>,
{
    validate_captioning_max_tokens(max_new_tokens)?;
    let mut input_ids = vec![decoder_start_token_id];
    let mut generated = Vec::new();
    for _ in 0..max_new_tokens {
        let logits = next_logits(&input_ids)?;
        let next = argmax(&logits)?;
        if next == eos_token_id {
            break;
        }
        input_ids.push(next);
        generated.push(next);
    }
    Ok(generated)
}

#[cfg(any(feature = "local-onnx", test))]
fn argmax(values: &[f32]) -> Result<u32> {
    values
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(index, _)| index as u32)
        .ok_or_else(|| {
            DetectError::InvalidArgument("ONNX caption decoder logits were empty".to_string())
        })
}

#[cfg(feature = "local-onnx")]
fn resolve_captioning_token_ids(
    options: &mut OnnxImageCaptioningOptions,
    tokenizer: &tokenizers::Tokenizer,
    info: &OnnxCaptioningBundleInfo,
) -> Result<()> {
    let tokenizer_config = info
        .tokenizer_config_path
        .as_deref()
        .map(read_json)
        .transpose()?;
    if options.decoder_start_token_id.is_none() {
        options.decoder_start_token_id = tokenizer_special_id(
            tokenizer,
            tokenizer_config.as_ref(),
            &["decoder_start_token", "bos_token", "cls_token"],
            &["<|startoftext|>", "<s>", "[CLS]"],
        );
    }
    if options.eos_token_id.is_none() {
        options.eos_token_id = tokenizer_special_id(
            tokenizer,
            tokenizer_config.as_ref(),
            &["eos_token", "sep_token"],
            &["<|endoftext|>", "</s>", "[SEP]"],
        );
    }
    if options.pad_token_id.is_none() {
        options.pad_token_id = tokenizer_special_id(
            tokenizer,
            tokenizer_config.as_ref(),
            &["pad_token"],
            &["<|endoftext|>", "<pad>", "[PAD]"],
        )
        .or(options.eos_token_id);
    }
    Ok(())
}

#[cfg(feature = "local-onnx")]
fn tokenizer_special_id(
    tokenizer: &tokenizers::Tokenizer,
    tokenizer_config: Option<&serde_json::Value>,
    config_keys: &[&str],
    fallback_tokens: &[&str],
) -> Option<u32> {
    fallback_tokens
        .iter()
        .copied()
        .chain(config_keys.iter().filter_map(|key| {
            tokenizer_config.and_then(|config| special_token_from_config(config, key))
        }))
        .find_map(|token| tokenizer.token_to_id(token))
}

#[cfg(feature = "local-onnx")]
fn special_token_from_config<'a>(config: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    let value = config.get(key)?;
    value
        .as_str()
        .or_else(|| value.get("content").and_then(serde_json::Value::as_str))
}

#[cfg(feature = "local-onnx")]
fn run_caption_encoder(
    encoder: &mut runtime_onnx::OnnxSession,
    input: &image_analysis_processing::ImageModelTensor,
) -> Result<runtime_onnx::OnnxF32Tensor> {
    use runtime_onnx::{OnnxRunner, OnnxTensor, OnnxTensorValue};

    let metadata = encoder.metadata().map_err(runtime_onnx_error)?;
    let input_name = runtime_onnx::input_name(&metadata, 0)
        .map_err(runtime_onnx_error)?
        .to_string();
    let outputs = encoder
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
    runtime_onnx::first_f32_output(&outputs)
        .map(Clone::clone)
        .map_err(runtime_onnx_error)
        .and_then(|tensor| {
            validate_encoder_hidden_states(&tensor)?;
            Ok(tensor)
        })
}

#[cfg(feature = "local-onnx")]
fn run_caption_decoder(
    decoder: &mut runtime_onnx::OnnxSession,
    input_ids: &[u32],
    encoder_hidden_states: &runtime_onnx::OnnxF32Tensor,
) -> Result<Vec<f32>> {
    use runtime_onnx::{OnnxRunner, OnnxTensor, OnnxTensorValue};

    let metadata = decoder.metadata().map_err(runtime_onnx_error)?;
    let input_ids_name = find_input_name(&metadata, &["input_ids"], 0)?;
    let hidden_name = find_input_name(
        &metadata,
        &["encoder_hidden_states", "encoder_outputs", "hidden_states"],
        1,
    )?;
    let mut inputs = vec![
        runtime_onnx::OnnxNamedTensor {
            name: input_ids_name,
            tensor: OnnxTensorValue::I64(
                OnnxTensor::new(
                    vec![1, input_ids.len()],
                    input_ids.iter().map(|id| i64::from(*id)).collect(),
                )
                .map_err(runtime_onnx_error)?,
            ),
        },
        runtime_onnx::OnnxNamedTensor {
            name: hidden_name,
            tensor: OnnxTensorValue::F32(encoder_hidden_states.clone()),
        },
    ];
    if let Some(mask_name) = metadata
        .inputs
        .iter()
        .find(|input| input.name.contains("encoder_attention_mask"))
        .map(|input| input.name.clone())
    {
        let sequence_len = encoder_hidden_states.shape.get(1).copied().unwrap_or(1);
        inputs.push(runtime_onnx::OnnxNamedTensor {
            name: mask_name,
            tensor: OnnxTensorValue::I64(
                OnnxTensor::new(vec![1, sequence_len], vec![1_i64; sequence_len])
                    .map_err(runtime_onnx_error)?,
            ),
        });
    }
    let outputs = decoder.run(inputs).map_err(runtime_onnx_error)?;
    let logits = caption_logits_output(&outputs)?;
    final_token_logits(logits)
}

#[cfg(feature = "local-onnx")]
fn find_input_name(
    metadata: &runtime_onnx::OnnxSessionMetadata,
    needles: &[&str],
    fallback_index: usize,
) -> Result<String> {
    metadata
        .inputs
        .iter()
        .find(|input| needles.iter().any(|needle| input.name.contains(needle)))
        .or_else(|| metadata.inputs.get(fallback_index))
        .map(|input| input.name.clone())
        .ok_or_else(|| {
            DetectError::InvalidArgument(format!(
                "ONNX caption decoder is missing input #{fallback_index}"
            ))
        })
}

#[cfg(any(feature = "local-onnx", test))]
fn caption_logits_output(
    outputs: &[runtime_onnx::OnnxNamedTensor],
) -> Result<&runtime_onnx::OnnxF32Tensor> {
    runtime_onnx::f32_output_by_preferred_name_or_index(outputs, &["logits"], 0)
        .map_err(runtime_onnx_error)
}

#[cfg(any(feature = "local-onnx", test))]
fn validate_encoder_hidden_states(tensor: &runtime_onnx::OnnxF32Tensor) -> Result<()> {
    match tensor.shape.as_slice() {
        [batch, sequence, hidden] if *batch > 0 && *sequence > 0 && *hidden > 0 => Ok(()),
        shape => Err(DetectError::InvalidArgument(format!(
            "ONNX caption encoder hidden states must have shape `[batch, sequence, hidden]`, got `{shape:?}`"
        ))),
    }
}

#[cfg(any(feature = "local-onnx", test))]
fn final_token_logits(tensor: &runtime_onnx::OnnxF32Tensor) -> Result<Vec<f32>> {
    let vocab = match tensor.shape.as_slice() {
        [sequence, vocab] if *sequence > 0 && *vocab > 0 => *vocab,
        [batch, sequence, vocab] if *batch > 0 && *sequence > 0 && *vocab > 0 => *vocab,
        shape => {
            return Err(DetectError::InvalidArgument(format!(
                "ONNX caption logits must have shape `[sequence, vocab]` or `[batch, sequence, vocab]`, got `{shape:?}`"
            )))
        }
    };
    if tensor.values.len() < vocab {
        return Err(DetectError::InvalidArgument(
            "ONNX caption logits values do not match shape".to_string(),
        ));
    }
    Ok(tensor.values[tensor.values.len() - vocab..].to_vec())
}

#[cfg(any(feature = "local-onnx", test))]
fn validate_decoded_caption_text(text: &str) -> Result<()> {
    if text.trim().is_empty() {
        return Err(DetectError::InvalidArgument(
            "ONNX image captioning decoded an empty caption".to_string(),
        ));
    }
    Ok(())
}

#[cfg(any(feature = "local-onnx", test))]
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

fn read_json(path: &Path) -> Result<serde_json::Value> {
    let data = std::fs::read(path)?;
    serde_json::from_slice(&data).map_err(|err| {
        DetectError::Source(format!("failed to parse JSON `{}`: {err}", path.display()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_runtime::{ModelBundle, ModelBundleFile, ModelBundleManifest};

    fn test_bundle(files: &[(&str, &str, &str)]) -> (tempfile::TempDir, ModelBundle) {
        let temp = tempfile::tempdir().unwrap();
        let mut manifest_files = BTreeMap::new();
        for (remote_path, local_path, contents) in files {
            let path = temp.path().join(local_path);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(&path, contents).unwrap();
            manifest_files.insert(
                (*remote_path).to_string(),
                ModelBundleFile {
                    remote_path: (*remote_path).to_string(),
                    local_path: (*local_path).to_string(),
                    size_bytes: contents.len() as u64,
                },
            );
        }
        let bundle = ModelBundle {
            root: temp.path().to_path_buf(),
            manifest: ModelBundleManifest {
                schema_version: 1,
                name: "caption-test".to_string(),
                repo_id: "owner/model".to_string(),
                revision: "main".to_string(),
                task: ModelTask::Custom("image_captioning".to_string()),
                files: manifest_files,
            },
        };
        (temp, bundle)
    }

    #[test]
    fn captioning_catalog_reports_defaults() {
        let catalog = image_captioning_catalog(None);
        assert!(catalog
            .iter()
            .any(|entry| entry.id == "vit-gpt2-image-captioning"));
        assert!(catalog
            .iter()
            .any(|entry| entry.id == "blip-image-captioning-base"));
        assert_eq!(
            parse_task("caption"),
            Some(ImageCaptioningTask::ImageCaptioning)
        );
    }

    #[test]
    fn caption_can_carry_score_and_attributes() {
        let caption = ImageCaption::new("a cube")
            .score(0.8)
            .attribute("source", "fixture");
        assert_eq!(caption.score, Some(0.8));
        assert_eq!(
            caption.attributes.get("source").map(String::as_str),
            Some("fixture")
        );
    }

    #[test]
    fn greedy_decode_stops_on_eos() {
        let mut calls = 0;
        let generated = greedy_decode_token_ids(1, 3, 8, |_| {
            calls += 1;
            Ok(match calls {
                1 => vec![0.0, 0.0, 8.0, 0.0],
                _ => vec![0.0, 0.0, 0.0, 9.0],
            })
        })
        .unwrap();
        assert_eq!(generated, vec![2]);
        assert_eq!(calls, 2);
    }

    #[test]
    fn greedy_decode_excludes_eos_token_from_payload() {
        let generated =
            greedy_decode_token_ids(1, 2, 8, |_| Ok(vec![0.0, 0.0, 10.0, 1.0])).unwrap();
        assert!(generated.is_empty());
    }

    #[test]
    fn greedy_decode_honors_token_limit() {
        let generated = greedy_decode_token_ids(1, 9, 3, |_| Ok(vec![0.0, 0.0, 1.0])).unwrap();
        assert_eq!(generated, vec![2, 2, 2]);
    }

    #[test]
    fn captioning_options_validate_required_token_ids() {
        let missing = OnnxImageCaptioningOptions::default();
        assert!(validate_captioning_options(&missing).is_err());

        let missing_eos = OnnxImageCaptioningOptions {
            decoder_start_token_id: Some(1),
            ..OnnxImageCaptioningOptions::default()
        };
        let err = validate_captioning_options(&missing_eos).unwrap_err();
        assert!(err.to_string().contains("eos_token_id"));

        let present = OnnxImageCaptioningOptions {
            decoder_start_token_id: Some(1),
            eos_token_id: Some(2),
            ..OnnxImageCaptioningOptions::default()
        };
        assert!(validate_captioning_options(&present).is_ok());
    }

    #[test]
    fn captioning_options_from_bundle_reads_config_and_preprocessing() {
        let (_temp, bundle) = test_bundle(&[
            (
                "config.json",
                "config.json",
                r#"{"decoder_start_token_id":1,"eos_token_id":2,"pad_token_id":0,"max_new_tokens":12}"#,
            ),
            (
                "preprocessor_config.json",
                "preprocessor_config.json",
                r#"{"size":{"width":128,"height":96},"rescale_factor":0.5}"#,
            ),
            ("tokenizer.json", "tokenizer.json", "{}"),
            (
                "onnx/encoder_model.onnx",
                "onnx/encoder_model.onnx",
                "encoder",
            ),
            (
                "onnx/decoder_model.onnx",
                "onnx/decoder_model.onnx",
                "decoder",
            ),
        ]);
        let options = captioning_options_from_bundle(&bundle).unwrap();
        assert_eq!(options.decoder_start_token_id, Some(1));
        assert_eq!(options.eos_token_id, Some(2));
        assert_eq!(options.pad_token_id, Some(0));
        assert_eq!(options.max_new_tokens, 12);
        assert_eq!(options.preprocessing.input_width, 128);
        assert_eq!(options.preprocessing.input_height, 96);
    }

    #[test]
    fn captioning_bundle_allows_missing_preprocessor_config() {
        let (_temp, bundle) = test_bundle(&[
            (
                "config.json",
                "config.json",
                r#"{"decoder_start_token_id":1,"eos_token_id":2}"#,
            ),
            ("tokenizer.json", "tokenizer.json", "{}"),
            (
                "onnx/encoder_model.onnx",
                "onnx/encoder_model.onnx",
                "encoder",
            ),
            (
                "onnx/decoder_model.onnx",
                "onnx/decoder_model.onnx",
                "decoder",
            ),
        ]);
        let options = captioning_options_from_bundle(&bundle).unwrap();
        assert_eq!(options.preprocessing.input_width, 224);
        assert_eq!(options.preprocessing.input_height, 224);
    }

    #[test]
    fn captioning_bundle_missing_tokenizer_reports_required_filename() {
        let (_temp, bundle) = test_bundle(&[
            (
                "config.json",
                "config.json",
                r#"{"decoder_start_token_id":1,"eos_token_id":2}"#,
            ),
            (
                "onnx/encoder_model.onnx",
                "onnx/encoder_model.onnx",
                "encoder",
            ),
            (
                "onnx/decoder_model.onnx",
                "onnx/decoder_model.onnx",
                "decoder",
            ),
        ]);
        let err = captioning_options_from_bundle(&bundle).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("caption-test"));
        assert!(message.contains("tokenizer.json"));
        assert!(message.contains("image_captioning"));
    }

    #[test]
    fn encoder_hidden_states_accepts_rank_three() {
        let tensor = runtime_onnx::OnnxF32Tensor::new(vec![1, 2, 3], vec![0.0; 6]).unwrap();
        validate_encoder_hidden_states(&tensor).unwrap();
    }

    #[test]
    fn final_token_logits_selects_last_position() {
        let tensor =
            runtime_onnx::OnnxF32Tensor::new(vec![1, 2, 3], vec![0.1, 0.2, 0.3, 1.0, 2.0, 3.0])
                .unwrap();
        assert_eq!(final_token_logits(&tensor).unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn final_token_logits_supports_implicit_batch_shape() {
        let tensor =
            runtime_onnx::OnnxF32Tensor::new(vec![2, 3], vec![0.1, 0.2, 0.3, 1.0, 2.0, 3.0])
                .unwrap();
        assert_eq!(final_token_logits(&tensor).unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn caption_logits_output_prefers_logits_name() {
        let outputs = vec![
            runtime_onnx::OnnxNamedTensor {
                name: "past_key_values".to_string(),
                tensor: runtime_onnx::OnnxTensorValue::F32(
                    runtime_onnx::OnnxF32Tensor::new(vec![1, 1, 2], vec![1.0, 2.0]).unwrap(),
                ),
            },
            runtime_onnx::OnnxNamedTensor {
                name: "logits".to_string(),
                tensor: runtime_onnx::OnnxTensorValue::F32(
                    runtime_onnx::OnnxF32Tensor::new(vec![1, 1, 3], vec![3.0, 4.0, 5.0]).unwrap(),
                ),
            },
        ];
        assert_eq!(
            final_token_logits(caption_logits_output(&outputs).unwrap()).unwrap(),
            vec![3.0, 4.0, 5.0]
        );
    }

    #[test]
    fn final_token_logits_rejects_unsupported_shape() {
        let tensor = runtime_onnx::OnnxF32Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let err = final_token_logits(&tensor).unwrap_err();
        assert!(err.to_string().contains("[3]"));
    }

    #[test]
    fn encoder_hidden_states_rejects_unsupported_shape() {
        let tensor = runtime_onnx::OnnxF32Tensor::new(vec![1, 3], vec![1.0, 2.0, 3.0]).unwrap();
        let err = validate_encoder_hidden_states(&tensor).unwrap_err();
        assert!(err.to_string().contains("hidden states"));
    }

    #[test]
    fn empty_decoded_caption_is_rejected() {
        let err = validate_decoded_caption_text("  ").unwrap_err();
        assert!(err.to_string().contains("empty caption"));
    }
}
