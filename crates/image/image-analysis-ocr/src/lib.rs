#![doc = include_str!("../README.md")]

pub mod surface;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr;

use image_analysis_core::{compact_image, ImageView};
#[cfg(feature = "local-onnx")]
use image_analysis_processing::preprocess_image_for_model;
use image_analysis_processing::{
    image_model_preprocessing_from_config, validate_image_model_preprocessing,
    ImageModelPreprocessing,
};
use model_runtime::{HuggingFaceModelSpec, ModelRuntimeBackend, ModelTask};
use serde::{Deserialize, Serialize};
use text_core::{
    TextAnnotationSpan, TextDocumentContract, TextProvenance, TextSourceRef, TextSpan,
};
use video_analysis_core::{BoundingBox, DetectError, FramePosition, Result, VideoFrame};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Variants describing OCR model presets.
pub enum OcrPreset {
    #[default]
    /// Printed document text recognition with TrOCR.
    TrOcrBasePrinted,
    /// Handwritten text recognition with TrOCR.
    TrOcrBaseHandwritten,
    /// Printed document text recognition with Xenova TrOCR ONNX encoder/decoder files.
    TrOcrBasePrintedOnnx,
    /// Handwritten text recognition with Xenova TrOCR ONNX encoder/decoder files.
    TrOcrBaseHandwrittenOnnx,
    /// OCR-focused document understanding with Donut on CORD receipts.
    DonutBaseCordV2,
    /// Document transcription model for academic/scientific pages.
    NougatBase,
}

impl OcrPreset {
    /// Constant for all.
    pub const ALL: &'static [Self] = &[
        Self::TrOcrBasePrinted,
        Self::TrOcrBaseHandwritten,
        Self::TrOcrBasePrintedOnnx,
        Self::TrOcrBaseHandwrittenOnnx,
        Self::DonutBaseCordV2,
        Self::NougatBase,
    ];

    /// Borrows this value as a str.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TrOcrBasePrinted => "trocr-base-printed",
            Self::TrOcrBaseHandwritten => "trocr-base-handwritten",
            Self::TrOcrBasePrintedOnnx => "trocr-base-printed-onnx",
            Self::TrOcrBaseHandwrittenOnnx => "trocr-base-handwritten-onnx",
            Self::DonutBaseCordV2 => "donut-base-cord-v2",
            Self::NougatBase => "nougat-base",
        }
    }

    /// Returns repo identifier.
    pub fn repo_id(self) -> &'static str {
        match self {
            Self::TrOcrBasePrinted => "microsoft/trocr-base-printed",
            Self::TrOcrBaseHandwritten => "microsoft/trocr-base-handwritten",
            Self::TrOcrBasePrintedOnnx => "Xenova/trocr-base-printed",
            Self::TrOcrBaseHandwrittenOnnx => "Xenova/trocr-base-handwritten",
            Self::DonutBaseCordV2 => "naver-clova-ix/donut-base-finetuned-cord-v2",
            Self::NougatBase => "facebook/nougat-base",
        }
    }

    /// Returns model spec.
    pub fn model_spec(self) -> HuggingFaceModelSpec {
        let base = HuggingFaceModelSpec::new(
            self.repo_id(),
            ModelTask::Custom("optical_character_recognition".to_string()),
        )
        .name(self.as_str())
        .file("config.json")
        .file("preprocessor_config.json");

        match self {
            Self::TrOcrBasePrinted | Self::TrOcrBaseHandwritten => base
                .first_available_file(["model.safetensors", "pytorch_model.bin"])
                .file("tokenizer.json")
                .file("vocab.json")
                .file("merges.txt"),
            Self::TrOcrBasePrintedOnnx | Self::TrOcrBaseHandwrittenOnnx => base
                .file("tokenizer.json")
                .optional_file("generation_config.json")
                .optional_file("tokenizer_config.json")
                .optional_file("vocab.json")
                .optional_file("merges.txt")
                .first_available_file([
                    "onnx/encoder_model_quantized.onnx",
                    "onnx/encoder_model.onnx",
                    "onnx/encoder_model_fp16.onnx",
                ])
                .first_available_file([
                    "onnx/decoder_model_quantized.onnx",
                    "onnx/decoder_model.onnx",
                    "onnx/decoder_model_fp16.onnx",
                ]),
            Self::DonutBaseCordV2 => base
                .first_available_file(["model.safetensors", "pytorch_model.bin"])
                .file("tokenizer.json")
                .optional_file("sentencepiece.bpe.model"),
            Self::NougatBase => base
                .first_available_file(["model.safetensors", "pytorch_model.bin"])
                .file("tokenizer.json")
                .optional_file("tokenizer_config.json")
                .optional_file("special_tokens_map.json"),
        }
    }
}

impl FromStr for OcrPreset {
    type Err = DetectError;

    fn from_str(input: &str) -> Result<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|preset| preset.as_str() == input)
            .ok_or_else(|| {
                DetectError::InvalidArgument(format!(
                    "unknown OCR preset `{input}`; expected one of {}",
                    Self::ALL
                        .iter()
                        .map(|preset| preset.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })
    }
}

/// Returns default OCR model spec.
pub fn default_ocr_model_spec() -> HuggingFaceModelSpec {
    OcrPreset::default().model_spec()
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Variants describing OCR technique families.
pub enum OcrTechnique {
    /// A Hugging Face model-backed recognizer.
    HuggingFaceModel(HuggingFaceModelSpec),
    /// A local ONNX encoder-decoder recognizer.
    OnnxModel(HuggingFaceModelSpec),
    /// A local or remote command-style adapter such as Tesseract, EasyOCR, or PaddleOCR.
    ExternalCommand(ExternalOcrCommandSpec),
    /// A deterministic recognizer, usually for tests or simple threshold/layout passes.
    Heuristic,
    /// A custom integration.
    Custom(String),
}

impl OcrTechnique {
    /// Returns a compact technique kind.
    pub fn kind(&self) -> &str {
        match self {
            Self::HuggingFaceModel(_) => "huggingface_model",
            Self::OnnxModel(_) => "onnx_model",
            Self::ExternalCommand(_) => "external_command",
            Self::Heuristic => "heuristic",
            Self::Custom(value) => value.as_str(),
        }
    }

    /// Returns the model runtime backend usually associated with this technique.
    pub fn runtime_backend(&self) -> ModelRuntimeBackend {
        match self {
            Self::HuggingFaceModel(_) => ModelRuntimeBackend::Candle,
            Self::OnnxModel(_) => ModelRuntimeBackend::Onnx,
            Self::ExternalCommand(_) => ModelRuntimeBackend::External,
            Self::Heuristic => ModelRuntimeBackend::Heuristic,
            Self::Custom(value) => ModelRuntimeBackend::Custom(value.clone()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// OCR model metadata for UI, CLI, and server discovery.
pub struct OcrCatalogEntry {
    /// Stable local entry id.
    pub id: String,
    /// Upstream model id or adapter id.
    pub model_id: String,
    /// Preferred runtime.
    pub runtime: ModelRuntimeBackend,
    /// Whether this runtime is available in the current build.
    pub supported: bool,
    /// Optional fallback entry id.
    pub fallback: Option<String>,
    /// Human-readable note for unsupported or setup-required paths.
    pub note: Option<String>,
}

/// Returns OCR catalog entries without downloading or running models.
pub fn ocr_catalog() -> Vec<OcrCatalogEntry> {
    OcrPreset::ALL
        .iter()
        .copied()
        .map(|preset| {
            let spec = preset.model_spec();
            let runtime = match preset {
                OcrPreset::TrOcrBasePrintedOnnx | OcrPreset::TrOcrBaseHandwrittenOnnx => {
                    ModelRuntimeBackend::Onnx
                }
                _ => ModelRuntimeBackend::Candle,
            };
            let supported = matches!(runtime, ModelRuntimeBackend::Onnx) && cfg!(feature = "local-onnx");
            OcrCatalogEntry {
                id: preset.as_str().to_string(),
                model_id: spec.repo_id_value().unwrap_or(preset.as_str()).to_string(),
                runtime,
                supported,
                fallback: if supported {
                    None
                } else {
                    Some("imported-ocr-document".to_string())
                },
                note: match preset {
                    OcrPreset::TrOcrBasePrintedOnnx | OcrPreset::TrOcrBaseHandwrittenOnnx => {
                        Some("Server-only ONNX TrOCR execution requires the `local-onnx` feature and an explicit local model request.".to_string())
                    }
                    _ => Some("Catalog metadata only in this slice; default builds do not execute OCR models.".to_string()),
                },
            }
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Data type for external OCR command integration metadata.
pub struct ExternalOcrCommandSpec {
    /// Human-readable name.
    pub name: String,
    /// Executable or service command.
    pub command: String,
    /// Default command arguments.
    pub args: Vec<String>,
    /// Attributes for adapter-specific metadata.
    pub attributes: BTreeMap<String, String>,
}

impl ExternalOcrCommandSpec {
    /// Creates a new value.
    pub fn new(name: impl Into<String>, command: impl Into<String>) -> Result<Self> {
        let name = name.into();
        let command = command.into();
        if name.trim().is_empty() || command.trim().is_empty() {
            return Err(DetectError::InvalidArgument(
                "external OCR command name and command are required".to_string(),
            ));
        }
        Ok(Self {
            name,
            command,
            args: Vec::new(),
            attributes: BTreeMap::new(),
        })
    }

    /// Returns arg.
    pub fn arg(mut self, value: impl Into<String>) -> Self {
        self.args.push(value.into());
        self
    }

    /// Returns attribute.
    pub fn attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Data type for OCR request options.
pub struct OcrRequest {
    /// Optional technique override.
    pub technique: Option<OcrTechnique>,
    /// Preferred languages, using BCP-47 or engine-specific short codes.
    pub languages: Vec<String>,
    /// Whether recognizers should preserve line/block/page layout when possible.
    pub preserve_layout: bool,
    /// Whether recognizers should include token-level spans when possible.
    pub include_tokens: bool,
    /// Minimum confidence accepted in post-processing.
    pub min_confidence: Option<u8>,
    /// Attributes for adapter-specific options.
    pub attributes: BTreeMap<String, String>,
}

impl Default for OcrRequest {
    fn default() -> Self {
        Self {
            technique: Some(OcrTechnique::HuggingFaceModel(default_ocr_model_spec())),
            languages: Vec::new(),
            preserve_layout: true,
            include_tokens: true,
            min_confidence: None,
            attributes: BTreeMap::new(),
        }
    }
}

impl OcrRequest {
    /// Creates a new value.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns technique.
    pub fn technique(mut self, value: OcrTechnique) -> Self {
        self.technique = Some(value);
        self
    }

    /// Returns model preset.
    pub fn model_preset(mut self, preset: OcrPreset) -> Self {
        self.technique = Some(match preset {
            OcrPreset::TrOcrBasePrintedOnnx | OcrPreset::TrOcrBaseHandwrittenOnnx => {
                OcrTechnique::OnnxModel(preset.model_spec())
            }
            _ => OcrTechnique::HuggingFaceModel(preset.model_spec()),
        });
        self
    }

    /// Returns languages.
    pub fn languages(mut self, values: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.languages = values.into_iter().map(Into::into).collect();
        self
    }

    /// Returns preserve layout.
    pub fn preserve_layout(mut self, value: bool) -> Self {
        self.preserve_layout = value;
        self
    }

    /// Returns include tokens.
    pub fn include_tokens(mut self, value: bool) -> Self {
        self.include_tokens = value;
        self
    }

    /// Returns min confidence.
    pub fn min_confidence(mut self, value: u8) -> Self {
        self.min_confidence = Some(value.min(100));
        self
    }

    /// Returns attribute.
    pub fn attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
/// Data type for OCR confidence as a 0-100 score.
pub struct OcrConfidence(u8);

impl OcrConfidence {
    /// Creates a new value.
    pub fn new(value: u8) -> Self {
        Self(value.min(100))
    }

    /// Returns raw value.
    pub fn value(self) -> u8 {
        self.0
    }
}

impl From<u8> for OcrConfidence {
    fn from(value: u8) -> Self {
        Self::new(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// Data type for OCR text token.
pub struct OcrToken {
    /// Text content.
    pub text: String,
    /// Optional region in image coordinates.
    pub region: Option<BoundingBox>,
    /// Optional confidence.
    pub confidence: Option<OcrConfidence>,
    /// Attributes for adapter-specific metadata.
    pub attributes: BTreeMap<String, String>,
}

impl OcrToken {
    /// Creates a new value.
    pub fn new(text: impl Into<String>) -> Result<Self> {
        let text = text.into();
        if text.trim().is_empty() {
            return Err(DetectError::InvalidArgument(
                "OCR token text is required".to_string(),
            ));
        }
        Ok(Self {
            text,
            region: None,
            confidence: None,
            attributes: BTreeMap::new(),
        })
    }

    /// Returns region.
    pub fn region(mut self, value: BoundingBox) -> Self {
        self.region = Some(value);
        self
    }

    /// Returns confidence.
    pub fn confidence(mut self, value: impl Into<OcrConfidence>) -> Self {
        self.confidence = Some(value.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// Data type for OCR text line.
pub struct OcrTextLine {
    /// Text content.
    pub text: String,
    /// Optional region in image coordinates.
    pub region: Option<BoundingBox>,
    /// Optional confidence.
    pub confidence: Option<OcrConfidence>,
    /// Tokens in this line.
    pub tokens: Vec<OcrToken>,
    /// Attributes for adapter-specific metadata.
    pub attributes: BTreeMap<String, String>,
}

impl OcrTextLine {
    /// Creates a new value.
    pub fn new(text: impl Into<String>) -> Result<Self> {
        let text = text.into();
        if text.trim().is_empty() {
            return Err(DetectError::InvalidArgument(
                "OCR line text is required".to_string(),
            ));
        }
        Ok(Self {
            text,
            region: None,
            confidence: None,
            tokens: Vec::new(),
            attributes: BTreeMap::new(),
        })
    }

    /// Returns region.
    pub fn region(mut self, value: BoundingBox) -> Self {
        self.region = Some(value);
        self
    }

    /// Returns confidence.
    pub fn confidence(mut self, value: impl Into<OcrConfidence>) -> Self {
        self.confidence = Some(value.into());
        self
    }

    /// Returns token.
    pub fn token(mut self, value: OcrToken) -> Self {
        self.tokens.push(value);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Variants describing OCR block role.
pub enum OcrBlockKind {
    /// Paragraph-like flowing text.
    Paragraph,
    /// Heading or title text.
    Heading,
    /// Table-like structured text.
    Table,
    /// Form/key-value text.
    Form,
    /// Caption text.
    Caption,
    /// Custom block role.
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// Data type for OCR text block.
pub struct OcrTextBlock {
    /// Block role.
    pub kind: OcrBlockKind,
    /// Text content.
    pub text: String,
    /// Optional region in image coordinates.
    pub region: Option<BoundingBox>,
    /// Optional confidence.
    pub confidence: Option<OcrConfidence>,
    /// Lines in this block.
    pub lines: Vec<OcrTextLine>,
    /// Attributes for adapter-specific metadata.
    pub attributes: BTreeMap<String, String>,
}

impl OcrTextBlock {
    /// Creates a new paragraph block.
    pub fn paragraph(text: impl Into<String>) -> Result<Self> {
        Self::new(OcrBlockKind::Paragraph, text)
    }

    /// Creates a new value.
    pub fn new(kind: OcrBlockKind, text: impl Into<String>) -> Result<Self> {
        let text = text.into();
        if text.trim().is_empty() {
            return Err(DetectError::InvalidArgument(
                "OCR block text is required".to_string(),
            ));
        }
        Ok(Self {
            kind,
            text,
            region: None,
            confidence: None,
            lines: Vec::new(),
            attributes: BTreeMap::new(),
        })
    }

    /// Returns region.
    pub fn region(mut self, value: BoundingBox) -> Self {
        self.region = Some(value);
        self
    }

    /// Returns confidence.
    pub fn confidence(mut self, value: impl Into<OcrConfidence>) -> Self {
        self.confidence = Some(value.into());
        self
    }

    /// Returns line.
    pub fn line(mut self, value: OcrTextLine) -> Self {
        self.lines.push(value);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// Data type for rich OCR output from an image.
pub struct OcrDocument {
    /// Full extracted text, usually reading-order normalized.
    pub text: String,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Optional detected language.
    pub language: Option<String>,
    /// Optional confidence.
    pub confidence: Option<OcrConfidence>,
    /// Structured text blocks.
    pub blocks: Vec<OcrTextBlock>,
    /// Attributes for adapter-specific metadata.
    pub attributes: BTreeMap<String, String>,
}

impl OcrDocument {
    /// Creates a new value.
    pub fn new(text: impl Into<String>, width: u32, height: u32) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(DetectError::InvalidDimensions { width, height });
        }
        let text = text.into();
        Ok(Self {
            text,
            width,
            height,
            language: None,
            confidence: None,
            blocks: Vec::new(),
            attributes: BTreeMap::new(),
        })
    }

    /// Builds an empty result from an image.
    pub fn empty_for_image(image: &ImageView<'_>) -> Self {
        Self {
            text: String::new(),
            width: image.width,
            height: image.height,
            language: None,
            confidence: None,
            blocks: Vec::new(),
            attributes: BTreeMap::new(),
        }
    }

    /// Returns language.
    pub fn language(mut self, value: impl Into<String>) -> Self {
        self.language = Some(value.into());
        self
    }

    /// Returns confidence.
    pub fn confidence(mut self, value: impl Into<OcrConfidence>) -> Self {
        self.confidence = Some(value.into());
        self
    }

    /// Returns block.
    pub fn block(mut self, value: OcrTextBlock) -> Self {
        self.blocks.push(value);
        self
    }

    /// Returns attribute.
    pub fn attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// Converts this OCR result into the canonical text-core document contract.
    pub fn to_text_document_contract(
        &self,
        options: OcrTextContractOptions,
    ) -> OcrTextDocumentConversion {
        convert_ocr_document_to_text_contract(self, options)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Options for converting OCR output into a text-core document contract.
pub struct OcrTextContractOptions {
    /// Stable document id.
    pub id: String,
    /// Optional source id for the originating image.
    pub source_id: Option<String>,
    /// Optional source URI for the originating image.
    pub source_uri: Option<String>,
    /// Provenance operation name.
    pub operation: String,
    /// Optional model id.
    pub model_id: Option<String>,
    /// Runtime label.
    pub runtime: String,
    /// Layout mode label.
    pub layout_mode: String,
    /// Whether to include block spans when they align with document text.
    pub include_block_spans: bool,
    /// Whether to include line spans when they align with document text.
    pub include_line_spans: bool,
    /// Whether to include token spans when they align with document text.
    pub include_token_spans: bool,
    /// Extra attributes to copy into the text document.
    pub attributes: BTreeMap<String, String>,
}

impl Default for OcrTextContractOptions {
    fn default() -> Self {
        Self {
            id: "ocr-document".to_string(),
            source_id: None,
            source_uri: None,
            operation: "image.ocr.toTextDocument".to_string(),
            model_id: None,
            runtime: "ocr".to_string(),
            layout_mode: "single_sequence".to_string(),
            include_block_spans: true,
            include_line_spans: true,
            include_token_spans: true,
            attributes: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Result of converting OCR output into a text-core document contract.
pub struct OcrTextDocumentConversion {
    /// Converted text-core document contract.
    pub document: TextDocumentContract,
    /// Count of OCR blocks aligned to text spans.
    pub aligned_blocks: usize,
    /// Count of OCR lines aligned to text spans.
    pub aligned_lines: usize,
    /// Count of OCR tokens aligned to text spans.
    pub aligned_tokens: usize,
    /// Count of OCR blocks that could not be aligned.
    pub unaligned_blocks: usize,
    /// Count of OCR lines that could not be aligned.
    pub unaligned_lines: usize,
    /// Count of OCR tokens that could not be aligned.
    pub unaligned_tokens: usize,
}

fn convert_ocr_document_to_text_contract(
    ocr: &OcrDocument,
    options: OcrTextContractOptions,
) -> OcrTextDocumentConversion {
    let block_count = ocr.blocks.len();
    let line_count = ocr
        .blocks
        .iter()
        .map(|block| block.lines.len())
        .sum::<usize>();
    let token_count = ocr
        .blocks
        .iter()
        .flat_map(|block| &block.lines)
        .map(|line| line.tokens.len())
        .sum::<usize>();

    let mut document = TextDocumentContract::new(options.id, ocr.text.clone());
    document.language = ocr.language.clone();
    document.source = Some(TextSourceRef {
        source_id: options.source_id,
        source_kind: Some("ocr_image".to_string()),
        uri: options.source_uri,
        media_timestamp: None,
        duration_seconds: None,
    });
    document.provenance.push(TextProvenance {
        crate_name: Some(env!("CARGO_PKG_NAME").to_string()),
        operation: Some(options.operation.clone()),
        model_id: options.model_id,
        runtime: Some(options.runtime.clone()),
        confidence: ocr
            .confidence
            .map(|confidence| confidence.value() as f32 / 100.0),
    });
    document.attributes = ocr.attributes.clone();
    document
        .attributes
        .insert("ocr.width".to_string(), ocr.width.to_string());
    document
        .attributes
        .insert("ocr.height".to_string(), ocr.height.to_string());
    document
        .attributes
        .insert("ocr.blockCount".to_string(), block_count.to_string());
    document
        .attributes
        .insert("ocr.lineCount".to_string(), line_count.to_string());
    document
        .attributes
        .insert("ocr.tokenCount".to_string(), token_count.to_string());
    document
        .attributes
        .insert("ocr.layoutMode".to_string(), options.layout_mode.clone());
    if let Some(confidence) = ocr.confidence {
        document
            .attributes
            .insert("ocr.confidence".to_string(), confidence.value().to_string());
    }
    document.attributes.extend(options.attributes);

    let mut block_alignment = SpanAlignment::new(&ocr.text);
    let mut line_alignment = SpanAlignment::new(&ocr.text);
    let mut token_alignment = SpanAlignment::new(&ocr.text);
    let mut conversion = OcrTextDocumentConversion {
        document,
        aligned_blocks: 0,
        aligned_lines: 0,
        aligned_tokens: 0,
        unaligned_blocks: 0,
        unaligned_lines: 0,
        unaligned_tokens: 0,
    };

    for (block_index, block) in ocr.blocks.iter().enumerate() {
        if options.include_block_spans {
            if let Some(span) = block_alignment.find_span(&block.text) {
                conversion.document.annotations.push(TextAnnotationSpan {
                    span,
                    token_start: None,
                    token_end: None,
                    source_segment_id: Some(format!("ocr:block:{block_index}")),
                });
                conversion.aligned_blocks += 1;
            } else {
                conversion.unaligned_blocks += 1;
            }
        }

        for (line_index, line) in block.lines.iter().enumerate() {
            if options.include_line_spans {
                if let Some(span) = line_alignment.find_span(&line.text) {
                    conversion.document.annotations.push(TextAnnotationSpan {
                        span,
                        token_start: None,
                        token_end: None,
                        source_segment_id: Some(format!(
                            "ocr:block:{block_index}:line:{line_index}"
                        )),
                    });
                    conversion.aligned_lines += 1;
                } else {
                    conversion.unaligned_lines += 1;
                }
            }

            for (token_index, token) in line.tokens.iter().enumerate() {
                if options.include_token_spans {
                    if let Some(span) = token_alignment.find_span(&token.text) {
                        conversion.document.annotations.push(TextAnnotationSpan {
                            span,
                            token_start: Some(token_index),
                            token_end: Some(token_index + 1),
                            source_segment_id: Some(format!(
                                "ocr:block:{block_index}:line:{line_index}:token:{token_index}"
                            )),
                        });
                        conversion.aligned_tokens += 1;
                    } else {
                        conversion.unaligned_tokens += 1;
                    }
                }
            }
        }
    }

    conversion
}

struct SpanAlignment<'a> {
    text: &'a str,
    cursor: usize,
}

impl<'a> SpanAlignment<'a> {
    fn new(text: &'a str) -> Self {
        Self { text, cursor: 0 }
    }

    fn find_span(&mut self, needle: &str) -> Option<TextSpan> {
        let needle = needle.trim();
        if needle.is_empty() {
            return None;
        }
        let relative = self.text.get(self.cursor..)?.find(needle);
        let byte_start = relative
            .map(|offset| self.cursor + offset)
            .or_else(|| self.text.find(needle))?;
        let byte_end = byte_start + needle.len();
        self.cursor = byte_end;
        Some(text_span(self.text, byte_start, byte_end))
    }
}

fn text_span(text: &str, byte_start: usize, byte_end: usize) -> TextSpan {
    TextSpan {
        byte_start,
        byte_end,
        char_start: text[..byte_start].chars().count(),
        char_end: text[..byte_end].chars().count(),
    }
}

/// Trait for OCR backend implementations.
pub trait OcrBackend {
    /// Returns recognize image.
    fn recognize_image(
        &mut self,
        image: &ImageView<'_>,
        request: &OcrRequest,
    ) -> Result<OcrDocument>;
}

/// Trait for model-backed OCR backend implementations.
pub trait ModelBackedOcrBackend: OcrBackend {
    /// Returns model spec.
    fn model_spec(&self) -> HuggingFaceModelSpec;
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Options for resolving a local OCR model bundle.
pub struct OcrLocalModelOptions {
    /// OCR preset to resolve.
    pub preset: OcrPreset,
    /// Root directory containing model-runtime bundles.
    pub bundle_root: PathBuf,
    /// Whether missing bundles may be downloaded.
    pub auto_download: bool,
    /// Whether downloads should report progress.
    pub download_progress: bool,
    /// Optional Hugging Face token.
    pub hf_token: Option<String>,
    /// Optional Hugging Face cache directory.
    pub cache_dir: Option<PathBuf>,
    /// Maximum download retries.
    pub max_retries: usize,
}

impl Default for OcrLocalModelOptions {
    fn default() -> Self {
        Self {
            preset: OcrPreset::TrOcrBasePrintedOnnx,
            bundle_root: PathBuf::from(".model-runtime"),
            auto_download: false,
            download_progress: false,
            hf_token: None,
            cache_dir: None,
            max_retries: 1,
        }
    }
}

impl OcrLocalModelOptions {
    /// Converts to model-runtime bundle resolution options.
    pub fn bundle_resolve_options(&self) -> model_runtime::ModelBundleResolveOptions {
        model_runtime::ModelBundleResolveOptions {
            bundle_root: self.bundle_root.clone(),
            auto_download: self.auto_download,
            download_progress: self.download_progress,
            hf_token: self.hf_token.clone(),
            cache_dir: self.cache_dir.clone(),
            max_retries: self.max_retries,
            overwrite: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Options for ONNX TrOCR execution.
pub struct OnnxTrOcrOptions {
    /// Image preprocessing options.
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

impl Default for OnnxTrOcrOptions {
    fn default() -> Self {
        Self {
            preprocessing: ImageModelPreprocessing {
                input_width: 384,
                input_height: 384,
                rescale_factor: 1.0 / 255.0,
                mean: [0.5, 0.5, 0.5],
                std: [0.5, 0.5, 0.5],
                channel_order: image_analysis_processing::ChannelOrder::Rgb,
            },
            max_new_tokens: 128,
            decoder_start_token_id: None,
            eos_token_id: None,
            pad_token_id: None,
        }
    }
}

#[derive(Debug)]
/// ONNX TrOCR OCR backend.
pub struct OnnxTrOcrBackend {
    spec: HuggingFaceModelSpec,
    options: OnnxTrOcrOptions,
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

impl OnnxTrOcrBackend {
    /// Resolves and builds the backend from local model options.
    pub fn from_local_model(options: OcrLocalModelOptions) -> Result<Self> {
        #[cfg(not(feature = "local-onnx"))]
        {
            let _ = options;
            Err(local_onnx_feature_error())
        }
        #[cfg(feature = "local-onnx")]
        {
            let spec = options.preset.model_spec();
            let bundle_options = options.bundle_resolve_options();
            let bundle = model_runtime::resolve_or_download_bundle(&spec, &bundle_options)
                .map_err(model_runtime_error)?;
            Self::from_bundle(bundle)
        }
    }

    /// Builds this backend from a model-runtime bundle.
    pub fn from_bundle(bundle: model_runtime::ModelBundle) -> Result<Self> {
        let info = validate_onnx_trocr_bundle(&bundle)?;
        #[cfg(not(feature = "local-onnx"))]
        let options = onnx_trocr_options_from_bundle(&bundle)?;
        #[cfg(feature = "local-onnx")]
        let mut options = onnx_trocr_options_from_bundle(&bundle)?;

        #[cfg(feature = "local-onnx")]
        {
            let tokenizer =
                tokenizers::Tokenizer::from_file(&info.tokenizer_path).map_err(|err| {
                    DetectError::Source(format!(
                        "failed to load tokenizer `{}`: {err}",
                        info.tokenizer_path.display()
                    ))
                })?;
            resolve_trocr_token_ids(&mut options, &tokenizer, &info)?;
            validate_trocr_options(&options)?;
            return Ok(Self {
                spec: bundle_spec(&bundle),
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
            spec: bundle_spec(&bundle),
            options,
            encoder_path: info.encoder_path,
            decoder_path: info.decoder_path,
            tokenizer_path: info.tokenizer_path,
        })
    }

    /// Returns ONNX TrOCR options.
    pub fn options(&self) -> &OnnxTrOcrOptions {
        &self.options
    }
}

impl OcrBackend for OnnxTrOcrBackend {
    fn recognize_image(
        &mut self,
        image: &ImageView<'_>,
        request: &OcrRequest,
    ) -> Result<OcrDocument> {
        #[cfg(feature = "local-onnx")]
        {
            validate_trocr_options(&self.options)?;
            let input = preprocess_image_for_model(image, &self.options.preprocessing)?;
            let encoder_hidden_states = run_trocr_encoder(&mut self.encoder, &input)?;
            let decoder_start_token_id = self.options.decoder_start_token_id.ok_or_else(|| {
                DetectError::InvalidArgument(
                    "ONNX TrOCR requires decoder_start_token_id".to_string(),
                )
            })?;
            let eos_token_id = self.options.eos_token_id.ok_or_else(|| {
                DetectError::InvalidArgument("ONNX TrOCR requires eos_token_id".to_string())
            })?;
            let generated = greedy_decode_trocr_token_ids(
                decoder_start_token_id,
                eos_token_id,
                self.options.max_new_tokens,
                |input_ids| run_trocr_decoder(&mut self.decoder, input_ids, &encoder_hidden_states),
            )?;
            let text = self
                .tokenizer
                .decode(&generated, true)
                .map_err(|err| DetectError::Source(format!("failed to decode OCR text: {err}")))?
                .trim()
                .to_string();
            validate_decoded_ocr_text(&text)?;
            let mut line = OcrTextLine::new(text.clone())?;
            if request.include_tokens {
                for token in text.split_whitespace() {
                    if let Ok(ocr_token) = OcrToken::new(token) {
                        line = line.token(ocr_token);
                    }
                }
            }
            let block = OcrTextBlock::paragraph(text.clone())?.line(line);
            let language = request.languages.first().cloned();
            let mut document = OcrDocument::new(text, image.width, image.height)?
                .block(block)
                .attribute(
                    "modelId",
                    self.spec.repo_id_value().unwrap_or("").to_string(),
                )
                .attribute("runtime", "onnx");
            if let Some(language) = language {
                document = document.language(language);
            }
            Ok(document)
        }

        #[cfg(not(feature = "local-onnx"))]
        {
            let _ = (image, request);
            let _ = (&self.encoder_path, &self.decoder_path, &self.tokenizer_path);
            Err(local_onnx_feature_error())
        }
    }
}

impl ModelBackedOcrBackend for OnnxTrOcrBackend {
    fn model_spec(&self) -> HuggingFaceModelSpec {
        self.spec.clone()
    }
}

#[derive(Debug, Clone, PartialEq)]
struct OnnxTrOcrBundleInfo {
    config_path: PathBuf,
    generation_config_path: Option<PathBuf>,
    preprocessor_config_path: PathBuf,
    tokenizer_config_path: Option<PathBuf>,
    tokenizer_path: PathBuf,
    encoder_path: PathBuf,
    decoder_path: PathBuf,
}

fn validate_onnx_trocr_bundle(bundle: &model_runtime::ModelBundle) -> Result<OnnxTrOcrBundleInfo> {
    match &bundle.manifest.task {
        ModelTask::Custom(task)
            if task == "optical_character_recognition" || task == "image_to_text" => {}
        task => {
            return Err(DetectError::InvalidArgument(format!(
                "ONNX TrOCR bundle {} task must be optical_character_recognition, got {:?}",
                bundle_descriptor(bundle),
                task
            )))
        }
    }
    let config_path = required_bundle_file(bundle, "config.json")?;
    let preprocessor_config_path = required_bundle_file(bundle, "preprocessor_config.json")?;
    let tokenizer_path = required_bundle_file(bundle, "tokenizer.json")?;
    let generation_config_path = bundle.file_path("generation_config.json");
    let tokenizer_config_path = bundle.file_path("tokenizer_config.json");
    let encoder_path = bundle
        .manifest
        .files
        .values()
        .find(|file| file.remote_path.ends_with(".onnx") && file.remote_path.contains("encoder"))
        .map(|file| bundle.root.join(&file.local_path))
        .ok_or_else(|| {
            DetectError::InvalidArgument(format!(
                "ONNX TrOCR bundle {} must contain an encoder `.onnx` model selected by remote path containing `encoder`",
                bundle_descriptor(bundle)
            ))
        })?;
    let decoder_path = bundle
        .manifest
        .files
        .values()
        .find(|file| {
            file.remote_path.ends_with(".onnx")
                && file.remote_path.contains("decoder")
                && !file.remote_path.contains("with_past")
                && !file.remote_path.contains("merged")
        })
        .map(|file| bundle.root.join(&file.local_path))
        .ok_or_else(|| {
            DetectError::InvalidArgument(format!(
                "ONNX TrOCR bundle {} must contain a decoder `.onnx` model selected by remote path containing `decoder` and excluding `with_past`/`merged`",
                bundle_descriptor(bundle)
            ))
        })?;
    Ok(OnnxTrOcrBundleInfo {
        config_path,
        generation_config_path,
        preprocessor_config_path,
        tokenizer_config_path,
        tokenizer_path,
        encoder_path,
        decoder_path,
    })
}

/// Returns ONNX TrOCR options from a model-runtime bundle.
pub fn onnx_trocr_options_from_bundle(
    bundle: &model_runtime::ModelBundle,
) -> Result<OnnxTrOcrOptions> {
    let info = validate_onnx_trocr_bundle(bundle)?;
    let config = read_json(&info.config_path)?;
    let generation = info
        .generation_config_path
        .as_deref()
        .map(read_json)
        .transpose()?;
    let preprocessing =
        image_model_preprocessing_from_config(&read_json(&info.preprocessor_config_path)?)?;
    validate_image_model_preprocessing(&preprocessing)?;
    let mut options = OnnxTrOcrOptions {
        preprocessing,
        max_new_tokens: token_config_usize(&config, "max_new_tokens")
            .or_else(|| {
                generation
                    .as_ref()
                    .and_then(|value| token_config_usize(value, "max_new_tokens"))
            })
            .or_else(|| {
                generation
                    .as_ref()
                    .and_then(|value| token_config_usize(value, "max_length"))
            })
            .or_else(|| token_config_usize(&config, "max_length"))
            .unwrap_or(128),
        decoder_start_token_id: token_id_from_configs(
            &config,
            generation.as_ref(),
            "decoder_start_token_id",
        ),
        eos_token_id: token_id_from_configs(&config, generation.as_ref(), "eos_token_id"),
        pad_token_id: token_id_from_configs(&config, generation.as_ref(), "pad_token_id"),
    };
    if options.decoder_start_token_id.is_none() {
        options.decoder_start_token_id =
            token_id_from_configs(&config, generation.as_ref(), "bos_token_id");
    }
    validate_trocr_max_tokens(options.max_new_tokens)?;
    Ok(options)
}

fn token_config_usize(config: &serde_json::Value, key: &str) -> Option<usize> {
    config
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

fn token_id_from_configs(
    config: &serde_json::Value,
    generation: Option<&serde_json::Value>,
    key: &str,
) -> Option<u32> {
    generation
        .and_then(|value| config_token_id(value, key))
        .or_else(|| config_token_id(config, key))
}

fn config_token_id(config: &serde_json::Value, key: &str) -> Option<u32> {
    let value = config.get(key)?;
    value
        .as_u64()
        .or_else(|| value.as_array()?.first()?.as_u64())
        .and_then(|value| u32::try_from(value).ok())
}

#[cfg(any(feature = "local-onnx", test))]
fn validate_trocr_options(options: &OnnxTrOcrOptions) -> Result<()> {
    validate_image_model_preprocessing(&options.preprocessing)?;
    validate_trocr_max_tokens(options.max_new_tokens)?;
    if options.decoder_start_token_id.is_none() {
        return Err(DetectError::InvalidArgument(
            "ONNX TrOCR requires decoder_start_token_id".to_string(),
        ));
    }
    if options.eos_token_id.is_none() {
        return Err(DetectError::InvalidArgument(
            "ONNX TrOCR requires eos_token_id".to_string(),
        ));
    }
    Ok(())
}

fn validate_trocr_max_tokens(max_new_tokens: usize) -> Result<()> {
    if max_new_tokens == 0 {
        return Err(DetectError::InvalidArgument(
            "ONNX TrOCR max_new_tokens must be greater than zero".to_string(),
        ));
    }
    Ok(())
}

#[cfg(any(feature = "local-onnx", test))]
fn greedy_decode_trocr_token_ids<F>(
    decoder_start_token_id: u32,
    eos_token_id: u32,
    max_new_tokens: usize,
    mut next_logits: F,
) -> Result<Vec<u32>>
where
    F: FnMut(&[u32]) -> Result<Vec<f32>>,
{
    validate_trocr_max_tokens(max_new_tokens)?;
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
            DetectError::InvalidArgument("ONNX TrOCR decoder logits were empty".to_string())
        })
}

#[cfg(feature = "local-onnx")]
fn resolve_trocr_token_ids(
    options: &mut OnnxTrOcrOptions,
    tokenizer: &tokenizers::Tokenizer,
    info: &OnnxTrOcrBundleInfo,
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
            &["<s>", "[CLS]", "<|startoftext|>"],
        );
    }
    if options.eos_token_id.is_none() {
        options.eos_token_id = tokenizer_special_id(
            tokenizer,
            tokenizer_config.as_ref(),
            &["eos_token", "sep_token"],
            &["</s>", "[SEP]", "<|endoftext|>"],
        );
    }
    if options.pad_token_id.is_none() {
        options.pad_token_id = tokenizer_special_id(
            tokenizer,
            tokenizer_config.as_ref(),
            &["pad_token"],
            &["<pad>", "[PAD]", "</s>"],
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
fn run_trocr_encoder(
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
fn run_trocr_decoder(
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
    let logits = trocr_logits_output(&outputs)?;
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
                "ONNX TrOCR decoder is missing input #{fallback_index}"
            ))
        })
}

#[cfg(any(feature = "local-onnx", test))]
fn trocr_logits_output(
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
            "ONNX TrOCR encoder hidden states must have shape `[batch, sequence, hidden]`, got `{shape:?}`"
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
                "ONNX TrOCR logits must have shape `[sequence, vocab]` or `[batch, sequence, vocab]`, got `{shape:?}`"
            )))
        }
    };
    if tensor.values.len() < vocab {
        return Err(DetectError::InvalidArgument(
            "ONNX TrOCR logits values do not match shape".to_string(),
        ));
    }
    Ok(tensor.values[tensor.values.len() - vocab..].to_vec())
}

#[cfg(any(feature = "local-onnx", test))]
fn validate_decoded_ocr_text(text: &str) -> Result<()> {
    if text.trim().is_empty() {
        return Err(DetectError::InvalidArgument(
            "ONNX TrOCR decoded empty OCR text".to_string(),
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

#[cfg(feature = "local-onnx")]
fn model_runtime_error(error: model_runtime::ModelRuntimeError) -> DetectError {
    match error {
        model_runtime::ModelRuntimeError::InvalidArgument(message) => {
            DetectError::InvalidArgument(message)
        }
        model_runtime::ModelRuntimeError::Source(message) => DetectError::Source(message),
        model_runtime::ModelRuntimeError::Io(error) => DetectError::Io(error),
    }
}

#[cfg(not(feature = "local-onnx"))]
fn local_onnx_feature_error() -> DetectError {
    DetectError::Source(
        "native ONNX TrOCR OCR execution requires building image-analysis-ocr with the `local-onnx` feature"
            .to_string(),
    )
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

fn bundle_spec(bundle: &model_runtime::ModelBundle) -> HuggingFaceModelSpec {
    HuggingFaceModelSpec::new(
        bundle.manifest.repo_id.clone(),
        bundle.manifest.task.clone(),
    )
    .name(bundle.manifest.name.clone())
    .revision(bundle.manifest.revision.clone())
}

fn read_json(path: &Path) -> Result<serde_json::Value> {
    let data = std::fs::read(path)?;
    serde_json::from_slice(&data).map_err(|err| {
        DetectError::Source(format!("failed to parse JSON `{}`: {err}", path.display()))
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// OCR backend that invokes an external command and reads an `OcrDocument` JSON object from stdout.
pub struct JsonCommandOcrBackend {
    command: PathBuf,
    args: Vec<String>,
}

impl JsonCommandOcrBackend {
    /// Creates a new backend.
    pub fn new(command: impl Into<PathBuf>) -> Self {
        Self {
            command: command.into(),
            args: Vec::new(),
        }
    }

    /// Returns this backend with command args.
    pub fn args(mut self, args: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.args = args.into_iter().map(Into::into).collect();
        self
    }

    /// Returns command path.
    pub fn command(&self) -> &std::path::Path {
        &self.command
    }

    /// Returns command args.
    pub fn command_args(&self) -> &[String] {
        &self.args
    }
}

impl OcrBackend for JsonCommandOcrBackend {
    fn recognize_image(
        &mut self,
        image: &ImageView<'_>,
        request: &OcrRequest,
    ) -> Result<OcrDocument> {
        let temp_image = tempfile::Builder::new()
            .suffix(".png")
            .tempfile()
            .map_err(DetectError::Io)?;
        let image_path = temp_image.path().to_path_buf();
        let owned = compact_image(image)?;
        image_analysis_io::write_image(&image_path, &owned)?;

        let mut saw_placeholder = false;
        let mut args = Vec::with_capacity(self.args.len() + 1);
        let image_arg = image_path.to_string_lossy().into_owned();
        for arg in &self.args {
            if arg.contains("{image}") {
                saw_placeholder = true;
                args.push(arg.replace("{image}", &image_arg));
            } else {
                args.push(arg.clone());
            }
        }
        if !saw_placeholder {
            args.push(image_arg);
        }

        let request_json = serde_json::to_string(&JsonCommandOcrRequest::from_request(
            request,
            image.width,
            image.height,
        ))
        .map_err(|err| DetectError::Source(format!("failed to serialize OCR request: {err}")))?;

        let output = Command::new(&self.command)
            .args(&args)
            .env("OCR_REQUEST_JSON", request_json)
            .output()
            .map_err(|err| {
                DetectError::Source(format!(
                    "failed to run OCR command `{}`: {err}",
                    self.command.display()
                ))
            })?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(DetectError::Source(format!(
                "OCR command `{}` exited with status {}{}",
                self.command.display(),
                output.status,
                if stderr.trim().is_empty() {
                    String::new()
                } else {
                    format!(": {}", stderr.trim())
                }
            )));
        }

        let mut value: serde_json::Value = serde_json::from_slice(&output.stdout)
            .map_err(|err| DetectError::Source(format!("invalid OCR JSON: {err}")))?;
        let object = value.as_object_mut().ok_or_else(|| {
            DetectError::Source("invalid OCR document: expected JSON object".to_string())
        })?;
        object
            .entry("width")
            .or_insert_with(|| serde_json::json!(image.width));
        object
            .entry("height")
            .or_insert_with(|| serde_json::json!(image.height));
        let document: OcrDocument = serde_json::from_value(value)
            .map_err(|err| DetectError::Source(format!("invalid OCR document: {err}")))?;
        validate_ocr_document(&document)?;
        Ok(document)
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct JsonCommandOcrRequest<'a> {
    width: u32,
    height: u32,
    languages: &'a [String],
    preserve_layout: bool,
    include_tokens: bool,
    min_confidence: Option<u8>,
    attributes: &'a BTreeMap<String, String>,
}

impl<'a> JsonCommandOcrRequest<'a> {
    fn from_request(request: &'a OcrRequest, width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            languages: &request.languages,
            preserve_layout: request.preserve_layout,
            include_tokens: request.include_tokens,
            min_confidence: request.min_confidence,
            attributes: &request.attributes,
        }
    }
}

fn validate_ocr_document(document: &OcrDocument) -> Result<()> {
    if document.width == 0 || document.height == 0 {
        return Err(DetectError::InvalidDimensions {
            width: document.width,
            height: document.height,
        });
    }
    for block in &document.blocks {
        if block.text.trim().is_empty() {
            return Err(DetectError::Source(
                "invalid OCR document: block text is required".to_string(),
            ));
        }
        for line in &block.lines {
            if line.text.trim().is_empty() {
                return Err(DetectError::Source(
                    "invalid OCR document: line text is required".to_string(),
                ));
            }
            for token in &line.tokens {
                if token.text.trim().is_empty() {
                    return Err(DetectError::Source(
                        "invalid OCR document: token text is required".to_string(),
                    ));
                }
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Data type for OCR output at one video frame.
pub struct FrameOcrDocument {
    /// The position value.
    pub position: FramePosition,
    /// OCR output for this frame.
    pub document: OcrDocument,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
/// Data type for OCR output over video frames.
pub struct VideoOcrDocument {
    /// OCR output for each sampled frame.
    pub frames: Vec<FrameOcrDocument>,
}

impl VideoOcrDocument {
    /// Returns combined text.
    pub fn combined_text(&self) -> String {
        self.frames
            .iter()
            .map(|frame| frame.document.text.trim())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Data type for OCR pipeline.
pub struct OcrPipeline<B> {
    backend: B,
    request: OcrRequest,
}

impl<B> OcrPipeline<B> {
    /// Creates a new value.
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            request: OcrRequest::default(),
        }
    }

    /// Returns request.
    pub fn request(mut self, value: OcrRequest) -> Self {
        self.request = value;
        self
    }

    /// Returns backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Returns backend mut.
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }
}

impl<B: OcrBackend> OcrPipeline<B> {
    /// Returns recognize image.
    pub fn recognize_image(&mut self, image: &ImageView<'_>) -> Result<OcrDocument> {
        self.backend.recognize_image(image, &self.request)
    }

    /// Returns recognize frame.
    pub fn recognize_frame(&mut self, frame: &VideoFrame<'_>) -> Result<FrameOcrDocument> {
        let image = ImageView::from_video_frame(frame)?;
        let document = self.recognize_image(&image)?;
        Ok(FrameOcrDocument {
            position: frame.position,
            document,
        })
    }

    /// Returns recognize frames.
    pub fn recognize_frames<'a>(
        &mut self,
        frames: impl IntoIterator<Item = &'a VideoFrame<'a>>,
    ) -> Result<VideoOcrDocument> {
        let mut output = VideoOcrDocument::default();
        for frame in frames {
            output.frames.push(self.recognize_frame(frame)?);
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image_analysis_core::OwnedImage;
    use model_runtime::{ModelBundle, ModelBundleFile, ModelBundleManifest};
    use std::fs;
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;
    use tempfile::tempdir;
    use video_analysis_core::{PixelFormat, Timebase, Timestamp};

    struct StubOcrBackend;

    impl OcrBackend for StubOcrBackend {
        fn recognize_image(
            &mut self,
            image: &ImageView<'_>,
            request: &OcrRequest,
        ) -> Result<OcrDocument> {
            let language = request
                .languages
                .first()
                .cloned()
                .unwrap_or_else(|| "und".to_string());
            Ok(OcrDocument::new("Hello OCR", image.width, image.height)?
                .language(language)
                .confidence(96)
                .block(
                    OcrTextBlock::paragraph("Hello OCR")?
                        .region(BoundingBox::new(1, 1, image.width - 1, image.height - 1)?)
                        .line(
                            OcrTextLine::new("Hello OCR")?
                                .token(OcrToken::new("Hello")?)
                                .token(OcrToken::new("OCR")?),
                        ),
                ))
        }
    }

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
                name: "trocr-test".to_string(),
                repo_id: "Xenova/trocr-base-printed".to_string(),
                revision: "main".to_string(),
                task: ModelTask::Custom("optical_character_recognition".to_string()),
                files: manifest_files,
            },
        };
        (temp, bundle)
    }

    #[test]
    fn default_preset_uses_trocr_printed() {
        assert_eq!(
            default_ocr_model_spec().repo_id_value(),
            Some("microsoft/trocr-base-printed")
        );
        assert_eq!(
            OcrPreset::from_str("trocr-base-handwritten")
                .unwrap()
                .model_spec()
                .repo_id_value(),
            Some("microsoft/trocr-base-handwritten")
        );
    }

    #[test]
    fn onnx_trocr_presets_use_xenova_encoder_decoder_specs() {
        let printed = OcrPreset::TrOcrBasePrintedOnnx.model_spec();
        assert_eq!(printed.repo_id_value(), Some("Xenova/trocr-base-printed"));
        let files = format!("{:?}", printed.files);
        assert!(files.contains("encoder"));
        assert!(files.contains("decoder"));

        let request = OcrRequest::new().model_preset(OcrPreset::TrOcrBaseHandwrittenOnnx);
        assert!(matches!(
            request.technique,
            Some(OcrTechnique::OnnxModel(_))
        ));
        assert!(ocr_catalog()
            .iter()
            .any(|entry| entry.id == "trocr-base-printed-onnx"
                && entry.model_id == "Xenova/trocr-base-printed"));
    }

    #[test]
    fn onnx_trocr_bundle_validates_required_files_and_decoder_selection() {
        let (_temp, bundle) = test_bundle(&[
            (
                "config.json",
                "config.json",
                r#"{"decoder_start_token_id":1,"eos_token_id":2}"#,
            ),
            (
                "generation_config.json",
                "generation_config.json",
                r#"{"max_new_tokens":12}"#,
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
                "onnx/decoder_model_merged.onnx",
                "onnx/decoder_model_merged.onnx",
                "merged",
            ),
            (
                "onnx/decoder_model.onnx",
                "onnx/decoder_model.onnx",
                "decoder",
            ),
        ]);
        let info = validate_onnx_trocr_bundle(&bundle).unwrap();
        assert!(info.decoder_path.ends_with("onnx/decoder_model.onnx"));
        let options = onnx_trocr_options_from_bundle(&bundle).unwrap();
        assert_eq!(options.decoder_start_token_id, Some(1));
        assert_eq!(options.eos_token_id, Some(2));
        assert_eq!(options.max_new_tokens, 12);
        assert_eq!(options.preprocessing.input_width, 128);
        assert_eq!(options.preprocessing.input_height, 96);
    }

    #[test]
    fn onnx_trocr_bundle_reports_missing_encoder_decoder_tokenizer() {
        let (_temp, bundle) = test_bundle(&[
            ("config.json", "config.json", r#"{}"#),
            (
                "preprocessor_config.json",
                "preprocessor_config.json",
                r#"{"size":384}"#,
            ),
        ]);
        let error = onnx_trocr_options_from_bundle(&bundle)
            .unwrap_err()
            .to_string();
        assert!(error.contains("tokenizer.json"));

        let (_temp, bundle) = test_bundle(&[
            ("config.json", "config.json", r#"{}"#),
            (
                "preprocessor_config.json",
                "preprocessor_config.json",
                r#"{"size":384}"#,
            ),
            ("tokenizer.json", "tokenizer.json", "{}"),
        ]);
        let error = validate_onnx_trocr_bundle(&bundle).unwrap_err().to_string();
        assert!(error.contains("encoder"));

        let (_temp, bundle) = test_bundle(&[
            ("config.json", "config.json", r#"{}"#),
            (
                "preprocessor_config.json",
                "preprocessor_config.json",
                r#"{"size":384}"#,
            ),
            ("tokenizer.json", "tokenizer.json", "{}"),
            (
                "onnx/encoder_model.onnx",
                "onnx/encoder_model.onnx",
                "encoder",
            ),
            (
                "onnx/decoder_model_merged.onnx",
                "onnx/decoder_model_merged.onnx",
                "merged",
            ),
        ]);
        let error = validate_onnx_trocr_bundle(&bundle).unwrap_err().to_string();
        assert!(error.contains("decoder"));
        assert!(error.contains("with_past"));
    }

    #[test]
    fn token_ids_read_generation_before_config_and_arrays() {
        let config = serde_json::json!({"eos_token_id": [9], "decoder_start_token_id": 1});
        let generation = serde_json::json!({"eos_token_id": 2, "pad_token_id": 0});
        assert_eq!(
            token_id_from_configs(&config, Some(&generation), "eos_token_id"),
            Some(2)
        );
        assert_eq!(
            token_id_from_configs(&config, Some(&generation), "decoder_start_token_id"),
            Some(1)
        );
        assert_eq!(
            token_id_from_configs(&config, Some(&generation), "pad_token_id"),
            Some(0)
        );
    }

    #[test]
    fn greedy_decoder_stops_on_eos_and_rejects_empty_logits() {
        let mut calls = 0;
        let generated = greedy_decode_trocr_token_ids(1, 3, 8, |_| {
            calls += 1;
            Ok(match calls {
                1 => vec![0.0, 0.0, 8.0, 0.0],
                _ => vec![0.0, 0.0, 0.0, 9.0],
            })
        })
        .unwrap();
        assert_eq!(generated, vec![2]);
        assert_eq!(calls, 2);

        let error = greedy_decode_trocr_token_ids(1, 3, 8, |_| Ok(Vec::new()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("logits"));
        assert!(validate_decoded_ocr_text(" ").is_err());
    }

    #[test]
    fn onnx_trocr_options_and_logits_validate_shapes() {
        let missing = OnnxTrOcrOptions::default();
        assert!(validate_trocr_options(&missing).is_err());
        let present = OnnxTrOcrOptions {
            decoder_start_token_id: Some(1),
            eos_token_id: Some(2),
            ..OnnxTrOcrOptions::default()
        };
        assert!(validate_trocr_options(&present).is_ok());

        let hidden = runtime_onnx::OnnxF32Tensor::new(vec![1, 2, 3], vec![0.0; 6]).unwrap();
        assert!(validate_encoder_hidden_states(&hidden).is_ok());
        let logits =
            runtime_onnx::OnnxF32Tensor::new(vec![1, 2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
                .unwrap();
        assert_eq!(final_token_logits(&logits).unwrap(), vec![3.0, 4.0, 5.0]);
        let outputs = vec![runtime_onnx::OnnxNamedTensor {
            name: "logits".to_string(),
            tensor: runtime_onnx::OnnxTensorValue::F32(logits),
        }];
        assert_eq!(trocr_logits_output(&outputs).unwrap().shape, vec![1, 2, 3]);
    }

    #[test]
    fn ocr_document_converts_to_text_document_contract_with_spans() {
        let document = OcrDocument::new("Hello OCR", 640, 480)
            .unwrap()
            .language("en")
            .confidence(95)
            .block(
                OcrTextBlock::paragraph("Hello OCR").unwrap().line(
                    OcrTextLine::new("Hello OCR")
                        .unwrap()
                        .token(OcrToken::new("Hello").unwrap())
                        .token(OcrToken::new("OCR").unwrap()),
                ),
            );
        let conversion = document.to_text_document_contract(OcrTextContractOptions {
            id: "doc-1".to_string(),
            source_id: Some("image-1".to_string()),
            model_id: Some("trocr-base-printed-onnx".to_string()),
            runtime: "onnx".to_string(),
            ..OcrTextContractOptions::default()
        });
        assert_eq!(conversion.document.id, "doc-1");
        assert_eq!(conversion.document.text, "Hello OCR");
        assert_eq!(conversion.document.language.as_deref(), Some("en"));
        assert_eq!(
            conversion
                .document
                .source
                .as_ref()
                .and_then(|source| source.source_kind.as_deref()),
            Some("ocr_image")
        );
        assert_eq!(
            conversion
                .document
                .attributes
                .get("ocr.width")
                .map(String::as_str),
            Some("640")
        );
        assert_eq!(
            conversion.document.provenance[0].model_id.as_deref(),
            Some("trocr-base-printed-onnx")
        );
        assert_eq!(conversion.aligned_blocks, 1);
        assert_eq!(conversion.aligned_lines, 1);
        assert_eq!(conversion.aligned_tokens, 2);
        assert_eq!(conversion.document.annotations.len(), 4);
    }

    #[test]
    fn ocr_document_conversion_counts_unaligned_structure_without_panics() {
        let document = OcrDocument::new("Only visible text", 10, 10)
            .unwrap()
            .block(
                OcrTextBlock::paragraph("Missing block").unwrap().line(
                    OcrTextLine::new("Only visible text")
                        .unwrap()
                        .token(OcrToken::new("Only").unwrap())
                        .token(OcrToken::new("Missing").unwrap()),
                ),
            );
        let conversion = document.to_text_document_contract(OcrTextContractOptions::default());
        assert_eq!(conversion.unaligned_blocks, 1);
        assert_eq!(conversion.aligned_lines, 1);
        assert_eq!(conversion.aligned_tokens, 1);
        assert_eq!(conversion.unaligned_tokens, 1);
    }

    #[test]
    fn request_can_select_multiple_model_and_command_techniques() {
        let request = OcrRequest::new()
            .model_preset(OcrPreset::DonutBaseCordV2)
            .languages(["en", "de"])
            .min_confidence(125);
        assert_eq!(request.min_confidence, Some(100));
        assert_eq!(request.languages, ["en".to_string(), "de".to_string()]);
        assert!(matches!(
            request.technique,
            Some(OcrTechnique::HuggingFaceModel(_))
        ));

        let command = ExternalOcrCommandSpec::new("tesseract", "tesseract")
            .unwrap()
            .arg("--psm")
            .arg("6");
        let technique = OcrTechnique::ExternalCommand(command);
        assert_eq!(technique.kind(), "external_command");
        assert_eq!(technique.runtime_backend(), ModelRuntimeBackend::External);
    }

    #[test]
    fn rich_text_blocks_preserve_layout() {
        let block = OcrTextBlock::new(OcrBlockKind::Heading, "Invoice 42")
            .unwrap()
            .confidence(91)
            .line(
                OcrTextLine::new("Invoice 42")
                    .unwrap()
                    .token(OcrToken::new("Invoice").unwrap())
                    .token(OcrToken::new("42").unwrap()),
            );
        let document = OcrDocument::new("Invoice 42", 320, 180)
            .unwrap()
            .language("en")
            .block(block);
        assert_eq!(document.blocks[0].lines[0].tokens.len(), 2);
        assert_eq!(document.language.as_deref(), Some("en"));
    }

    #[test]
    fn pipeline_runs_ocr_on_images_and_video_frames() {
        let image = OwnedImage::new_rgb(8, 8, vec![255; 8 * 8 * 3]).unwrap();
        let mut pipeline = OcrPipeline::new(StubOcrBackend)
            .request(OcrRequest::new().languages(["en"]).include_tokens(true));

        let document = pipeline.recognize_image(&image.as_view()).unwrap();
        assert_eq!(document.text, "Hello OCR");
        assert_eq!(document.blocks[0].lines[0].tokens[0].text, "Hello");

        let frame_position = FramePosition {
            frame_index: 3,
            timestamp: Timestamp::new(3, Timebase::new(1, 30)),
        };
        let frame = VideoFrame {
            position: frame_position,
            width: 8,
            height: 8,
            pixel_format: PixelFormat::Rgb24,
            data: &image.data,
            stride: image.stride,
        };
        let frame_document = pipeline.recognize_frame(&frame).unwrap();
        assert_eq!(frame_document.position.frame_index, 3);
        assert_eq!(frame_document.document.text, "Hello OCR");
    }

    #[test]
    fn empty_document_uses_image_dimensions() {
        let image = OwnedImage::new_rgb(2, 3, vec![0; 18]).unwrap();
        let document = OcrDocument::empty_for_image(&image.as_view());
        assert_eq!((document.width, document.height), (2, 3));
        assert!(document.text.is_empty());
    }

    #[test]
    fn rejects_empty_structural_text() {
        assert!(OcrToken::new(" ").is_err());
        assert!(OcrTextLine::new("").is_err());
        assert!(OcrTextBlock::paragraph("").is_err());
        assert!(OcrDocument::new("", 0, 1).is_err());
    }

    #[test]
    fn frame_positions_remain_plain_core_types() {
        let position = FramePosition {
            frame_index: 9,
            timestamp: Timestamp::new(9, Timebase::new(1, 24)),
        };
        assert_eq!(position.timestamp.seconds(), 0.375);
    }

    #[test]
    fn json_command_backend_parses_valid_document() {
        let script = fixture_script(
            r#"#!/bin/sh
test -f "$1" || exit 9
printf '%s' '{"text":"Slide Title","width":4,"height":3,"language":"en","confidence":88,"blocks":[{"kind":"heading","text":"Slide Title","region":{"x":1,"y":1,"width":2,"height":1},"confidence":90,"lines":[],"attributes":{}}],"attributes":{}}'
"#,
        );
        let image = OwnedImage::new_rgb(4, 3, vec![255; 36]).unwrap();
        let mut backend = JsonCommandOcrBackend::new(script.path());
        let document = backend
            .recognize_image(&image.as_view(), &OcrRequest::new().languages(["en"]))
            .unwrap();
        assert_eq!(document.text, "Slide Title");
        assert_eq!((document.width, document.height), (4, 3));
        assert_eq!(document.blocks[0].kind, OcrBlockKind::Heading);
        assert_eq!(document.blocks[0].region.unwrap().x, 1);
    }

    #[test]
    fn json_command_backend_fills_missing_dimensions() {
        let script = fixture_script(
            r#"#!/bin/sh
printf '%s' '{"text":"No dimensions","blocks":[{"kind":"paragraph","text":"No dimensions","lines":[],"attributes":{}}],"attributes":{}}'
"#,
        );
        let image = OwnedImage::new_rgb(5, 2, vec![0; 30]).unwrap();
        let mut backend = JsonCommandOcrBackend::new(script.path()).args(["--image={image}"]);
        let document = backend
            .recognize_image(&image.as_view(), &OcrRequest::new())
            .unwrap();
        assert_eq!((document.width, document.height), (5, 2));
    }

    #[test]
    fn json_command_backend_reports_command_failure_and_invalid_json() {
        let failing = fixture_script(
            r#"#!/bin/sh
echo "no OCR available" >&2
exit 42
"#,
        );
        let invalid = fixture_script(
            r#"#!/bin/sh
printf '%s' 'not-json'
"#,
        );
        let image = OwnedImage::new_rgb(2, 2, vec![0; 12]).unwrap();

        let mut failing_backend = JsonCommandOcrBackend::new(failing.path());
        let error = failing_backend
            .recognize_image(&image.as_view(), &OcrRequest::new())
            .unwrap_err()
            .to_string();
        assert!(error.contains("exited with status"));
        assert!(error.contains("no OCR available"));

        let mut invalid_backend = JsonCommandOcrBackend::new(invalid.path());
        let error = invalid_backend
            .recognize_image(&image.as_view(), &OcrRequest::new())
            .unwrap_err()
            .to_string();
        assert!(error.contains("invalid OCR JSON"));
    }

    #[test]
    fn json_command_backend_exposes_request_environment() {
        let script = fixture_script(
            r#"#!/bin/sh
case "$OCR_REQUEST_JSON" in
  *'"languages":["de"]'*'"preserveLayout":false'*'"includeTokens":false'*'"minConfidence":77'*)
    printf '%s' '{"text":"Umgebung","blocks":[],"attributes":{}}'
    ;;
  *)
    echo "$OCR_REQUEST_JSON" >&2
    exit 7
    ;;
esac
"#,
        );
        let image = OwnedImage::new_rgb(3, 3, vec![0; 27]).unwrap();
        let request = OcrRequest::new()
            .languages(["de"])
            .preserve_layout(false)
            .include_tokens(false)
            .min_confidence(77);
        let mut backend = JsonCommandOcrBackend::new(script.path());
        let document = backend.recognize_image(&image.as_view(), &request).unwrap();
        assert_eq!(document.text, "Umgebung");
        assert_eq!((document.width, document.height), (3, 3));
    }

    struct FixtureScript {
        _dir: tempfile::TempDir,
        path: std::path::PathBuf,
    }

    impl FixtureScript {
        fn path(&self) -> &std::path::Path {
            &self.path
        }
    }

    fn fixture_script(contents: &str) -> FixtureScript {
        let dir = tempdir().unwrap();
        let path = dir.path().join("ocr-fixture.sh");
        let mut file = fs::File::create(&path).unwrap();
        file.write_all(contents.as_bytes()).unwrap();
        let mut permissions = file.metadata().unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&path, permissions).unwrap();
        drop(file);
        FixtureScript { _dir: dir, path }
    }
}
