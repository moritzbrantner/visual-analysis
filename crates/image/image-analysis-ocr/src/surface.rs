//! Library-owned runtime surface for `image-analysis-ocr`.

use std::path::PathBuf;

use image_analysis_core::contracts::{sample_image_json, ImagePayload};
use runtime_core::{
    describe_surface_response, structured_operation_response, OperationId, PackageSurface,
    RuntimeCapabilities, RuntimeRequirement, SurfaceExecutionMode, SurfaceExecutionPlan,
    SurfaceOperation, SurfaceRequest, SurfaceResponse, SurfaceSideEffect,
};
use serde::Deserialize;
use video_analysis_core::BoundingBox;

use crate::{
    ocr_catalog, OcrBackend, OcrBlockKind, OcrConfidence, OcrDocument, OcrLocalModelOptions,
    OcrPreset, OcrRequest, OcrTechnique, OcrTextBlock, OcrTextContractOptions, OcrTextLine,
    OcrToken, OnnxTrOcrBackend,
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
                "OCR model presets, native TrOCR execution, imported OCR summaries, and text document conversion.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "image.ocr.recognize",
                "Recognize OCR text",
                "Runs local ONNX TrOCR against an in-memory image payload or local image path. Model downloads remain opt-in through autoDownload.",
                serde_json::json!({"image": sample_image_json(), "model": "trocr-base-printed-onnx", "autoDownload": false, "maxNewTokens": 64, "languages": ["en"]}),
            ),
            operation(
                "image.ocr.toTextDocument",
                "OCR to text document",
                "Converts imported OCR output into a text-core TextDocumentContract with provenance, source metadata, attributes, and best-effort span annotations.",
                serde_json::json!({"document": {"text": "Hello OCR", "width": 640, "height": 480, "language": "en", "confidence": 95, "blocks": [{"kind": "paragraph", "text": "Hello OCR", "lines": [{"text": "Hello OCR", "tokens": [{"text": "Hello"}, {"text": "OCR"}]}]}]}, "id": "ocr-doc-1", "modelId": "trocr-base-printed-onnx", "runtime": "onnx"}),
            ),
            operation(
                "image.ocr.models",
                "OCR models",
                "Lists OCR model catalog entries and runtime availability without downloading or running models.",
                serde_json::json!({}),
            ),
            operation(
                "image.ocr.presets",
                "OCR presets",
                "Lists supported OCR preset ids and model specs without downloading or running models.",
                serde_json::json!({}),
            ),
            operation(
                "image.ocr.requestSummary",
                "OCR request summary",
                "Validates OCR request-shaped input and returns normalized option summary.",
                serde_json::json!({"preset": "trocr-base-printed", "languages": ["en"], "preserveLayout": true, "includeTokens": false, "minConfidence": 80}),
            ),
            operation(
                "image.ocr.documentSummary",
                "OCR document summary",
                "Validates imported OCR document-shaped input and returns text, block, line, and token counts.",
                serde_json::json!({"text": "Hello OCR", "width": 640, "height": 480, "language": "en", "confidence": 95, "blocks": [{"kind": "paragraph", "text": "Hello OCR", "lines": [{"text": "Hello OCR", "tokens": [{"text": "Hello"}, {"text": "OCR"}]}]}]}),
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
    let mut operation = runtime_core::surface_operation(id, name, description, example_request);
    if id == "image.ocr.recognize" {
        let plan = model_execution_plan(id);
        operation.input_schema["xExecutionPlan"] =
            runtime_core::surface_execution_plan_value(&plan);
        operation.output_schema["xExecutionPlan"] =
            runtime_core::surface_execution_plan_value(&plan);
        operation.wasm_supported = false;
    }
    if matches!(
        id,
        "describe"
            | "image.ocr.models"
            | "image.ocr.presets"
            | "image.ocr.requestSummary"
            | "image.ocr.documentSummary"
    ) {
        operation.input_schema["xOperationCategory"] = serde_json::json!("debug");
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
        progress_unit: Some("tokens".to_string()),
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
        "image.ocr.recognize" => recognize_value(parse_input(request.input)?)?,
        "image.ocr.toTextDocument" => to_text_document_value(parse_input(request.input)?)?,
        "image.ocr.models" => models_value(),
        "image.ocr.presets" => presets_value(),
        "image.ocr.requestSummary" => request_summary_value(parse_input(request.input)?)?,
        "image.ocr.documentSummary" => document_summary_value(parse_input(request.input)?)?,
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
struct RecognizeInput {
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
    max_new_tokens: Option<usize>,
    #[serde(default)]
    languages: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RequestSummaryInput {
    #[serde(default)]
    preset: Option<String>,
    #[serde(default)]
    technique: Option<String>,
    #[serde(default)]
    languages: Vec<String>,
    #[serde(default = "default_true")]
    preserve_layout: bool,
    #[serde(default = "default_true")]
    include_tokens: bool,
    #[serde(default)]
    min_confidence: Option<u8>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DocumentInput {
    text: String,
    width: u32,
    height: u32,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    confidence: Option<u8>,
    #[serde(default)]
    blocks: Vec<BlockInput>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ToTextDocumentInput {
    document: DocumentInput,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    source_id: Option<String>,
    #[serde(default)]
    source_uri: Option<String>,
    #[serde(default)]
    model_id: Option<String>,
    #[serde(default)]
    runtime: Option<String>,
    #[serde(default)]
    layout_mode: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BlockInput {
    #[serde(default = "default_block_kind")]
    kind: String,
    text: String,
    #[serde(default)]
    confidence: Option<u8>,
    #[serde(default)]
    region: Option<BoxInput>,
    #[serde(default)]
    lines: Vec<LineInput>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LineInput {
    text: String,
    #[serde(default)]
    confidence: Option<u8>,
    #[serde(default)]
    region: Option<BoxInput>,
    #[serde(default)]
    tokens: Vec<TokenInput>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TokenInput {
    text: String,
    #[serde(default)]
    confidence: Option<u8>,
    #[serde(default)]
    region: Option<BoxInput>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BoxInput {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

fn presets_value() -> serde_json::Value {
    serde_json::json!({
        "presets": OcrPreset::ALL.iter().map(|preset| {
            let spec = preset.model_spec();
            serde_json::json!({
                "id": preset.as_str(),
                "repoId": spec.repo_id_value(),
                "name": spec.name,
                "task": spec.task.as_protocol_str(),
                "files": spec.files
            })
        }).collect::<Vec<_>>()
    })
}

fn models_value() -> serde_json::Value {
    serde_json::json!({
        "models": ocr_catalog()
    })
}

fn recognize_value(input: RecognizeInput) -> Result<serde_json::Value, String> {
    if input.image.is_some() && input.image_path.is_some() {
        return Err("image.ocr.recognize accepts either image or imagePath, not both".to_string());
    }
    if input.image.is_none() && input.image_path.is_none() {
        return Err("image.ocr.recognize requires image or imagePath".to_string());
    }

    let model = input
        .model
        .unwrap_or_else(|| "trocr-base-printed-onnx".to_string());
    let preset = model.parse::<OcrPreset>().map_err(|error| error.to_string())?;
    if !matches!(
        preset,
        OcrPreset::TrOcrBasePrintedOnnx | OcrPreset::TrOcrBaseHandwrittenOnnx
    ) {
        return Err(format!(
            "image.ocr.recognize requires an ONNX TrOCR preset, got `{model}`"
        ));
    }

    if input.max_new_tokens.is_some_and(|value| value == 0) {
        return Err("OCR maxNewTokens must be greater than zero".to_string());
    }

    let auto_download = input.auto_download.unwrap_or(false);
    let mut local_options = OcrLocalModelOptions {
        preset,
        auto_download,
        ..OcrLocalModelOptions::default()
    };
    if let Some(model_root) = input.model_root.as_deref() {
        local_options.bundle_root = PathBuf::from(model_root);
    }
    let bundle_path = local_options.bundle_root.display().to_string();
    let mut backend = OnnxTrOcrBackend::from_local_model(local_options)
        .map_err(|error| error.to_string())?;
    if let Some(max_new_tokens) = input.max_new_tokens {
        backend.options.max_new_tokens = max_new_tokens;
    }
    let max_new_tokens = backend.options().max_new_tokens;
    let request = OcrRequest::new()
        .model_preset(preset)
        .languages(input.languages.clone());

    let document = if let Some(image) = input.image {
        let view = image.view().map_err(|error| error.to_string())?;
        backend
            .recognize_image(&view, &request)
            .map_err(|error| error.to_string())?
    } else {
        let image_path = input.image_path.expect("validated imagePath");
        let image = image_analysis_io::read_image(&image_path)
            .map_err(|error| format!("failed to read OCR image `{image_path}`: {error}"))?;
        backend
            .recognize_image(&image.as_view(), &request)
            .map_err(|error| error.to_string())?
    };

    Ok(serde_json::json!({
        "executed": true,
        "nativeOnly": true,
        "cliSupported": true,
        "wasmSupported": false,
        "serverSupported": true,
        "document": document,
        "modelId": preset.as_str(),
        "bundlePath": bundle_path,
        "runtime": "onnx",
        "autoDownload": auto_download,
        "maxNewTokens": max_new_tokens,
        "languages": input.languages
    }))
}

fn to_text_document_value(input: ToTextDocumentInput) -> Result<serde_json::Value, String> {
    let document = build_document(input.document)?;
    let options = OcrTextContractOptions {
        id: input.id.unwrap_or_else(|| "ocr-document".to_string()),
        source_id: input.source_id,
        source_uri: input.source_uri,
        model_id: input.model_id,
        runtime: input.runtime.unwrap_or_else(|| "ocr".to_string()),
        layout_mode: input
            .layout_mode
            .unwrap_or_else(|| "imported_layout".to_string()),
        ..OcrTextContractOptions::default()
    };
    let conversion = document.to_text_document_contract(options);
    Ok(serde_json::json!({
        "document": conversion.document,
        "alignedBlocks": conversion.aligned_blocks,
        "alignedLines": conversion.aligned_lines,
        "alignedTokens": conversion.aligned_tokens,
        "unalignedBlocks": conversion.unaligned_blocks,
        "unalignedLines": conversion.unaligned_lines,
        "unalignedTokens": conversion.unaligned_tokens
    }))
}

fn request_summary_value(input: RequestSummaryInput) -> Result<serde_json::Value, String> {
    let mut request = OcrRequest::new()
        .languages(input.languages)
        .preserve_layout(input.preserve_layout)
        .include_tokens(input.include_tokens);
    if let Some(confidence) = input.min_confidence {
        request = request.min_confidence(confidence);
    }
    if let Some(preset) = input.preset {
        let preset = preset
            .parse::<OcrPreset>()
            .map_err(|error| error.to_string())?;
        request = request.model_preset(preset);
    } else if let Some(technique) = input.technique {
        request = request.technique(match technique.as_str() {
            "heuristic" => OcrTechnique::Heuristic,
            other => OcrTechnique::Custom(other.to_string()),
        });
    }
    let technique = request.technique.as_ref();
    Ok(serde_json::json!({
        "technique": technique.map(|technique| technique.kind()),
        "runtime": technique.map(|technique| technique.runtime_backend().as_str().to_string()),
        "languages": request.languages,
        "preserveLayout": request.preserve_layout,
        "includeTokens": request.include_tokens,
        "minConfidence": request.min_confidence
    }))
}

fn document_summary_value(input: DocumentInput) -> Result<serde_json::Value, String> {
    let document = build_document(input)?;
    let line_count = document
        .blocks
        .iter()
        .map(|block| block.lines.len())
        .sum::<usize>();
    let token_count = document
        .blocks
        .iter()
        .flat_map(|block| &block.lines)
        .map(|line| line.tokens.len())
        .sum::<usize>();
    Ok(serde_json::json!({
        "textLength": document.text.chars().count(),
        "blocks": document.blocks.len(),
        "lines": line_count,
        "tokens": token_count,
        "dimensions": {"width": document.width, "height": document.height},
        "language": document.language,
        "confidence": document.confidence.map(|confidence| confidence.value())
    }))
}

fn build_document(input: DocumentInput) -> Result<OcrDocument, String> {
    let mut document = OcrDocument::new(input.text, input.width, input.height)
        .map_err(|error| error.to_string())?;
    if let Some(language) = input.language {
        document = document.language(language);
    }
    if let Some(confidence) = input.confidence {
        document = document.confidence(OcrConfidence::new(confidence));
    }
    for block in input.blocks {
        document = document.block(build_block(block)?);
    }
    Ok(document)
}

fn build_block(input: BlockInput) -> Result<OcrTextBlock, String> {
    let mut block = OcrTextBlock::new(parse_block_kind(&input.kind), input.text)
        .map_err(|error| error.to_string())?;
    if let Some(confidence) = input.confidence {
        block = block.confidence(confidence);
    }
    if let Some(region) = input.region {
        block = block.region(region.bounding_box()?);
    }
    for line in input.lines {
        block = block.line(build_line(line)?);
    }
    Ok(block)
}

fn build_line(input: LineInput) -> Result<OcrTextLine, String> {
    let mut line = OcrTextLine::new(input.text).map_err(|error| error.to_string())?;
    if let Some(confidence) = input.confidence {
        line = line.confidence(confidence);
    }
    if let Some(region) = input.region {
        line = line.region(region.bounding_box()?);
    }
    for token in input.tokens {
        line = line.token(build_token(token)?);
    }
    Ok(line)
}

fn build_token(input: TokenInput) -> Result<OcrToken, String> {
    let mut token = OcrToken::new(input.text).map_err(|error| error.to_string())?;
    if let Some(confidence) = input.confidence {
        token = token.confidence(confidence);
    }
    if let Some(region) = input.region {
        token = token.region(region.bounding_box()?);
    }
    Ok(token)
}

fn parse_block_kind(value: &str) -> OcrBlockKind {
    match value {
        "paragraph" => OcrBlockKind::Paragraph,
        "heading" => OcrBlockKind::Heading,
        "table" => OcrBlockKind::Table,
        "form" => OcrBlockKind::Form,
        "caption" => OcrBlockKind::Caption,
        other => OcrBlockKind::Custom(other.to_string()),
    }
}

impl BoxInput {
    fn bounding_box(self) -> Result<BoundingBox, String> {
        BoundingBox::new(self.x, self.y, self.width, self.height).map_err(|error| error.to_string())
    }
}

fn parse_input<T: for<'de> Deserialize<'de>>(input: serde_json::Value) -> Result<T, String> {
    serde_json::from_value(input).map_err(|error| format!("invalid request: {error}"))
}

fn default_true() -> bool {
    true
}

fn default_block_kind() -> String {
    "paragraph".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_surface_lists_ocr_operations() {
        let ids = package_surface()
            .operations
            .into_iter()
            .map(|operation| operation.id.0)
            .collect::<Vec<_>>();
        assert!(ids.contains(&"image.ocr.recognize".to_string()));
        assert!(ids.contains(&"image.ocr.toTextDocument".to_string()));
        assert!(ids.contains(&"image.ocr.models".to_string()));
        assert!(ids.contains(&"image.ocr.presets".to_string()));
        assert!(ids.contains(&"image.ocr.documentSummary".to_string()));
    }

    #[test]
    fn recognize_requires_exactly_one_image_source() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.ocr.recognize"),
            input: serde_json::json!({"model": "trocr-base-printed-onnx"}),
        })
        .expect_err("missing image");
        assert!(error.contains("requires image or imagePath"));
    }

    #[test]
    fn recognize_rejects_non_onnx_preset_before_model_resolution() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.ocr.recognize"),
            input: serde_json::json!({
                "image": sample_image_json(),
                "model": "trocr-base-printed",
                "autoDownload": false
            }),
        })
        .expect_err("non ONNX preset");
        assert!(error.contains("requires an ONNX TrOCR preset"));
    }

    #[test]
    fn to_text_document_returns_text_core_contract() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.ocr.toTextDocument"),
            input: serde_json::json!({
                "document": {
                    "text": "Hello OCR",
                    "width": 10,
                    "height": 8,
                    "language": "en",
                    "confidence": 95,
                    "blocks": [{"text": "Hello OCR", "lines": [{"text": "Hello OCR", "tokens": [{"text": "Hello"}, {"text": "OCR"}]}]}]
                },
                "id": "ocr-doc-1",
                "modelId": "trocr-base-printed-onnx",
                "runtime": "onnx"
            }),
        })
        .expect("text document");
        assert_eq!(response.value["document"]["id"], "ocr-doc-1");
        assert_eq!(response.value["document"]["text"], "Hello OCR");
        assert_eq!(
            response.value["document"]["source"]["sourceKind"],
            "ocr_image"
        );
        assert_eq!(response.value["alignedTokens"], 2);
    }

    #[test]
    fn document_summary_counts_nested_tokens() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.ocr.documentSummary"),
            input: serde_json::json!({
                "text": "Hello OCR",
                "width": 10,
                "height": 8,
                "blocks": [{"text": "Hello OCR", "lines": [{"text": "Hello OCR", "tokens": [{"text": "Hello"}, {"text": "OCR"}]}]}]
            }),
        })
        .expect("document summary");
        assert_eq!(response.value["blocks"], 1);
        assert_eq!(response.value["tokens"], 2);
    }

    #[test]
    fn document_summary_rejects_empty_token_text() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.ocr.documentSummary"),
            input: serde_json::json!({
                "text": "Hello OCR",
                "width": 10,
                "height": 8,
                "blocks": [{"text": "Hello OCR", "lines": [{"text": "Hello OCR", "tokens": [{"text": ""}]}]}]
            }),
        })
        .expect_err("empty token");
        assert!(error.contains("token text"));
    }
}
