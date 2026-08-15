# image-analysis-ocr

OCR model presets, rich text outputs, and backend contracts for `moritzbrantner-video-analysis`.

## Feature flags

- `local-onnx`: enables native server/CLI TrOCR ONNX execution through
  `runtime-onnx` and `tokenizers`.
- `external-tests`: enables ignored smoke tests that require a local TrOCR
  ONNX model bundle under `.model-runtime`.

## Runtime Surface

- Workflow operations: `image.ocr.recognize` describes the server-only TrOCR
  ONNX execution path, and `image.ocr.toTextDocument` converts imported OCR
  output into a `text_core::TextDocumentContract`.
- Debug operations: `image.ocr.models`, `image.ocr.presets`,
  `image.ocr.requestSummary`, `image.ocr.documentSummary`, and `describe`
  inspect model availability, request options, imported documents, and package
  metadata.
- Default surface calls do not download models or run OCR. Native TrOCR
  execution requires constructing `OnnxTrOcrBackend` in a build compiled with
  `local-onnx`, and downloads remain opt-in through local model options.

## Example

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use image_analysis_ocr::{OcrPreset, OcrRequest};

let spec = OcrPreset::TrOcrBasePrinted.model_spec();
let onnx_spec = OcrPreset::TrOcrBasePrintedOnnx.model_spec();
let request = OcrRequest::new()
    .languages(["en"])
    .preserve_layout(true);

assert_eq!(spec.repo_id, "microsoft/trocr-base-printed");
assert_eq!(onnx_spec.repo_id, "Xenova/trocr-base-printed");
assert!(request.preserve_layout);
# Ok(())
# }
```

## Related crates

- `image-analysis-core`
- `image-analysis-classification, image-analysis-embeddings, image-analysis-captioning, image-analysis-ocr, image-analysis-segmentation, or image-analysis-detection`
- `model-runtime`
