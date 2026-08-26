# image-analysis-ocr

OCR model presets, rich text outputs, and backend contracts for `moritzbrantner-video-analysis`.

## Feature flags

- `local-onnx`: enables native TrOCR ONNX execution through `runtime-onnx` and `tokenizers`.
- `external-tests`: enables ignored semantic acceptance tests that use a real TrOCR ONNX bundle.

## Runtime Surface

- Primary workflow: `image.ocr.recognize` runs native TrOCR for the printed or handwritten Xenova ONNX presets. It accepts either an in-memory image payload or `imagePath`; `autoDownload` remains opt-in and model bundles default to `.model-runtime`.
- Xenova ONNX presets require the encoder/decoder ONNX files and tokenizer/config assets only; PyTorch `model.safetensors`/`pytorch_model.bin` weights are not part of those bundle requirements.
- Workflow operation `image.ocr.toTextDocument` converts OCR output into a `text_core::TextDocumentContract`.
- Debug operations: `image.ocr.models`, `image.ocr.presets`, `image.ocr.requestSummary`, `image.ocr.documentSummary`, and `describe` inspect model availability, request options, imported documents, and package metadata.
- CLI and server adapters enable `local-onnx`. WASM does not execute TrOCR.

The semantic external test runs the standard `image.ocr.recognize` surface against `tests/fixtures/ocr/trocr-hello.png` and requires the recognized text to contain `HELLO`.

## Related crates

- `image-analysis-core`
- `image-analysis-io`
- `model-runtime`
- `runtime-onnx`
- `text-core`
