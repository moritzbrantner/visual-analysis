# image-analysis-detection

Object detection built from native ONNX inference, image segmentation masks, and deterministic color-blob detection.

## Feature flags

- `onnx`: enables native ONNX Runtime execution for learned detectors.
- `external-tests`: enables ignored semantic acceptance tests and implies `onnx`.

## Runtime Surface

- Primary workflow: `image.detection.detect` runs the `Xenova/detr-resnet-50` ONNX preset against an in-memory image payload or `imagePath`. `autoDownload` remains opt-in and model bundles default to `.model-runtime`.
- Workflow operation `image.detection.colorBlob` keeps the deterministic in-memory red-blob detector as a lightweight fallback.
- Debug operations: `image.detection.models`, `image.detection.boxSummary`, and `describe` inspect model metadata and imported boxes without running a detector.
- CLI and server adapters enable `onnx`. WASM keeps deterministic/imported workflows but does not execute DETR.
- YuNet face-detection primitives remain available at the library level but are not promoted to the standard runtime surface in this slice.

The semantic external test runs the standard `image.detection.detect` surface against a pinned Hugging Face COCO fixture and requires high-confidence cat detections on both sides of the image.

## Related crates

- `image-analysis-core`
- `image-analysis-io`
- `image-analysis-processing`
- `image-analysis-segmentation`
- `model-runtime`
- `runtime-onnx`
