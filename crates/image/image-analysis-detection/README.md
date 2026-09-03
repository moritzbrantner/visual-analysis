# image-analysis-detection

Object and face detection built from native ONNX inference, image segmentation masks, and deterministic color-blob detection.

## Feature flags

- `onnx`: enables native ONNX Runtime execution for learned detectors.
- `external-tests`: enables ignored semantic acceptance tests and implies `onnx`.

## Runtime Surface

- `image.detection.detect` runs the `Xenova/detr-resnet-50` ONNX preset against an in-memory image payload or `imagePath`.
- `image.detection.detectFaces` runs the OpenCV YuNet ONNX face detector against the same input contract. Face results expose pixel `region` values plus a `normalizedRegion` and normalized landmarks so downstream video/corpus adapters can preserve an explicit coordinate space.
- Model downloads remain opt-in through `autoDownload`; model bundles default to `.model-runtime`.
- `image.detection.colorBlob` keeps the deterministic in-memory red-blob detector as a lightweight object-detection fallback.
- Debug operations `image.detection.models`, `image.detection.boxSummary`, and `describe` inspect model metadata and imported boxes without running a detector.
- CLI and server adapters enable `onnx`. WASM keeps deterministic/imported workflows but does not execute DETR or YuNet.

The semantic external object-detection test runs `image.detection.detect` against a pinned Hugging Face COCO fixture and requires high-confidence cat detections on both sides of the image. YuNet model execution remains local/native and is covered structurally by the runtime-surface and ONNX decoding tests unless an explicit external face fixture is supplied.

## Related crates

- `image-analysis-core`
- `image-analysis-io`
- `image-analysis-processing`
- `image-analysis-segmentation`
- `model-runtime`
- `runtime-onnx`
