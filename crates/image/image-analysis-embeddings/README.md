# image-analysis-embeddings

Concrete image and face embedding contracts and presets for `moritzbrantner-video-analysis`.

Model download and bundle handling belong to `moritzbrantner-model-runtime`; this crate owns embedding values, embedder backend traits, model-facing preprocessing, and embedding catalog metadata.

## Runtime Surface

- `image.embeddings.faceEmbed` runs the OpenCV SFace ONNX preset against an in-memory image or `imagePath`. A pixel-space `region` may be paired with the five normalized YuNet landmarks; when landmarks are present the face is similarity-aligned to OpenCV SFace’s canonical 112×112 reference before feature extraction. Region-only requests retain the compatibility crop path, and omitting both embeds the full image.
- `autoDownload` remains opt-in and model bundles default to `.model-runtime`.
- CLI and server adapters enable the `onnx` feature and can execute SFace natively. WASM keeps catalog/validation workflows only and reports `faceEmbed` as unsupported.
- `image.embeddings.validate` validates imported image or face vectors and optional face regions.
- Debug operations `image.embeddings.models`, `image.embeddings.schema`, and `describe` inspect catalogs, schemas, and package metadata.

Face vectors are biometric-derived evidence. Callers should persist them only where their data policy permits and should not infer verified real-world identity from embedding similarity alone.

## SFace alignment contract

OpenCV SFace recognition is defined over an aligned five-landmark face, not a stretched bounding-box crop. The supported landmark order is the YuNet/OpenCV order: right eye, left eye, nose tip, right mouth corner, left mouth corner. Surface landmarks are normalized to the source image and must be supplied together with the pixel-space region from the same detection. The response reports the applied preprocessing mode in `alignment` and `attributes.facePreprocessing`.
