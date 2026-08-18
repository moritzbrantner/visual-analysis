# vision-core

Shared visual detection, keypoint, embedding, and identity-match contracts for
image and video packages.

The crate owns deterministic contract types only. Model execution, image/video
loading, tracking, and reference-library matching live in specialized crates.

## Package Surface

`moritzbrantner-vision-core` owns the standard runtime surface used by its CLI,
server, WASM, and app companions. The surface accepts inline visual contract
payloads and exposes:

- `describe`
- `vision.validateDetection`
- `vision.validateEmbedding`
- `vision.validateIdentityMatch`
- `vision.convertDetectionSummary`

Transport packages:

- `moritzbrantner-vision-core-cli`
- `moritzbrantner-vision-core-server`
- `crates/bindings/vision-core-wasm`
- `packages/vision-core-wasm`
- `packages/vision-core-app`

Operations are deterministic, in-memory, and side-effect free. They validate or
summarize caller-provided JSON and do not download models, read media, run
native inference, or write files.
