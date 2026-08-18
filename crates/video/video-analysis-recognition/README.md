# video-analysis-recognition

Reference-embedding recognition helpers plus video model prediction contracts
for `moritzbrantner-video-analysis`.

## Feature flags

- No optional feature flags today.

## Example

```rust,ignore
use video_analysis_recognition::RecognitionLibrary;

let library = RecognitionLibrary::default();
let _ = library;
```

## Package surface

Primary workflow: `video.recognition.labelPlan`.

Workflow operations:

- `video.recognition.labelPlan`: Builds a reference library and searches candidate embeddings against it.
- `video.recognition.trackSummary`: Summarizes recognized labels over tracks and confidence acceptance thresholds.
- `video.recognition.confidence`: Summarizes confidence score ranges and threshold acceptance counts.

Debug operations:

- `describe`: inspect package metadata and runtime support.

Runtime support: library, CLI, server, and WASM wrappers expose these operations.

Run the primary workflow through the CLI:

```bash
cargo run -p moritzbrantner-video-analysis-recognition-cli -- run \
  --operation video.recognition.labelPlan \
  --json '{"candidates":[{"embedding":[1.0,0.0],"id":"candidate-1","kind":"face"}],"identities":[{"embeddings":[[1.0,0.0]],"id":"alice","kind":"face","label":"Alice"}],"options":{"maxResults":2,"minScore":0.5}}'
```

Successful responses use the shared package-surface shape with `operation`,
`title`, `message`, `summary`, and `result`. Default surface calls are
deterministic, local-first, and do not download models, write persistent files,
or execute external tools unless an operation explicitly documents native or
external-tool execution.

## Related crates

- `video-analysis-tracking`
- `model-runtime`
- `runtime-onnx` through the optional `onnx` feature
