# video-analysis-features

Feature extraction over retained `moritzbrantner-video-analysis` datasets.

## Feature flags

- No optional feature flags today.

## Example

```rust,ignore
use video_analysis_features::scene_duration_summary;

let _ = scene_duration_summary(&[]);
```

## Package surface

Primary workflow: `video.features.extract`.

Workflow operations:

- `video.features.extract`: Runs selected deterministic feature extractors over a retained dataset.
- `video.features.timelineSummary`: Summarizes retained dataset record counts and timestamp coverage before feature extraction.
- `video.features.aggregate`: Runs selected deterministic feature extractors and summarizes output feature names, scopes, and value kinds.

Debug operations:

- `describe`: inspect package metadata and runtime support.

Runtime support: library, CLI, server, and WASM wrappers expose these operations.

Run the primary workflow through the CLI:

```bash
cargo run -p moritzbrantner-video-analysis-features-cli -- run \
  --operation video.features.extract \
  --json '{"dataset":{"metadata":{"attributes":{},"created_at":null,"name":null,"schema_version":2,"source":null},"records":[]},"extractors":["scene_stats","observation_label_histogram","transcript_stats","audio_event_stats","track_summary","feature_vector_mean"]}'
```

Successful responses use the shared package-surface shape with `operation`,
`title`, `message`, `summary`, and `result`. Default surface calls are
deterministic, local-first, and do not download models, write persistent files,
or execute external tools unless an operation explicitly documents native or
external-tool execution.

## Related crates

- `video-analysis-dataset`
- `video-analysis-transform`
- `video-analysis-output`
