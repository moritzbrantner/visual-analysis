# video-analysis-transform

Filtering, joins, grouping, and resampling for `moritzbrantner-video-analysis` datasets.

## Feature flags

- No optional feature flags today.

## Example

```rust,ignore
use video_analysis_transform::DatasetTransformPipeline;

let pipeline = DatasetTransformPipeline::default();
let _ = pipeline;
```

## Package surface

Primary workflow: `video.transform.filter`.

Workflow operations:

- `video.transform.filter`: Filters retained dataset records by kind, time, scene, label, and capped result size.
- `video.transform.window`: Groups records into fixed-duration time windows.
- `video.transform.groupScenes`: Groups records under scene intervals using explicit scene indexes or frame bounds.
- `video.transform.resampleFeatures`: Aggregates numeric feature records into fixed time windows.

Debug operations:

- `describe`: inspect package metadata and runtime support.

Runtime support: library, CLI, server, and WASM wrappers expose these operations.

Run the primary workflow through the CLI:

```bash
cargo run -p moritzbrantner-video-analysis-transform-cli -- run \
  --operation video.transform.filter \
  --json '{"dataset":{"metadata":{"attributes":{},"created_at":null,"name":null,"schema_version":2,"source":null},"records":[]},"kinds":["observation"],"limit":1000}'
```

Successful responses use the shared package-surface shape with `operation`,
`title`, `message`, `summary`, and `result`. Default surface calls are
deterministic, local-first, and do not download models, write persistent files,
or execute external tools unless an operation explicitly documents native or
external-tool execution.

## Related crates

- `video-analysis-dataset`
- `video-analysis-features`
