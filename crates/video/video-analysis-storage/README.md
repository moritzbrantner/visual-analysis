# video-analysis-storage

Dataset persistence for `moritzbrantner-video-analysis`.

## Feature flags

- No optional feature flags today.

## Example

```rust,ignore
use video_analysis_storage::{load_dataset_dir, write_dataset_dir};

write_dataset_dir("output/report", &Default::default())?;
let dataset = load_dataset_dir("output/report")?;

let _ = dataset;
```

## Package surface

Primary workflow: `video.storage.manifestPlan`.

Workflow operations:

- `video.storage.manifestPlan`: Builds a dataset manifest preview without writing files.
- `video.storage.jsonlPreview`: Serializes a capped preview of dataset records as JSON lines without writing files.

Debug operations:

- `describe`: inspect package metadata and runtime support.

Runtime support: library, CLI, server, and WASM wrappers expose these operations.

Run the primary workflow through the CLI:

```bash
cargo run -p moritzbrantner-video-analysis-storage-cli -- run \
  --operation video.storage.manifestPlan \
  --json '{"dataset":{"metadata":{"attributes":{},"created_at":null,"name":null,"schema_version":2,"source":null},"records":[]},"recordsPath":"records.jsonl"}'
```

Successful responses use the shared package-surface shape with `operation`,
`title`, `message`, `summary`, and `result`. Default surface calls are
deterministic, local-first, and do not download models, write persistent files,
or execute external tools unless an operation explicitly documents native or
external-tool execution.

## Related crates

- `video-analysis-dataset`
- `video-analysis-output`
