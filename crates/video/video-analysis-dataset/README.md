# video-analysis-dataset

Serializable retained analysis records for `moritzbrantner-video-analysis`.

## Versioned timestamp interoperability

Dataset schema **v2** remains the default and keeps its existing flattened
`{pts, timebase_num, timebase_den, seconds}` timestamp records unchanged.
For interoperability with other Foundation consumers, explicitly call
`dataset.promote_to_canonical_v3()` before persisting. Schema **v3** then uses
the canonical `moenarch-media-core::Timestamp` JSON representation
`{pts, timebase: {num, den}}`; the `pts` wire and positive rational validation
come from Foundation, and derived `seconds` is never stored in v3.

Serialization and deserialization of `AnalysisDataset` are version-aware;
`video-analysis-storage` also writes/reads schema v3 timestamps in JSONL
using the dataset manifest version. Existing v2 JSON and JSONL remain readable.
Migration checks for conflicting legacy `seconds` values and rejects invalid
timebases rather than silently rounding or changing timestamp identity.

## Feature flags

- No optional feature flags today.

## Example

```rust,ignore
use video_analysis_dataset::AnalysisDataset;

let mut dataset = AnalysisDataset::default();
dataset.sort_records();

let _ = dataset;
```

## Package surface

Primary workflow: `video.dataset.summary`.

Workflow operations:

- `video.dataset.summary`: Summarizes retained dataset metadata, record counts, time range, and frame range.
- `video.dataset.recordsByKind`: Returns a capped record preview filtered by dataset record kind.

Debug operations:

- `describe`: inspect package metadata and runtime support.

Runtime support: library, CLI, server, and WASM wrappers expose these operations.

Run the primary workflow through the CLI:

```bash
cargo run -p moritzbrantner-video-analysis-dataset-cli -- run \
  --operation video.dataset.summary \
  --json '{"dataset":{"metadata":{"attributes":{},"created_at":null,"name":null,"schema_version":2,"source":null},"records":[]}}'
```

Successful responses use the shared package-surface shape with `operation`,
`title`, `message`, `summary`, and `result`. Default surface calls are
deterministic, local-first, and do not download models, write persistent files,
or execute external tools unless an operation explicitly documents native or
external-tool execution.

## Related crates

- `video-analysis-core`
- `video-analysis-storage`
- `video-analysis-transform`
