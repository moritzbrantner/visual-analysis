# video-analysis-output

CSV, HTML, JSON, EDL, FCP, OTIO, and qpfile report helpers for
`moritzbrantner-video-analysis`.

## Feature flags

- No optional feature flags today.

## Example

```rust,ignore
use video_analysis_output::{write_scene_list_csv, write_scene_list_otio};

let mut bytes = Vec::new();
write_scene_list_csv(&mut bytes, &[])?;
write_scene_list_otio(&mut bytes, &[])?;

let _ = bytes;
```

## Package surface

Primary workflow: `video.output.reportSummary`.

Workflow operations:

- `video.output.reportSummary`: Summarizes detection result scenes, cuts, metrics, and JSON report size.

Debug operations:

- `describe`: inspect package metadata and runtime support.
- `video.output.csvPlan`: Renders scene and optional stats CSV previews in memory without writing files.
- `video.output.htmlPlan`: Renders an HTML scene list preview in memory without writing files.
- `video.output.editListPlan`: Renders EDL, FCP7 XML, FCPXML, OTIO JSON, or qpfile scene-list previews in memory without writing files.

Runtime support: library, CLI, server, and WASM wrappers expose these operations.

Run the primary workflow through the CLI:

```bash
cargo run -p moritzbrantner-video-analysis-output-cli -- run \
  --operation video.output.reportSummary \
  --json '{"result":{"cuts":[{"detector":"content","frameIndex":30,"score":0.9}],"framesProcessed":30,"metrics":[{"frameIndex":0,"values":{"content":0.1}}],"scenes":[{"endFrame":30,"startFrame":0}]}}'
```

Successful responses use the shared package-surface shape with `operation`,
`title`, `message`, `summary`, and `result`. Default surface calls are
deterministic, local-first, and do not download models, write persistent files,
or execute external tools unless an operation explicitly documents native or
external-tool execution.

## Related crates

- `video-analysis-core`
- `video-analysis-features`
- `@moritzbrantner/video-analysis-ui`
