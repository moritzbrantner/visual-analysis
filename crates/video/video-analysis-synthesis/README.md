# video-analysis-synthesis

Deterministic storyboard and frame synthesis for `moritzbrantner-video-analysis`.

## Feature flags

- No optional feature flags today.

## Example

```rust,ignore
use video_analysis_synthesis::Storyboard;

let storyboard = Storyboard::default();
let _ = storyboard.render()?;
```

## Package surface

Primary workflow: `video.synthesis.framePlan`.

Workflow operations:

- `video.synthesis.framePlan`: Generates one deterministic synthetic frame and returns capped pixel preview metadata.
- `video.synthesis.renderSummary`: Summarizes generated frame dimensions, pixel counts, and capped preview pixels.

Debug operations:

- `describe`: inspect package metadata and runtime support.
- `video.synthesis.overlayPlan`: Builds deterministic storyboard frames from observations and reports inferred overlay regions.

Runtime support: library, CLI, server, and WASM wrappers expose these operations.

Run the primary workflow through the CLI:

```bash
cargo run -p moritzbrantner-video-analysis-synthesis-cli -- run \
  --operation video.synthesis.framePlan \
  --json '{"background":{"blue":20,"green":18,"red":16},"config":{"fpsDen":1,"fpsNum":30,"height":16,"width":16},"frameIndex":0,"previewLimit":8,"regions":[{"color":{"blue":0,"green":0,"red":255},"region":{"height":4,"width":4,"x":2,"y":2}}]}'
```

Successful responses use the shared package-surface shape with `operation`,
`title`, `message`, `summary`, and `result`. Default surface calls are
deterministic, local-first, and do not download models, write persistent files,
or execute external tools unless an operation explicitly documents native or
external-tool execution.

## Related crates

- `video-analysis-core`
- `data-inversion-core`
