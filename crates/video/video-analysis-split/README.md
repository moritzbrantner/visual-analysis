# video-analysis-split

FFmpeg-backed scene splitting utilities for `moritzbrantner-video-analysis`.

## Feature flags

- `external-tests`: enables ignored FFmpeg split tests
- `ffmpeg-native`: exposes the native split executor type; the command executor
  remains the compatibility default

## Example

```rust,ignore
use video_analysis_split::{build_split_plan, CommandFfmpegSplitExecutor, SplitOptions};
use video_analysis_core::Scene;

let scenes: Vec<Scene> = Vec::new();
let plan = build_split_plan("video.mp4", &scenes, &SplitOptions::default()).unwrap();
assert!(plan.jobs.is_empty());
let _executor = CommandFfmpegSplitExecutor;
```

## Package surface

Primary workflow: `video.split.scenePlan`.

Workflow operations:

- `video.split.scenePlan`: Builds an in-memory split plan from scene intervals without executing FFmpeg.
- `video.split.segmentManifest`: Summarizes segment names, time ranges, frame ranges, and output paths.

Debug operations:

- `describe`: inspect package metadata and runtime support.
- `video.split.namingPlan`: Builds deterministic file names for split outputs and reports collisions.

Runtime support: library, CLI, server, and WASM wrappers expose these operations.

Run the primary workflow through the CLI:

```bash
cargo run -p moritzbrantner-video-analysis-split-cli -- run \
  --operation video.split.scenePlan \
  --json '{"inputVideoPath":"demo.mp4","options":{"outputDir":"segments","template":"$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4"},"scenes":[{"endFrame":30,"fpsDen":1,"fpsNum":30,"startFrame":0}]}'
```

Successful responses use the shared package-surface shape with `operation`,
`title`, `message`, `summary`, and `result`. Default surface calls are
deterministic, local-first, and do not download models, write persistent files,
or execute external tools unless an operation explicitly documents native or
external-tool execution.

## Related crates

- `video-analysis-detectors`
- `video-analysis-ffmpeg`
- `video-analysis-cli`
