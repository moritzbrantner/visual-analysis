# video-analysis-posture-io

COCO-style posture JSON and 3D stick-figure export helpers for `moritzbrantner-video-analysis`.

## Feature flags

- No optional feature flags today.

## Example

```rust,ignore
use video_analysis_posture_io::{read_coco_keypoints_json, write_stick_figure_gltf};

let poses = read_coco_keypoints_json("poses.json")?;
let figure = poses[0].to_stick_figure_3d()?;
write_stick_figure_gltf("pose.gltf", &figure)?;
```

## Related crates

- `video-analysis-posture`
- `three-d-processing-io`

## Package surface

Workflow operations:

- `video.postureIo.formatSummary`

Debug operations:

- `describe`
- `video.postureIo.parsePlan`
- `video.postureIo.exportPlan`

Runtime limits:

Operations are deterministic, local-first, and side-effect free. They return inline JSON reports and do not download models, write files, or run native tools.

Invalid input returns a clear error through `run_surface_operation`; successful
responses include `operation`, `title`, `message`, `summary`, and `result` while
keeping existing top-level domain fields for compatibility.
