# video-analysis-posture

Pose and skeleton helpers for `moritzbrantner-video-analysis`.

## Feature flags

- No optional feature flags today.

## Example

```rust,ignore
use video_analysis_posture::{JointName, Skeleton};

let skeleton = Skeleton::default();
let _ = skeleton.joint(JointName::Head);
```

## Related crates

- `video-analysis-recognition`
- `video-analysis-tracking`

## Package surface

Workflow operations:

- `video.posture.keypointSummary`

Debug operations:

- `describe`
- `video.posture.skeletonPlan`
- `video.posture.motionSummary`

Runtime limits:

Operations are deterministic, local-first, and side-effect free. They return inline JSON reports and do not download models, write files, or run native tools.

Invalid input returns a clear error through `run_surface_operation`; successful
responses include `operation`, `title`, `message`, `summary`, and `result` while
keeping existing top-level domain fields for compatibility.
