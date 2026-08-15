# video-analysis-tracking

IoU-based object tracking for `moritzbrantner-video-analysis`.

## Feature flags

- No optional feature flags today.

## Example

```rust,ignore
use video_analysis_core::BoundingBox;
use video_analysis_tracking::{
    detect_detection_collisions, detect_detection_collisions_with_broad_phase,
    CollisionBroadPhaseOptions, CollisionBroadPhaseStrategy, CollisionCellSize, CollisionOptions,
    IouTracker, TrackedDetection,
};

let detections = [
    TrackedDetection::new(BoundingBox::new(0, 0, 32, 32)?),
    TrackedDetection::new(BoundingBox::new(16, 16, 32, 32)?),
];

let collisions = detect_detection_collisions(&detections, CollisionOptions::default())?;
assert_eq!(collisions.len(), 1);

let optimized = detect_detection_collisions_with_broad_phase(
    &detections,
    CollisionOptions::default(),
    CollisionBroadPhaseOptions {
        strategy: CollisionBroadPhaseStrategy::SpatialHashGrid,
        cell_size: CollisionCellSize::Fixed {
            width: 32,
            height: 32,
        },
        ..CollisionBroadPhaseOptions::default()
    },
)?;
assert_eq!(optimized, collisions);

let _tracker = IouTracker::new(Default::default())?;
```

## Package surface

Primary workflow: `video.tracking.trackSummary`.

Workflow operations:

- `video.tracking.trackSummary`: Runs IoU tracking over frame detections and summarizes resulting tracks.
- `video.tracking.smoothPath`: Applies deterministic moving-average smoothing to ordered track points.

Debug operations:

- `describe`: inspect package metadata and runtime support.
- `video.tracking.assignmentPlan`: Builds a deterministic IoU association plan between previous and next detections.

Runtime support: library, CLI, server, and WASM wrappers expose these operations.

Run the primary workflow through the CLI:

```bash
cargo run -p moritzbrantner-video-analysis-tracking-cli -- run \
  --operation video.tracking.trackSummary \
  --json '{"frames":[{"detections":[{"label":"person","region":{"height":8,"width":8,"x":0,"y":0}}],"frameIndex":0},{"detections":[{"label":"person","region":{"height":8,"width":8,"x":1,"y":0}}],"frameIndex":1}],"options":{"maxMissedFrames":15,"minIou":0.3}}'
```

Successful responses use the shared package-surface shape with `operation`,
`title`, `message`, `summary`, and `result`. Default surface calls are
deterministic, local-first, and do not download models, write persistent files,
or execute external tools unless an operation explicitly documents native or
external-tool execution.

## Related crates

- `video-analysis-recognition`
- `video-analysis-posture`
