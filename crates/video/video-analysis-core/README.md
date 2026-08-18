# video-analysis-core

Core media, timing, runtime surface, detection, and analyzer contracts for
`moritzbrantner-video-analysis`.

Runtime DTOs used by package surfaces live in `runtime_core` and are re-exported under `video_analysis_core::runtime`,
including diagnostics, capabilities, package surfaces, surface requests,
surface responses, and lightweight job/artifact identifiers.

## Feature flags

- No optional feature flags today.

## Example

```rust,ignore
use video_analysis_core::{ScenePipeline, ThresholdDetector};

let detector = ThresholdDetector::new(27.0)?;
let mut pipeline = ScenePipeline::new(detector);

// Feed frames from an ingest crate, then finish the run.
let result = pipeline.finish_detection()?;
let _ = result.scenes;
```

## Related crates

- `video-analysis-detectors`
- `video-analysis-ingest`
- `video-analysis-dataset`

## Package surface

Workflow operations:

- `video.core.frameSummary`
- `video.core.timecode`
- `video.core.sceneSummary`

Debug operations:

- `describe`

Runtime limits:

Operations are deterministic, local-first, and side-effect free. They return
inline JSON reports and do not download models, write files, or run native
tools. Timecode operations convert frames, seconds, display timecodes, and
timestamps. Frame summaries validate dimensions, pixel formats, strides, and
buffer sizes. Scene summaries compute counts, cuts, timeline bounds, and
per-scene durations from frame ranges.

Invalid input returns typed `runtime_core::SurfaceError` JSON through
`run_surface_operation`; successful responses include `operation`, `title`,
`message`, `summary`, and `result` while keeping existing top-level domain
fields for compatibility.
