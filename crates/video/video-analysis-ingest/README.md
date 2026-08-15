# video-analysis-ingest

Media ingest traits and source adapters for `moritzbrantner-video-analysis`.

## Feature flags

- No optional feature flags today.

## Example

```rust,ignore
use video_analysis_ingest::{AudioFrameSource, TextSegmentSource, VideoFrameSource};

fn accept_sources(
    _video: &mut dyn VideoFrameSource,
    _audio: &mut dyn AudioFrameSource,
    _text: &mut dyn TextSegmentSource,
) {
}
```

## Related crates

- `video-analysis-core`
- `video-analysis-ffmpeg`
- `audio-analysis-io`

## Package surface

Workflow operations:

- `video.ingest.sourcePlan`
- `video.ingest.manifest`

Debug operations:

- `describe`
- `video.ingest.validate`

Runtime limits:

Operations are deterministic, local-first, and side-effect free. They return inline JSON reports and do not download models, write files, or run native tools.

Invalid input returns a clear error through `run_surface_operation`; successful
responses include `operation`, `title`, `message`, `summary`, and `result` while
keeping existing top-level domain fields for compatibility.
