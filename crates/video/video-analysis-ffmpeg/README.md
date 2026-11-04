# video-analysis-ffmpeg

FFmpeg-backed media ingest for `moritzbrantner-video-analysis`.

## Feature flags

- `ffmpeg-backend`: enables the backend surface
- `ffmpeg-command`: keeps the process-backed `ffmpeg`/`ffprobe` runtime enabled
- `ffmpeg-native`: enables native runtime API selection without requiring system FFmpeg development packages
- `ffmpeg-next-bindings`: reserved for future `ffmpeg-next`/system FFmpeg probing support
- `ffmpeg-tests`: enables decode-oriented tests
- `external-tests`: extends ignored external coverage
- `test-utils`: utilities for integration tests

## Example

```rust,ignore
use video_analysis_ffmpeg::{
    probe_streams, FfmpegAudioSource, FfmpegAudioSourceOptions,
};

let inventory = probe_streams("video.mkv")?;
let source = FfmpegAudioSource::open_path_with_options_checked(
    "video.mkv",
    FfmpegAudioSourceOptions::recorded().audio_stream_index(1),
)?;
```

`probe_streams` returns every stream with its global container index and, for
audio, its zero-based audio ordinal. Explicit audio selection is validated
against that inventory before FFmpeg decode starts. Omitting the selector keeps
the existing first-audio-stream behavior.

### Exact timestamps for recorded VFR media

For local recorded video, use
`FfmpegVideoSource::open_path_with_exact_frame_pts(path, FfmpegSourceOptions::recorded())`.
This opt-in mode probes the selected FFprobe video stream's **integer frame PTS,
original timebase, and last-frame packet duration**. Decoded frames carry
those positions, and `detect_content_scenes` uses their exclusive final end
rather than adding one PTS tick. Existing callers keep the fast frame-index/FPS
timing behavior and require no schema or dependency migration.

The exact mode performs a separate FFprobe frame scan before decoding, so it
costs extra CPU and memory. It fails when exact PTS, final packet duration, or
decoded frame-count parity is unavailable; it does not reconstruct PTS from
floating seconds. Live inputs, input URLs, and custom frame-changing FFmpeg
arguments are not supported in this mode. It reports **stream PTS**, which can
require an explicit exact offset mapping before consumers call it media-relative.

## Related crates

- `video-analysis-ingest`
- `audio-analysis-io`
- `video-analysis-cli`

## Package surface

Workflow operations:

- `video.ffmpeg.probePlan`

Debug operations:

- `describe`
- `video.ffmpeg.extractFramesPlan`
- `video.ffmpeg.filterGraphPlan`

Runtime limits:

The package surface is side-effect free. FFmpeg probing, decode, and extraction stay behind the crate feature flags and command/native runtime paths described above.

Invalid input returns a clear error through `run_surface_operation`; successful
responses include `operation`, `title`, `message`, `summary`, and `result` while
keeping existing top-level domain fields for compatibility.
