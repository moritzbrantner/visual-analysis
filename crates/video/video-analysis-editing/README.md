# video-analysis-editing

CPU video frame editing primitives for `moritzbrantner-video-analysis`.

## Feature flags

- No optional feature flags today.

## Example

```rust,ignore
use video_analysis_editing::{FrameEdit, FrameEditor};

let editor = FrameEditor::new(vec![FrameEdit::Grayscale, FrameEdit::Invert]);
let _ = editor;
```

## Related crates

- `video-analysis-core`
- `image-analysis-processing`

## Package surface

Workflow operations:

- `video.editing.cutPlan`
- `video.editing.frameApply`

Debug operations:

- `describe`
- `video.editing.concatPlan`
- `video.editing.subtitlePlan`

Runtime limits:

Operations are deterministic, local-first, and side-effect free. They return inline JSON reports and do not download models, write files, or run native tools.

Invalid input returns a clear error through `run_surface_operation`; successful
responses include `operation`, `title`, `message`, `summary`, and `result` while
keeping existing top-level domain fields for compatibility.
