# image-analysis-core

Shared image views, pixel formats, and image statistics for `moritzbrantner-video-analysis`.

## Runtime Surface

- Workflow operations: `image.core.summary`, `image.core.lumaHistogram`, and
  `image.core.maskTensorSummary`.
- Debug operations: `describe` inspects package metadata and operation support.
- Operations use in-memory image JSON only; they do not read files, decode image
  formats, or run external tools.
- `ImagePayload` is the reusable Runtime Surface image payload contract for
  crates that accept in-memory image JSON.

## Feature flags

- No optional feature flags today.

## Example

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use image_analysis_core::{luma_histogram, mean_rgb, ImageView, OwnedImage};

let image = OwnedImage::new_rgb(1920, 1080, vec![0; 1920 * 1080 * 3])?;
let view: ImageView<'_> = image.as_view();

let _ = mean_rgb(&view)?;
let _ = luma_histogram(&view, 16)?;
# Ok(())
# }
```

`ImagePayload` keeps the serde-facing JSON contract stable while providing a
validated `ImageView` for library code:

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use image_analysis_core::{ImagePayload, ImagePixelFormat};

let payload = ImagePayload::new(2, 2, ImagePixelFormat::Rgb24, vec![0; 12]);
let view = payload.view()?;

assert_eq!(view.stride, 6);
# Ok(())
# }
```

## Related crates

- `image-analysis-processing`
- `image-analysis-segmentation`
- `video-analysis-core`
