# image-analysis-segmentation

Image segmentation primitives and SAM model defaults for `moritzbrantner-video-analysis`.

## Feature flags

- No optional feature flags today.

## Runtime Surface

- Workflow operations: `image.segmentation.maskSummary` summarizes raw or
  rectangle-built masks.
- Debug operations: `image.segmentation.model`, `image.segmentation.models`,
  `image.segmentation.promptSummary`, and `describe` inspect model metadata,
  prompt controls, and package metadata.
- The surface does not download models or run SAM.

## Backend execution contract

Use `segment_image_with_backend` as the standard Rust execution seam for an
`ImageSegmentationBackend`. The helper keeps request/result policy in this
package instead of duplicating it across model backends: manual prompts must be
present and inside the source image, automatic-mask generation cannot be mixed
with explicit point/box prompts, returned masks must match the source image and
their declared bounds, `min_mask_pixels` is enforced centrally, and a backend
must not return multiple accepted masks when `multimask_output` is disabled.

This does not add a built-in native SAM runtime. Model loading, inference, and
runtime-specific policy remain backend concerns; the existing SAM model catalog
continues to describe presets without downloading or executing them.

## Example

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use image_analysis_segmentation::{
    ImageSegmentationPrompt, ImageSegmentationRequest, SegmentationPoint,
};

let prompt = ImageSegmentationPrompt::new()
    .point(SegmentationPoint::foreground(200, 120))
    .multimask_output(false);

let request = ImageSegmentationRequest::new(prompt);
let _ = request.min_mask_pixels(32);
# Ok(())
# }
```

## Related crates

- `image-analysis-core`
- `image-analysis-detection`
- `image-analysis-classification, image-analysis-embeddings, image-analysis-captioning, image-analysis-ocr, image-analysis-segmentation, or image-analysis-detection`
