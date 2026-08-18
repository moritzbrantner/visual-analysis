# image-analysis-segmentation

Image segmentation primitives and SAM model defaults for `moritzbrantner-video-analysis`.

## Feature flags

- No optional feature flags today.

## Runtime Surface

- Workflow operations: `image.segmentation.maskSummary` summarizes raw or
  rectangle-built masks.
- Debug operations: `image.segmentation.model`,
  `image.segmentation.promptSummary`, and `describe` inspect model metadata,
  prompt controls, and package metadata.
- The surface does not download models or run SAM.

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
