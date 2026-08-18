# image-analysis-processing

CPU image processing primitives for `moritzbrantner-video-analysis`.

## Runtime Surface

- Workflow operations: `image.processing.apply`, `image.processing.pipeline`,
  `image.processing.composite`, and `image.processing.hash`.
- Debug operations: `describe` inspects package metadata and operation support.
- Operations process in-memory image JSON and return capped previews or hashes;
  they do not read files or use external image tools.
- `image.processing.apply` and `image.processing.pipeline` support crop,
  resize, blur, grayscale, invert, brightness/contrast, saturation, threshold,
  flips, 90-degree rotation, sharpen, blue noise, and Poisson noise.

## Feature flags

- No optional feature flags today.

## Example

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use image_analysis_processing::{ImageOperation, ImageProcessor};

let processor = ImageProcessor::new()
    .operation(ImageOperation::Grayscale)
    .operation(ImageOperation::BlueNoise { amount: 8, seed: 42 })
    .operation(ImageOperation::Threshold { level: 110 });

let _ = processor;
# Ok(())
# }
```

## Related crates

- `image-analysis-core`
- `video-analysis-editing`
