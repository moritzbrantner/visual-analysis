# image-analysis-io

Still-image PNG/JPEG/WebP loading and saving for `moritzbrantner-video-analysis`.

## Feature flags

- No optional feature flags today.

## Runtime Surface

- Workflow operations: `image.io.plan` returns read/write support for a path and
  inferred format.
- Debug operations: `image.io.supportedFormats`, `image.io.inferFormat`, and
  `describe` inspect format support and package metadata.
- Surface operations do not read or write files. Use `read_image` and
  `write_image` for actual filesystem I/O.

## Example

```rust,no_run
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use image_analysis_core::OwnedImage;
use image_analysis_io::{read_image, write_image};

let image = OwnedImage::new_rgb(2, 2, vec![255; 12])?;
write_image("preview.png", &image)?;
let roundtrip = read_image("preview.png")?;
assert_eq!(roundtrip.width, 2);
# Ok(())
# }
```

## Related crates

- `image-analysis-core`
- `image-analysis-processing`
