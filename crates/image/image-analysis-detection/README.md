# image-analysis-detection

Object detection built from image segmentation masks, color blob detection, and
SAM defaults for `moritzbrantner-video-analysis`.

## Feature flags

- No optional feature flags today.

## Runtime Surface

- Workflow operations: `image.detection.colorBlob` runs deterministic in-memory
  red blob detection.
- Debug operations: `image.detection.models`, `image.detection.boxSummary`, and
  `describe` inspect model metadata, imported boxes, and package metadata.
- `image.detection.models` does not download or run detector models.

## Example

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use image_analysis_detection::{ColorBlobDetector, ImageDetectionRequest, MaskProposalDetector};
use image_analysis_segmentation::ImageSegmentationBackend;
use image_analysis_core::ImageView;
use video_analysis_core::Result;

struct EmptyBackend;

impl ImageSegmentationBackend for EmptyBackend {
    fn segment_image(
        &mut self,
        _image: &ImageView<'_>,
        _request: &image_analysis_segmentation::ImageSegmentationRequest,
    ) -> Result<Vec<image_analysis_segmentation::ImageSegment>> {
        Ok(Vec::new())
    }
}

let mut detector = MaskProposalDetector::new(EmptyBackend)
    .request(ImageDetectionRequest::automatic_mask_proposals());
let image = image_analysis_core::OwnedImage::new_rgb(1, 1, vec![0, 0, 0])?;
let _ = detector.detect_image(&image.as_view())?;
let _ = ColorBlobDetector::red_car().detect_image(&image.as_view())?;
# Ok(())
# }
```

## Related crates

- `image-analysis-core`
- `image-analysis-segmentation`
- `image-analysis-classification, image-analysis-embeddings, image-analysis-captioning, image-analysis-ocr, image-analysis-segmentation, or image-analysis-detection`
