//! Detections derived from segmentation mask proposals.

use image_analysis_core::ImageView;
use image_analysis_segmentation::{
    segment_image_with_backend, ImageSegment, ImageSegmentationBackend, ImageSegmentationRequest,
};
use video_analysis_core::{Result, VideoFrame};

use crate::{FrameDetections, ImageDetection};

#[derive(Debug, Clone, PartialEq, Eq)]
/// Data type for image detection request.
pub struct ImageDetectionRequest {
    /// The segmentation value.
    pub segmentation: ImageSegmentationRequest,
    /// The min mask pixels value.
    pub min_mask_pixels: usize,
    /// The default label value.
    pub default_label: String,
}

impl ImageDetectionRequest {
    /// Returns automatic mask proposals.
    pub fn automatic_mask_proposals() -> Self {
        Self {
            segmentation: ImageSegmentationRequest::automatic_mask_generation(),
            ..Self::default()
        }
    }

    /// Returns min mask pixels.
    pub fn min_mask_pixels(mut self, value: usize) -> Self {
        self.min_mask_pixels = value.max(1);
        self
    }

    /// Returns default label.
    pub fn default_label(mut self, value: impl Into<String>) -> Self {
        self.default_label = value.into();
        self
    }
}

impl Default for ImageDetectionRequest {
    fn default() -> Self {
        Self {
            segmentation: ImageSegmentationRequest::default(),
            min_mask_pixels: 1,
            default_label: "object".to_string(),
        }
    }
}

/// Returns segment to detection.
pub fn segment_to_detection(segment: &ImageSegment, default_label: &str) -> ImageDetection {
    ImageDetection {
        label: segment
            .label
            .clone()
            .unwrap_or_else(|| default_label.to_string()),
        score: segment.score,
        region: segment.region,
        attributes: segment.attributes.clone(),
    }
}

/// Returns segments to detections.
pub fn segments_to_detections(
    segments: &[ImageSegment],
    min_mask_pixels: usize,
    default_label: &str,
) -> Vec<ImageDetection> {
    segments
        .iter()
        .filter(|segment| segment.mask.active_pixels() >= min_mask_pixels.max(1))
        .map(|segment| segment_to_detection(segment, default_label))
        .collect()
}

/// Data type for mask proposal detector.
pub struct MaskProposalDetector<B> {
    backend: B,
    request: ImageDetectionRequest,
}

impl<B> MaskProposalDetector<B> {
    /// Creates a new value.
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            request: ImageDetectionRequest::automatic_mask_proposals(),
        }
    }

    /// Returns request.
    pub fn request(mut self, value: ImageDetectionRequest) -> Self {
        self.request = value;
        self
    }

    /// Returns backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Returns backend mut.
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }
}

impl<B: ImageSegmentationBackend> MaskProposalDetector<B> {
    /// Returns detect image.
    pub fn detect_image(&mut self, image: &ImageView<'_>) -> Result<Vec<ImageDetection>> {
        let segments = segment_image_with_backend(
            &mut self.backend,
            image,
            &self.request.segmentation,
        )?;
        Ok(segments_to_detections(
            &segments,
            self.request.min_mask_pixels,
            &self.request.default_label,
        ))
    }

    /// Returns detect frame.
    pub fn detect_frame(&mut self, frame: &VideoFrame<'_>) -> Result<FrameDetections> {
        let image = ImageView::from_video_frame(frame)?;
        let detections = self.detect_image(&image)?;
        Ok(FrameDetections {
            position: frame.position,
            detections,
        })
    }
}
