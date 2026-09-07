#![doc = include_str!("../README.md")]

pub mod surface;
use std::collections::BTreeMap;

use image_analysis_core::{ImagePixelFormat, ImageView, OwnedImage};
use model_runtime::{HuggingFaceModelSpec, ModelTask};
use video_analysis_core::{BoundingBox, DetectError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Variants describing SAM image preset.
pub enum SamImagePreset {
    #[default]
    /// The ViT base variant.
    VitBase,
    /// The ViT large variant.
    VitLarge,
    /// The ViT huge variant.
    VitHuge,
}

impl SamImagePreset {
    /// Constant for all.
    pub const ALL: &'static [Self] = &[Self::VitBase, Self::VitLarge, Self::VitHuge];

    /// Borrows this value as a str.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::VitBase => "sam-vit-base",
            Self::VitLarge => "sam-vit-large",
            Self::VitHuge => "sam-vit-huge",
        }
    }

    /// Returns repo identifier.
    pub fn repo_id(self) -> &'static str {
        match self {
            Self::VitBase => "facebook/sam-vit-base",
            Self::VitLarge => "facebook/sam-vit-large",
            Self::VitHuge => "facebook/sam-vit-huge",
        }
    }

    /// Returns model spec.
    pub fn model_spec(self) -> HuggingFaceModelSpec {
        HuggingFaceModelSpec::new(self.repo_id(), ModelTask::ImageSegmentation)
            .name(self.as_str())
            .file("config.json")
            .file("preprocessor_config.json")
            .first_available_file(["model.safetensors", "pytorch_model.bin"])
    }
}

/// Returns default SAM model spec.
pub fn default_sam_model_spec() -> HuggingFaceModelSpec {
    SamImagePreset::default().model_spec()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Variants describing point label.
pub enum PointLabel {
    /// The foreground variant.
    Foreground,
    /// The background variant.
    Background,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Data type for segmentation point.
pub struct SegmentationPoint {
    /// The x value.
    pub x: u32,
    /// The y value.
    pub y: u32,
    /// Label assigned to this value.
    pub label: PointLabel,
}

impl SegmentationPoint {
    /// Creates a new value.
    pub const fn foreground(x: u32, y: u32) -> Self {
        Self {
            x,
            y,
            label: PointLabel::Foreground,
        }
    }

    /// Creates a new value.
    pub const fn background(x: u32, y: u32) -> Self {
        Self {
            x,
            y,
            label: PointLabel::Background,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
/// Data type for image segmentation prompt.
pub struct ImageSegmentationPrompt {
    /// The points value.
    pub points: Vec<SegmentationPoint>,
    /// The boxes value.
    pub boxes: Vec<BoundingBox>,
    /// The automatic mask generation value.
    pub automatic_mask_generation: bool,
    /// The multimask output value.
    pub multimask_output: bool,
}

impl ImageSegmentationPrompt {
    /// Creates a new value.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns automatic mask generation.
    pub fn automatic_mask_generation() -> Self {
        Self {
            automatic_mask_generation: true,
            multimask_output: true,
            ..Self::default()
        }
    }

    /// Returns point.
    pub fn point(mut self, point: SegmentationPoint) -> Self {
        self.points.push(point);
        self
    }

    /// Returns bounding box.
    pub fn bounding_box(mut self, region: BoundingBox) -> Self {
        self.boxes.push(region);
        self
    }

    /// Returns multimask output.
    pub fn multimask_output(mut self, value: bool) -> Self {
        self.multimask_output = value;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Data type for image segmentation request.
pub struct ImageSegmentationRequest {
    /// The prompt value.
    pub prompt: ImageSegmentationPrompt,
    /// The min mask pixels value.
    pub min_mask_pixels: usize,
}

impl ImageSegmentationRequest {
    /// Creates a new value.
    pub fn new(prompt: ImageSegmentationPrompt) -> Self {
        Self {
            prompt,
            min_mask_pixels: 1,
        }
    }

    /// Returns automatic mask generation.
    pub fn automatic_mask_generation() -> Self {
        Self::new(ImageSegmentationPrompt::automatic_mask_generation())
    }

    /// Returns min mask pixels.
    pub fn min_mask_pixels(mut self, value: usize) -> Self {
        self.min_mask_pixels = value.max(1);
        self
    }
}

impl Default for ImageSegmentationRequest {
    fn default() -> Self {
        Self::new(ImageSegmentationPrompt::default())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Data type for binary mask.
pub struct BinaryMask {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Underlying data buffer.
    pub data: Vec<u8>,
}

impl BinaryMask {
    /// Creates a new value.
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(DetectError::InvalidDimensions { width, height });
        }
        let expected = width as usize * height as usize;
        if data.len() != expected {
            return Err(DetectError::InvalidFrameBuffer {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Returns empty.
    pub fn empty(width: u32, height: u32) -> Result<Self> {
        Self::new(width, height, vec![0; width as usize * height as usize])
    }

    /// Returns filled rect.
    pub fn filled_rect(width: u32, height: u32, region: BoundingBox) -> Result<Self> {
        if region.x.saturating_add(region.width) > width
            || region.y.saturating_add(region.height) > height
        {
            return Err(DetectError::InvalidArgument(
                "mask region must fit inside the mask dimensions".to_string(),
            ));
        }
        let mut mask = Self::empty(width, height)?;
        for y in region.y..region.y + region.height {
            for x in region.x..region.x + region.width {
                let index = mask.index(x, y);
                mask.data[index] = u8::MAX;
            }
        }
        Ok(mask)
    }

    /// Returns whether is active.
    pub fn is_active(&self, x: u32, y: u32) -> bool {
        self.data[self.index(x, y)] != 0
    }

    /// Returns active pixels.
    pub fn active_pixels(&self) -> usize {
        self.data.iter().filter(|value| **value != 0).count()
    }

    /// Returns bounding box.
    pub fn bounding_box(&self) -> Option<BoundingBox> {
        let mut min_x = self.width;
        let mut min_y = self.height;
        let mut max_x = 0_u32;
        let mut max_y = 0_u32;
        let mut found = false;

        for y in 0..self.height {
            for x in 0..self.width {
                if !self.is_active(x, y) {
                    continue;
                }
                found = true;
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }

        found.then(|| BoundingBox {
            x: min_x,
            y: min_y,
            width: max_x - min_x + 1,
            height: max_y - min_y + 1,
        })
    }

    /// Converts this value to image.
    pub fn to_image(&self) -> Result<OwnedImage> {
        OwnedImage::new(
            self.width,
            self.height,
            ImagePixelFormat::Gray8,
            self.data.clone(),
            self.width as usize,
        )
    }

    fn index(&self, x: u32, y: u32) -> usize {
        y as usize * self.width as usize + x as usize
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for image segment.
pub struct ImageSegment {
    /// Label assigned to this value.
    pub label: Option<String>,
    /// Score assigned to this value.
    pub score: Option<f32>,
    /// The region value.
    pub region: BoundingBox,
    /// The mask value.
    pub mask: BinaryMask,
    /// The attributes value.
    pub attributes: BTreeMap<String, String>,
}

impl ImageSegment {
    /// Creates a new value.
    pub fn new(mask: BinaryMask) -> Result<Self> {
        let region = mask.bounding_box().ok_or_else(|| {
            DetectError::InvalidArgument(
                "segmentation mask must contain at least one active pixel".to_string(),
            )
        })?;
        Ok(Self {
            label: None,
            score: None,
            region,
            mask,
            attributes: BTreeMap::new(),
        })
    }

    /// Returns label.
    pub fn label(mut self, value: impl Into<String>) -> Self {
        self.label = Some(value.into());
        self
    }

    /// Returns score.
    pub fn score(mut self, value: f32) -> Self {
        self.score = Some(value);
        self
    }

    /// Returns attribute.
    pub fn attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }
}

/// Trait for image segmentation backend implementations.
pub trait ImageSegmentationBackend {
    /// Returns segment image.
    fn segment_image(
        &mut self,
        image: &ImageView<'_>,
        request: &ImageSegmentationRequest,
    ) -> Result<Vec<ImageSegment>>;
}

/// Executes one segmentation request through a backend while enforcing the
/// library-owned request and result invariants shared by every backend.
pub fn segment_image_with_backend<B: ImageSegmentationBackend + ?Sized>(
    backend: &mut B,
    image: &ImageView<'_>,
    request: &ImageSegmentationRequest,
) -> Result<Vec<ImageSegment>> {
    validate_segmentation_request(image, request)?;

    let segments = backend.segment_image(image, request)?;
    let mut accepted = Vec::with_capacity(segments.len());
    for (index, segment) in segments.into_iter().enumerate() {
        validate_backend_segment(image, index, &segment)?;
        if segment.mask.active_pixels() >= request.min_mask_pixels {
            accepted.push(segment);
        }
    }

    if !request.prompt.multimask_output && accepted.len() > 1 {
        return Err(DetectError::InvalidArgument(format!(
            "segmentation backend returned {} masks while multimask_output is false",
            accepted.len()
        )));
    }

    Ok(accepted)
}

fn validate_segmentation_request(
    image: &ImageView<'_>,
    request: &ImageSegmentationRequest,
) -> Result<()> {
    if request.min_mask_pixels == 0 {
        return Err(DetectError::InvalidArgument(
            "min_mask_pixels must be greater than zero".to_string(),
        ));
    }

    let prompt = &request.prompt;
    let has_explicit_prompt = !prompt.points.is_empty() || !prompt.boxes.is_empty();
    if prompt.automatic_mask_generation && has_explicit_prompt {
        return Err(DetectError::InvalidArgument(
            "automatic mask generation cannot be combined with point or box prompts".to_string(),
        ));
    }
    if !prompt.automatic_mask_generation && !has_explicit_prompt {
        return Err(DetectError::InvalidArgument(
            "manual segmentation requires at least one point or box prompt".to_string(),
        ));
    }

    for point in &prompt.points {
        if point.x >= image.width || point.y >= image.height {
            return Err(DetectError::InvalidArgument(format!(
                "segmentation point ({}, {}) must fit inside image dimensions {}x{}",
                point.x, point.y, image.width, image.height
            )));
        }
    }

    for region in &prompt.boxes {
        if region.x.saturating_add(region.width) > image.width
            || region.y.saturating_add(region.height) > image.height
        {
            return Err(DetectError::InvalidArgument(format!(
                "segmentation box at ({}, {}) with size {}x{} must fit inside image dimensions {}x{}",
                region.x,
                region.y,
                region.width,
                region.height,
                image.width,
                image.height
            )));
        }
    }

    Ok(())
}

fn validate_backend_segment(
    image: &ImageView<'_>,
    index: usize,
    segment: &ImageSegment,
) -> Result<()> {
    if segment.mask.width != image.width || segment.mask.height != image.height {
        return Err(DetectError::InvalidArgument(format!(
            "segmentation backend mask {index} has dimensions {}x{}; expected {}x{}",
            segment.mask.width, segment.mask.height, image.width, image.height
        )));
    }

    let expected = segment.mask.width as usize * segment.mask.height as usize;
    if segment.mask.data.len() != expected {
        return Err(DetectError::InvalidFrameBuffer {
            expected,
            actual: segment.mask.data.len(),
        });
    }

    let bounds = segment.mask.bounding_box().ok_or_else(|| {
        DetectError::InvalidArgument(format!(
            "segmentation backend mask {index} contains no active pixels"
        ))
    })?;
    if bounds != segment.region {
        return Err(DetectError::InvalidArgument(format!(
            "segmentation backend mask {index} region does not match its active mask bounds"
        )));
    }
    if segment.score.is_some_and(|score| !score.is_finite()) {
        return Err(DetectError::InvalidArgument(format!(
            "segmentation backend mask {index} score must be finite"
        )));
    }

    Ok(())
}

/// Trait for model-backed image segmentation backend implementations.
pub trait ModelBackedImageSegmentationBackend: ImageSegmentationBackend {
    /// Returns model spec.
    fn model_spec(&self) -> HuggingFaceModelSpec;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct StubSegmentationBackend {
        segments: Vec<ImageSegment>,
    }

    impl ImageSegmentationBackend for StubSegmentationBackend {
        fn segment_image(
            &mut self,
            _image: &ImageView<'_>,
            _request: &ImageSegmentationRequest,
        ) -> Result<Vec<ImageSegment>> {
            Ok(self.segments.clone())
        }
    }

    fn test_image(width: u32, height: u32) -> OwnedImage {
        OwnedImage::new(
            width,
            height,
            ImagePixelFormat::Gray8,
            vec![0; width as usize * height as usize],
            width as usize,
        )
        .unwrap()
    }

    fn rect_segment(width: u32, height: u32, region: BoundingBox) -> ImageSegment {
        ImageSegment::new(BinaryMask::filled_rect(width, height, region).unwrap()).unwrap()
    }

    #[test]
    fn binary_mask_bounding_box_tracks_active_region() {
        let mask = BinaryMask::filled_rect(8, 6, BoundingBox::new(2, 1, 3, 4).unwrap()).unwrap();
        assert_eq!(mask.active_pixels(), 12);
        assert_eq!(
            mask.bounding_box(),
            Some(BoundingBox::new(2, 1, 3, 4).unwrap())
        );
    }

    #[test]
    fn image_segment_rejects_empty_masks() {
        let mask = BinaryMask::empty(4, 4).unwrap();
        assert!(ImageSegment::new(mask).is_err());
    }

    #[test]
    fn segmentation_request_default_is_manual() {
        let request = ImageSegmentationRequest::default();
        assert!(!request.prompt.automatic_mask_generation);
        assert!(request.prompt.points.is_empty());
        assert!(request.prompt.boxes.is_empty());
    }

    #[test]
    fn segmentation_request_can_opt_into_automatic_masks() {
        let request = ImageSegmentationRequest::automatic_mask_generation();
        assert!(request.prompt.automatic_mask_generation);
        assert!(request.prompt.multimask_output);
    }

    #[test]
    fn segmentation_execution_filters_masks_below_minimum_size() {
        let image = test_image(4, 4);
        let mut backend = StubSegmentationBackend {
            segments: vec![
                rect_segment(4, 4, BoundingBox::new(0, 0, 1, 1).unwrap()),
                rect_segment(4, 4, BoundingBox::new(1, 1, 2, 2).unwrap()),
            ],
        };
        let request = ImageSegmentationRequest::new(
            ImageSegmentationPrompt::new()
                .point(SegmentationPoint::foreground(1, 1))
                .multimask_output(true),
        )
        .min_mask_pixels(2);

        let segments = segment_image_with_backend(&mut backend, &image.as_view(), &request).unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].mask.active_pixels(), 4);
        assert_eq!(segments[0].region, BoundingBox::new(1, 1, 2, 2).unwrap());
    }

    #[test]
    fn segmentation_execution_rejects_manual_requests_without_prompts() {
        let image = test_image(4, 4);
        let mut backend = StubSegmentationBackend::default();
        let error = segment_image_with_backend(
            &mut backend,
            &image.as_view(),
            &ImageSegmentationRequest::default(),
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("manual segmentation requires at least one point or box prompt"));
    }

    #[test]
    fn segmentation_execution_rejects_out_of_bounds_points() {
        let image = test_image(4, 4);
        let mut backend = StubSegmentationBackend::default();
        let request = ImageSegmentationRequest::new(
            ImageSegmentationPrompt::new().point(SegmentationPoint::foreground(4, 0)),
        );
        let error =
            segment_image_with_backend(&mut backend, &image.as_view(), &request).unwrap_err();
        assert!(error.to_string().contains("must fit inside image dimensions 4x4"));
    }

    #[test]
    fn segmentation_execution_rejects_backend_masks_with_wrong_dimensions() {
        let image = test_image(4, 4);
        let mut backend = StubSegmentationBackend {
            segments: vec![rect_segment(
                3,
                3,
                BoundingBox::new(0, 0, 2, 2).unwrap(),
            )],
        };
        let request = ImageSegmentationRequest::new(
            ImageSegmentationPrompt::new().point(SegmentationPoint::foreground(1, 1)),
        );
        let error =
            segment_image_with_backend(&mut backend, &image.as_view(), &request).unwrap_err();
        assert!(error.to_string().contains("expected 4x4"));
    }

    #[test]
    fn segmentation_execution_rejects_malformed_backend_mask_buffers() {
        let image = test_image(4, 4);
        let request = ImageSegmentationRequest::new(
            ImageSegmentationPrompt::new().point(SegmentationPoint::foreground(1, 1)),
        );

        for data in [vec![u8::MAX], vec![u8::MAX; 17]] {
            let actual = data.len();
            let segment = ImageSegment {
                label: None,
                score: None,
                region: BoundingBox::new(0, 0, 1, 1).unwrap(),
                mask: BinaryMask {
                    width: 4,
                    height: 4,
                    data,
                },
                attributes: BTreeMap::new(),
            };
            let mut backend = StubSegmentationBackend {
                segments: vec![segment],
            };
            let error =
                segment_image_with_backend(&mut backend, &image.as_view(), &request).unwrap_err();
            assert!(matches!(
                error,
                DetectError::InvalidFrameBuffer {
                    expected: 16,
                    actual: observed,
                } if observed == actual
            ));
        }
    }

    #[test]
    fn segmentation_execution_rejects_multiple_masks_when_multimask_is_disabled() {
        let image = test_image(4, 4);
        let mut backend = StubSegmentationBackend {
            segments: vec![
                rect_segment(4, 4, BoundingBox::new(0, 0, 2, 2).unwrap()),
                rect_segment(4, 4, BoundingBox::new(2, 2, 2, 2).unwrap()),
            ],
        };
        let request = ImageSegmentationRequest::new(
            ImageSegmentationPrompt::new().point(SegmentationPoint::foreground(1, 1)),
        );
        let error =
            segment_image_with_backend(&mut backend, &image.as_view(), &request).unwrap_err();
        assert!(error
            .to_string()
            .contains("multimask_output is false"));
    }

    #[test]
    fn default_sam_model_spec_uses_image_segmentation_task() {
        let spec = default_sam_model_spec();
        assert_eq!(spec.repo_id_value(), Some("facebook/sam-vit-base"));
        assert_eq!(spec.task, ModelTask::ImageSegmentation);
    }
}
