//! Connected color-blob detection for still images and video frames.

use std::collections::BTreeMap;

use image_analysis_core::ImageView;
use video_analysis_core::{BoundingBox, DetectError, Result, VideoFrame};

use crate::{FrameDetections, ImageDetection};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Color target for blob detection.
pub enum TargetColor {
    /// Red-dominant regions.
    Red,
}

#[derive(Debug, Clone, PartialEq)]
/// Options for simple color-blob object detection.
pub struct ColorBlobDetectionOptions {
    /// Target color to detect.
    pub target: TargetColor,
    /// Minimum red channel value for red targets.
    pub min_red: u8,
    /// Minimum dominance ratio versus the other RGB channels.
    pub dominance_ratio: f32,
    /// Minimum connected-component area after morphology.
    pub min_area_pixels: usize,
    /// Output object label.
    pub label: String,
    /// Whether to apply a 3x3 morphological opening before component extraction.
    pub morph_open_3x3: bool,
}

impl ColorBlobDetectionOptions {
    /// Returns red-car defaults matching the external OpenCV adapter.
    pub fn red_car() -> Self {
        Self::default()
    }

    /// Returns min area pixels.
    pub fn min_area_pixels(mut self, value: usize) -> Self {
        self.min_area_pixels = value.max(1);
        self
    }

    /// Returns label.
    pub fn label(mut self, value: impl Into<String>) -> Self {
        self.label = value.into();
        self
    }
}

impl Default for ColorBlobDetectionOptions {
    fn default() -> Self {
        Self {
            target: TargetColor::Red,
            min_red: 96,
            dominance_ratio: 1.35,
            min_area_pixels: 24,
            label: "car".to_string(),
            morph_open_3x3: true,
        }
    }
}

#[derive(Debug, Clone)]
/// Detector for connected color blobs in still images or video frames.
pub struct ColorBlobDetector {
    pub(crate) options: ColorBlobDetectionOptions,
}

impl ColorBlobDetector {
    /// Creates a new detector.
    pub fn new(options: ColorBlobDetectionOptions) -> Result<Self> {
        validate_color_blob_options(&options)?;
        Ok(Self { options })
    }

    /// Creates a red-car detector.
    pub fn red_car() -> Self {
        Self {
            options: ColorBlobDetectionOptions::red_car(),
        }
    }

    /// Returns options.
    pub fn options(&self) -> &ColorBlobDetectionOptions {
        &self.options
    }

    /// Detects color blobs in an image.
    pub fn detect_image(&mut self, image: &ImageView<'_>) -> Result<Vec<ImageDetection>> {
        image.validate()?;
        let width = image.width as usize;
        let height = image.height as usize;
        let mut mask = build_color_mask(*image, &self.options);
        if self.options.morph_open_3x3 {
            mask = dilate_3x3(width, height, &erode_3x3(width, height, &mask));
        }
        Ok(connected_components(
            width,
            height,
            &mask,
            &self.options.label,
            self.options.min_area_pixels,
            self.options.target,
        ))
    }

    /// Detects color blobs in a video frame.
    pub fn detect_frame(&mut self, frame: &VideoFrame<'_>) -> Result<FrameDetections> {
        let image = ImageView::from_video_frame(frame)?;
        let detections = self.detect_image(&image)?;
        Ok(FrameDetections {
            position: frame.position,
            detections,
        })
    }
}

fn validate_color_blob_options(options: &ColorBlobDetectionOptions) -> Result<()> {
    if !options.dominance_ratio.is_finite() || options.dominance_ratio <= 0.0 {
        return Err(DetectError::InvalidArgument(
            "dominance_ratio must be finite and positive".to_string(),
        ));
    }
    if options.min_area_pixels == 0 {
        return Err(DetectError::InvalidArgument(
            "min_area_pixels must be positive".to_string(),
        ));
    }
    if options.label.trim().is_empty() {
        return Err(DetectError::InvalidArgument(
            "color blob label must not be empty".to_string(),
        ));
    }
    Ok(())
}

fn build_color_mask(image: ImageView<'_>, options: &ColorBlobDetectionOptions) -> Vec<bool> {
    let mut mask = vec![false; image.width as usize * image.height as usize];
    for y in 0..image.height {
        for x in 0..image.width {
            let [red, green, blue] = image.pixel_rgb(x, y);
            let active = match options.target {
                TargetColor::Red => {
                    red >= options.min_red
                        && (red as f32) >= options.dominance_ratio * green as f32
                        && (red as f32) >= options.dominance_ratio * blue as f32
                }
            };
            mask[y as usize * image.width as usize + x as usize] = active;
        }
    }
    mask
}

fn erode_3x3(width: usize, height: usize, mask: &[bool]) -> Vec<bool> {
    let mut out = vec![false; mask.len()];
    if width < 3 || height < 3 {
        return out;
    }
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut active = true;
            'neighbors: for yy in y - 1..=y + 1 {
                for xx in x - 1..=x + 1 {
                    if !mask[yy * width + xx] {
                        active = false;
                        break 'neighbors;
                    }
                }
            }
            out[y * width + x] = active;
        }
    }
    out
}

fn dilate_3x3(width: usize, height: usize, mask: &[bool]) -> Vec<bool> {
    let mut out = vec![false; mask.len()];
    if width == 0 || height == 0 {
        return out;
    }
    for y in 0..height {
        for x in 0..width {
            let y0 = y.saturating_sub(1);
            let x0 = x.saturating_sub(1);
            let y1 = (y + 1).min(height - 1);
            let x1 = (x + 1).min(width - 1);
            let mut active = false;
            'neighbors: for yy in y0..=y1 {
                for xx in x0..=x1 {
                    if mask[yy * width + xx] {
                        active = true;
                        break 'neighbors;
                    }
                }
            }
            out[y * width + x] = active;
        }
    }
    out
}

fn connected_components(
    width: usize,
    height: usize,
    mask: &[bool],
    label: &str,
    min_area_pixels: usize,
    target: TargetColor,
) -> Vec<ImageDetection> {
    let mut seen = vec![false; mask.len()];
    let mut detections = Vec::new();
    let mut stack = Vec::new();
    for start in 0..mask.len() {
        if seen[start] || !mask[start] {
            continue;
        }
        let mut min_x = width;
        let mut min_y = height;
        let mut max_x = 0_usize;
        let mut max_y = 0_usize;
        let mut area = 0_usize;
        seen[start] = true;
        stack.push(start);
        while let Some(index) = stack.pop() {
            let x = index % width;
            let y = index / width;
            area += 1;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);

            let x0 = x.saturating_sub(1);
            let y0 = y.saturating_sub(1);
            let x1 = (x + 1).min(width - 1);
            let y1 = (y + 1).min(height - 1);
            for yy in y0..=y1 {
                for xx in x0..=x1 {
                    let next = yy * width + xx;
                    if !seen[next] && mask[next] {
                        seen[next] = true;
                        stack.push(next);
                    }
                }
            }
        }
        if area < min_area_pixels {
            continue;
        }
        let mut attributes = BTreeMap::new();
        attributes.insert(
            "color".to_string(),
            match target {
                TargetColor::Red => "red",
            }
            .to_string(),
        );
        attributes.insert("area".to_string(), area.to_string());
        if let Ok(region) = BoundingBox::new(
            min_x as u32,
            min_y as u32,
            (max_x - min_x + 1) as u32,
            (max_y - min_y + 1) as u32,
        ) {
            detections.push(ImageDetection {
                label: label.to_string(),
                score: Some(0.95),
                region,
                attributes,
            });
        }
    }
    detections
}
