//! Connected color-blob detection for still images and video frames.

use std::collections::BTreeMap;

use image_analysis_core::{ImagePixelFormat, ImageView};
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
    let width = image.width as usize;
    let bytes_per_pixel = image.pixel_format.bytes_per_pixel();
    let [red_offset, green_offset, blue_offset] = match image.pixel_format {
        ImagePixelFormat::Rgb24 => [0, 1, 2],
        ImagePixelFormat::Bgr24 => [2, 1, 0],
        ImagePixelFormat::Gray8 => [0, 0, 0],
    };
    let mut mask = vec![false; width * image.height as usize];
    for (row, mask_row) in image
        .data
        .chunks(image.stride)
        .zip(mask.chunks_exact_mut(width))
    {
        for (pixel, active) in row[..width * bytes_per_pixel]
            .chunks_exact(bytes_per_pixel)
            .zip(mask_row)
        {
            let (red, green, blue) = (pixel[red_offset], pixel[green_offset], pixel[blue_offset]);
            *active = match options.target {
                TargetColor::Red => {
                    red >= options.min_red
                        && (red as f32) >= options.dominance_ratio * green as f32
                        && (red as f32) >= options.dominance_ratio * blue as f32
                }
            };
        }
    }
    mask
}

fn erode_3x3(width: usize, height: usize, mask: &[bool]) -> Vec<bool> {
    let mut out = vec![false; mask.len()];
    if width < 3 || height < 3 {
        return out;
    }
    let mut columns = vec![false; width];
    for y in 1..height - 1 {
        columns.fill(true);
        for row in mask[(y - 1) * width..(y + 2) * width].chunks_exact(width) {
            for (column, &value) in columns.iter_mut().zip(row) {
                *column &= value;
            }
        }
        let out_row = &mut out[y * width..(y + 1) * width];
        for (active, window) in out_row[1..width - 1].iter_mut().zip(columns.windows(3)) {
            *active = window[0] & window[1] & window[2];
        }
    }
    out
}

fn dilate_3x3(width: usize, height: usize, mask: &[bool]) -> Vec<bool> {
    let mut out = vec![false; mask.len()];
    if width == 0 || height == 0 {
        return out;
    }
    let mut columns = vec![false; width];
    for y in 0..height {
        columns.fill(false);
        let rows = y.saturating_sub(1)..(y + 2).min(height);
        for row in mask[rows.start * width..rows.end * width].chunks_exact(width) {
            for (column, &value) in columns.iter_mut().zip(row) {
                *column |= value;
            }
        }
        let out_row = &mut out[y * width..(y + 1) * width];
        for (x, active) in out_row.iter_mut().enumerate() {
            let left = x.saturating_sub(1);
            let right = (x + 1).min(width - 1);
            *active = columns[left] | columns[x] | columns[right];
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

#[cfg(test)]
mod tests {
    use super::*;

    fn neighborhood(width: usize, height: usize, x: usize, y: usize) -> Vec<usize> {
        let mut indices = Vec::new();
        for yy in y.saturating_sub(1)..=(y + 1).min(height - 1) {
            for xx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                indices.push(yy * width + xx);
            }
        }
        indices
    }

    #[test]
    fn morphology_matches_per_pixel_definition_for_every_small_mask() {
        for (width, height) in [(1, 1), (1, 4), (4, 1), (2, 3), (3, 3), (4, 3), (3, 4), (4, 4)] {
            for bits in 0..1_u32 << (width * height) {
                let mask = (0..width * height)
                    .map(|index| bits >> index & 1 == 1)
                    .collect::<Vec<_>>();
                let eroded = erode_3x3(width, height, &mask);
                let dilated = dilate_3x3(width, height, &mask);
                for y in 0..height {
                    for x in 0..width {
                        let around = neighborhood(width, height, x, y);
                        let interior = x > 0 && y > 0 && x + 1 < width && y + 1 < height;
                        assert_eq!(
                            eroded[y * width + x],
                            interior && around.iter().all(|&index| mask[index])
                        );
                        assert_eq!(
                            dilated[y * width + x],
                            around.iter().any(|&index| mask[index])
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn color_mask_matches_pixel_rgb_across_formats_and_stride_padding() {
        let options = ColorBlobDetectionOptions::default();
        for format in [
            ImagePixelFormat::Rgb24,
            ImagePixelFormat::Bgr24,
            ImagePixelFormat::Gray8,
        ] {
            let (width, height, stride) = (5_u32, 3_u32, 5 * format.bytes_per_pixel() + 2);
            let data = (0..stride * height as usize)
                .map(|index| (index * 37 % 251) as u8)
                .collect::<Vec<_>>();
            let image = ImageView::new(width, height, format, &data, stride).unwrap();
            let mask = build_color_mask(image, &options);
            let mut active = 0;
            for y in 0..height {
                for x in 0..width {
                    let [red, green, blue] = image.pixel_rgb(x, y);
                    let expected = red >= options.min_red
                        && red as f32 >= options.dominance_ratio * green as f32
                        && red as f32 >= options.dominance_ratio * blue as f32;
                    active += usize::from(expected);
                    assert_eq!(mask[(y * width + x) as usize], expected);
                }
            }
            assert_eq!(mask.iter().filter(|value| **value).count(), active);
        }
    }
}
