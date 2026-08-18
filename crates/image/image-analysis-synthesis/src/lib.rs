#![doc = include_str!("../README.md")]

pub mod surface;
use data_inversion_core::{Generated, InformationFidelity, InversionMethod, InversionTrace};
use image_analysis_core::{ImagePixelFormat, OwnedImage};
use video_analysis_core::{BoundingBox, DetectError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Data type for RGB color.
pub struct RgbColor {
    /// The red value.
    pub red: u8,
    /// The green value.
    pub green: u8,
    /// The blue value.
    pub blue: u8,
}

impl RgbColor {
    /// Constant for black.
    pub const BLACK: Self = Self::new(0, 0, 0);
    /// Constant for white.
    pub const WHITE: Self = Self::new(255, 255, 255);

    /// Creates a new value.
    pub const fn new(red: u8, green: u8, blue: u8) -> Self {
        Self { red, green, blue }
    }

    /// Creates a new value.
    pub const fn gray(value: u8) -> Self {
        Self::new(value, value, value)
    }

    /// Returns luma.
    pub fn luma(self) -> u8 {
        (0.299 * self.red as f32 + 0.587 * self.green as f32 + 0.114 * self.blue as f32).round()
            as u8
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Data type for image synthesis config.
pub struct ImageSynthesisConfig {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// The pixel format value.
    pub pixel_format: ImagePixelFormat,
}

impl ImageSynthesisConfig {
    /// Creates a new value.
    pub fn new(width: u32, height: u32, pixel_format: ImagePixelFormat) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(DetectError::InvalidDimensions { width, height });
        }
        Ok(Self {
            width,
            height,
            pixel_format,
        })
    }

    /// Returns stride.
    pub fn stride(self) -> usize {
        self.width as usize * self.pixel_format.bytes_per_pixel()
    }
}

impl Default for ImageSynthesisConfig {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            pixel_format: ImagePixelFormat::Rgb24,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Data type for region paint.
pub struct RegionPaint {
    /// The region value.
    pub region: BoundingBox,
    /// The color value.
    pub color: RgbColor,
}

/// Returns solid image.
pub fn solid_image(color: RgbColor, config: ImageSynthesisConfig) -> Result<Generated<OwnedImage>> {
    ImageSynthesisConfig::new(config.width, config.height, config.pixel_format)?;
    let mut data = vec![0_u8; config.stride() * config.height as usize];
    for y in 0..config.height {
        for x in 0..config.width {
            write_pixel(&mut data, config, x, y, color);
        }
    }
    let image = OwnedImage::new(
        config.width,
        config.height,
        config.pixel_format,
        data,
        config.stride(),
    )?;
    let trace = InversionTrace::new("rgb_color", "owned_image", InformationFidelity::Placeholder)
        .note(
            "pixels",
            InversionMethod::Defaulted,
            "every pixel is filled from the same color",
        );
    Ok(Generated::new(image, trace))
}

/// Returns vertical gradient.
pub fn vertical_gradient(
    top: RgbColor,
    bottom: RgbColor,
    config: ImageSynthesisConfig,
) -> Result<Generated<OwnedImage>> {
    ImageSynthesisConfig::new(config.width, config.height, config.pixel_format)?;
    let mut data = vec![0_u8; config.stride() * config.height as usize];
    let denom = config.height.saturating_sub(1).max(1) as f32;
    for y in 0..config.height {
        let t = y as f32 / denom;
        let color = lerp_color(top, bottom, t);
        for x in 0..config.width {
            write_pixel(&mut data, config, x, y, color);
        }
    }
    let image = OwnedImage::new(
        config.width,
        config.height,
        config.pixel_format,
        data,
        config.stride(),
    )?;
    let trace = InversionTrace::new(
        "color_stops",
        "owned_image",
        InformationFidelity::Interpolated,
    )
    .note(
        "pixels",
        InversionMethod::Interpolated,
        "rows are linearly interpolated between two colors",
    );
    Ok(Generated::new(image, trace))
}

/// Returns image from luma histogram.
pub fn image_from_luma_histogram(
    histogram: &[u64],
    config: ImageSynthesisConfig,
) -> Result<Generated<OwnedImage>> {
    ImageSynthesisConfig::new(config.width, config.height, config.pixel_format)?;
    if histogram.is_empty() || histogram.len() > 256 {
        return Err(invalid_argument(
            "histogram bins must be in the range 1..=256",
        ));
    }
    let total = histogram.iter().sum::<u64>();
    if total == 0 {
        return Err(invalid_argument(
            "histogram must contain at least one count",
        ));
    }

    let pixels = config.width as usize * config.height as usize;
    let mut data = vec![0_u8; config.stride() * config.height as usize];
    let mut bin = 0_usize;
    let mut cumulative = histogram[0];
    for index in 0..pixels {
        let target = ((index as u128 * total as u128) / pixels as u128) as u64;
        while target >= cumulative && bin + 1 < histogram.len() {
            bin += 1;
            cumulative += histogram[bin];
        }
        let value = (((bin as f32 + 0.5) * 256.0) / histogram.len() as f32)
            .round()
            .clamp(0.0, 255.0) as u8;
        let x = (index % config.width as usize) as u32;
        let y = (index / config.width as usize) as u32;
        write_pixel(&mut data, config, x, y, RgbColor::gray(value));
    }

    let image = OwnedImage::new(
        config.width,
        config.height,
        config.pixel_format,
        data,
        config.stride(),
    )?;
    let trace = InversionTrace::new(
        "luma_histogram",
        "owned_image",
        InformationFidelity::Heuristic,
    )
    .note(
        "spatial_layout",
        InversionMethod::Inferred,
        "histogram counts are expanded into a deterministic scanline layout",
    );
    Ok(Generated::new(image, trace))
}

/// Returns paint regions.
pub fn paint_regions(
    background: RgbColor,
    regions: &[RegionPaint],
    config: ImageSynthesisConfig,
) -> Result<Generated<OwnedImage>> {
    let mut generated = solid_image(background, config)?;
    for paint in regions {
        paint_region(&mut generated.value, paint);
    }
    generated.trace.source_type = "regions".to_string();
    generated.trace.fidelity = InformationFidelity::Heuristic;
    generated.trace = generated.trace.note(
        "regions",
        InversionMethod::Preserved,
        "provided regions are clipped to the generated image bounds",
    );
    Ok(generated)
}

fn paint_region(image: &mut OwnedImage, paint: &RegionPaint) {
    let config = ImageSynthesisConfig {
        width: image.width,
        height: image.height,
        pixel_format: image.pixel_format,
    };
    let x_end = paint
        .region
        .x
        .saturating_add(paint.region.width)
        .min(image.width);
    let y_end = paint
        .region
        .y
        .saturating_add(paint.region.height)
        .min(image.height);
    for y in paint.region.y.min(image.height)..y_end {
        for x in paint.region.x.min(image.width)..x_end {
            write_pixel(&mut image.data, config, x, y, paint.color);
        }
    }
}

fn write_pixel(data: &mut [u8], config: ImageSynthesisConfig, x: u32, y: u32, color: RgbColor) {
    let bpp = config.pixel_format.bytes_per_pixel();
    let offset = y as usize * config.stride() + x as usize * bpp;
    match config.pixel_format {
        ImagePixelFormat::Rgb24 => {
            data[offset] = color.red;
            data[offset + 1] = color.green;
            data[offset + 2] = color.blue;
        }
        ImagePixelFormat::Bgr24 => {
            data[offset] = color.blue;
            data[offset + 1] = color.green;
            data[offset + 2] = color.red;
        }
        ImagePixelFormat::Gray8 => {
            data[offset] = color.luma();
        }
    }
}

fn lerp_color(left: RgbColor, right: RgbColor, t: f32) -> RgbColor {
    let channel = |a: u8, b: u8| a as f32 + (b as f32 - a as f32) * t;
    RgbColor::new(
        channel(left.red, right.red).round().clamp(0.0, 255.0) as u8,
        channel(left.green, right.green).round().clamp(0.0, 255.0) as u8,
        channel(left.blue, right.blue).round().clamp(0.0, 255.0) as u8,
    )
}

fn invalid_argument(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_image_from_histogram() {
        let generated = image_from_luma_histogram(
            &[1, 3],
            ImageSynthesisConfig::new(4, 1, ImagePixelFormat::Gray8).unwrap(),
        )
        .unwrap();
        assert_eq!(generated.value.data.len(), 4);
        assert!(generated.value.data[0] < generated.value.data[3]);
        assert_eq!(generated.trace.fidelity, InformationFidelity::Heuristic);
    }

    #[test]
    fn paints_region_over_background() {
        let region = RegionPaint {
            region: BoundingBox::new(1, 0, 1, 1).unwrap(),
            color: RgbColor::WHITE,
        };
        let generated = paint_regions(
            RgbColor::BLACK,
            &[region],
            ImageSynthesisConfig::new(2, 1, ImagePixelFormat::Rgb24).unwrap(),
        )
        .unwrap();
        assert_eq!(generated.value.data, vec![0, 0, 0, 255, 255, 255]);
    }
}
