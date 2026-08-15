//! Runtime-surface image payload contracts.

use std::fmt;

use serde::{Deserialize, Serialize};
use video_analysis_core::DetectError;

use crate::{ImagePixelFormat, ImageView, OwnedImage};

/// Serde-facing in-memory image payload used by image runtime surfaces.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ImagePayload {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Runtime-surface pixel format string.
    pub pixel_format: String,
    /// Optional row stride in bytes. Packed stride is used when omitted.
    pub stride: Option<usize>,
    /// Image buffer bytes.
    pub data: Vec<u8>,
}

/// Errors returned when turning an image payload into core image types.
#[derive(Debug)]
pub enum ImagePayloadError {
    /// The payload used an unsupported runtime-surface pixel format string.
    UnsupportedPixelFormat(String),
    /// The payload could not be represented as a valid image.
    InvalidImage(DetectError),
}

/// Result type for image payload contract operations.
pub type ImagePayloadResult<T> = Result<T, ImagePayloadError>;

impl ImagePayload {
    /// Creates a packed image payload.
    pub fn new(width: u32, height: u32, pixel_format: ImagePixelFormat, data: Vec<u8>) -> Self {
        Self {
            width,
            height,
            pixel_format: pixel_format_name(pixel_format).to_string(),
            stride: None,
            data,
        }
    }

    /// Creates an image payload with an explicit row stride.
    pub fn with_stride(
        width: u32,
        height: u32,
        pixel_format: ImagePixelFormat,
        stride: usize,
        data: Vec<u8>,
    ) -> Self {
        Self {
            width,
            height,
            pixel_format: pixel_format_name(pixel_format).to_string(),
            stride: Some(stride),
            data,
        }
    }

    /// Borrows this payload as an image view.
    pub fn view(&self) -> ImagePayloadResult<ImageView<'_>> {
        let pixel_format = parse_pixel_format(&self.pixel_format)?;
        let stride = self
            .stride
            .unwrap_or(self.width as usize * pixel_format.bytes_per_pixel());
        ImageView::new(self.width, self.height, pixel_format, &self.data, stride)
            .map_err(ImagePayloadError::InvalidImage)
    }

    /// Converts this payload into an owned image.
    pub fn owned(&self) -> ImagePayloadResult<OwnedImage> {
        let pixel_format = parse_pixel_format(&self.pixel_format)?;
        let stride = self
            .stride
            .unwrap_or(self.width as usize * pixel_format.bytes_per_pixel());
        OwnedImage::new(
            self.width,
            self.height,
            pixel_format,
            self.data.clone(),
            stride,
        )
        .map_err(ImagePayloadError::InvalidImage)
    }
}

/// Parses a runtime-surface pixel format string.
pub fn parse_pixel_format(value: &str) -> ImagePayloadResult<ImagePixelFormat> {
    match value {
        "rgb24" => Ok(ImagePixelFormat::Rgb24),
        "bgr24" => Ok(ImagePixelFormat::Bgr24),
        "gray8" => Ok(ImagePixelFormat::Gray8),
        other => Err(ImagePayloadError::UnsupportedPixelFormat(other.to_string())),
    }
}

/// Returns the runtime-surface string for a pixel format.
pub fn pixel_format_name(format: ImagePixelFormat) -> &'static str {
    match format {
        ImagePixelFormat::Rgb24 => "rgb24",
        ImagePixelFormat::Bgr24 => "bgr24",
        ImagePixelFormat::Gray8 => "gray8",
    }
}

/// Returns a small RGB sample image payload.
pub fn sample_image_payload() -> ImagePayload {
    ImagePayload::new(
        2,
        2,
        ImagePixelFormat::Rgb24,
        vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255],
    )
}

/// Returns a small RGB sample image as runtime-surface JSON.
pub fn sample_image_json() -> serde_json::Value {
    serde_json::to_value(sample_image_payload()).expect("sample image payload serializes")
}

impl fmt::Display for ImagePayloadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPixelFormat(value) => {
                write!(formatter, "unsupported image pixel format `{value}`")
            }
            Self::InvalidImage(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ImagePayloadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::UnsupportedPixelFormat(_) => None,
            Self::InvalidImage(error) => Some(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn image_payload_view_uses_default_stride() {
        let payload = ImagePayload::new(2, 2, ImagePixelFormat::Rgb24, vec![0; 12]);

        let image = payload.view().expect("valid image");

        assert_eq!(image.stride, 2 * ImagePixelFormat::Rgb24.bytes_per_pixel());
    }

    #[test]
    fn image_payload_view_respects_explicit_stride() {
        let payload = ImagePayload::with_stride(2, 2, ImagePixelFormat::Rgb24, 8, vec![0; 16]);

        let image = payload.view().expect("valid image");

        assert_eq!(image.stride, 8);
    }

    #[test]
    fn image_payload_rejects_unsupported_pixel_format() {
        let payload = ImagePayload {
            width: 2,
            height: 2,
            pixel_format: "rgba8".to_string(),
            stride: None,
            data: vec![0; 16],
        };

        let error = payload.view().expect_err("unsupported format");

        match error {
            ImagePayloadError::UnsupportedPixelFormat(value) => assert_eq!(value, "rgba8"),
            ImagePayloadError::InvalidImage(_) => panic!("expected unsupported pixel format"),
        }
    }

    #[test]
    fn image_payload_rejects_short_buffer() {
        let payload = ImagePayload::new(2, 2, ImagePixelFormat::Rgb24, vec![0; 3]);

        let error = payload.view().expect_err("short buffer");

        match error {
            ImagePayloadError::InvalidImage(_) => {}
            ImagePayloadError::UnsupportedPixelFormat(value) => {
                panic!("expected invalid image, got unsupported format {value}")
            }
        }
    }

    #[test]
    fn sample_image_json_keeps_runtime_surface_shape() {
        let sample = sample_image_json();
        let object = sample.as_object().expect("sample object");
        let keys = object.keys().map(String::as_str).collect::<BTreeSet<_>>();

        assert_eq!(
            keys,
            BTreeSet::from(["data", "height", "pixelFormat", "stride", "width"])
        );
        assert_eq!(sample["pixelFormat"], "rgb24");
    }
}
