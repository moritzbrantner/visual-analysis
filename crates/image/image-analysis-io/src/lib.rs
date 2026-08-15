#![doc = include_str!("../README.md")]

pub mod surface;
use std::fs;
use std::path::Path;

use image::{DynamicImage, GrayImage, ImageBuffer, ImageFormat, RgbImage};
use image_analysis_core::{ImagePixelFormat, OwnedImage};
use video_analysis_core::{DetectError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Variants describing image file format.
pub enum ImageFileFormat {
    /// The png variant.
    Png,
    /// The jpeg variant.
    Jpeg,
    /// The web p variant.
    WebP,
}

impl ImageFileFormat {
    /// Builds this value from path.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let extension = path
            .extension()
            .and_then(|value| value.to_str())
            .ok_or_else(|| {
                DetectError::InvalidArgument(format!(
                    "could not infer image format from `{}`",
                    path.display()
                ))
            })?;
        match extension.to_ascii_lowercase().as_str() {
            "png" => Ok(Self::Png),
            "jpg" | "jpeg" => Ok(Self::Jpeg),
            "webp" => Ok(Self::WebP),
            other => Err(DetectError::InvalidArgument(format!(
                "unsupported image file extension `{other}`"
            ))),
        }
    }

    fn as_image_format(self) -> ImageFormat {
        match self {
            Self::Png => ImageFormat::Png,
            Self::Jpeg => ImageFormat::Jpeg,
            Self::WebP => ImageFormat::WebP,
        }
    }
}

/// Reads image.
pub fn read_image(path: impl AsRef<Path>) -> Result<OwnedImage> {
    let path = path.as_ref();
    read_image_with_format(path, ImageFileFormat::from_path(path)?)
}

/// Reads image with format.
pub fn read_image_with_format(
    path: impl AsRef<Path>,
    format: ImageFileFormat,
) -> Result<OwnedImage> {
    let path = path.as_ref();
    let bytes = fs::read(path)?;
    let image =
        image::load_from_memory_with_format(&bytes, format.as_image_format()).map_err(|err| {
            DetectError::Source(format!("failed to decode `{}`: {err}", path.display()))
        })?;
    dynamic_to_owned(image)
}

/// Writes image.
pub fn write_image(path: impl AsRef<Path>, image: &OwnedImage) -> Result<()> {
    let path = path.as_ref();
    write_image_with_format(path, image, ImageFileFormat::from_path(path)?)
}

/// Writes image with format.
pub fn write_image_with_format(
    path: impl AsRef<Path>,
    image: &OwnedImage,
    format: ImageFileFormat,
) -> Result<()> {
    let path = path.as_ref();
    let dynamic = owned_to_dynamic(image)?;
    let mut file = fs::File::create(path)?;
    dynamic
        .write_to(&mut file, format.as_image_format())
        .map_err(|err| DetectError::Source(format!("failed to encode `{}`: {err}", path.display())))
}

fn dynamic_to_owned(image: DynamicImage) -> Result<OwnedImage> {
    match image {
        DynamicImage::ImageLuma8(image) => {
            OwnedImage::new_gray(image.width(), image.height(), image.into_raw())
        }
        DynamicImage::ImageLuma16(image) => {
            let image = DynamicImage::ImageLuma16(image).to_luma8();
            OwnedImage::new_gray(image.width(), image.height(), image.into_raw())
        }
        DynamicImage::ImageRgb8(image) => {
            OwnedImage::new_rgb(image.width(), image.height(), image.into_raw())
        }
        other => {
            let image = other.to_rgb8();
            OwnedImage::new_rgb(image.width(), image.height(), image.into_raw())
        }
    }
}

fn owned_to_dynamic(image: &OwnedImage) -> Result<DynamicImage> {
    match image.pixel_format {
        ImagePixelFormat::Rgb24 => {
            let buffer = RgbImage::from_raw(image.width, image.height, image.data.clone())
                .ok_or_else(|| invalid_image_buffer(image))?;
            Ok(DynamicImage::ImageRgb8(buffer))
        }
        ImagePixelFormat::Bgr24 => {
            let mut rgb = Vec::with_capacity(image.data.len());
            for chunk in image.data.chunks_exact(3) {
                rgb.extend_from_slice(&[chunk[2], chunk[1], chunk[0]]);
            }
            let buffer = RgbImage::from_raw(image.width, image.height, rgb)
                .ok_or_else(|| invalid_image_buffer(image))?;
            Ok(DynamicImage::ImageRgb8(buffer))
        }
        ImagePixelFormat::Gray8 => {
            let buffer: GrayImage =
                ImageBuffer::from_raw(image.width, image.height, image.data.clone())
                    .ok_or_else(|| invalid_image_buffer(image))?;
            Ok(DynamicImage::ImageLuma8(buffer))
        }
    }
}

fn invalid_image_buffer(image: &OwnedImage) -> DetectError {
    DetectError::InvalidFrameBuffer {
        expected: image.stride * image.height as usize,
        actual: image.data.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use tempfile::tempdir;

    #[test]
    fn png_roundtrip_is_lossless() {
        let temp = tempdir().unwrap();
        let path = temp.path().join("roundtrip.png");
        let image = OwnedImage::new_rgb(2, 1, vec![255, 0, 0, 0, 255, 0]).unwrap();
        write_image(&path, &image).unwrap();
        let loaded = read_image(&path).unwrap();
        assert_eq!(loaded, image);
    }

    #[test]
    fn jpeg_and_webp_roundtrip_dimensions() {
        let temp = tempdir().unwrap();
        let image = OwnedImage::new_rgb(3, 2, vec![180; 18]).unwrap();

        let jpeg = temp.path().join("preview.jpg");
        write_image(&jpeg, &image).unwrap();
        let decoded_jpeg = read_image(&jpeg).unwrap();
        assert_eq!((decoded_jpeg.width, decoded_jpeg.height), (3, 2));

        let webp = temp.path().join("preview.webp");
        write_image(&webp, &image).unwrap();
        let decoded_webp = read_image(&webp).unwrap();
        assert_eq!((decoded_webp.width, decoded_webp.height), (3, 2));
    }
}
