#![doc = include_str!("../README.md")]

pub mod contracts;
pub mod surface;
use tensor_data::F32Tensor;
use video_analysis_core::{DetectError, PixelFormat, Result, VideoFrame};

pub use contracts::{ImagePayload, ImagePayloadError, ImagePayloadResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Variants describing image pixel format.
pub enum ImagePixelFormat {
    /// The rgb24 variant.
    Rgb24,
    /// The bgr24 variant.
    Bgr24,
    /// The gray8 variant.
    Gray8,
}

impl ImagePixelFormat {
    /// Creates a new value.
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgb24 | Self::Bgr24 => 3,
            Self::Gray8 => 1,
        }
    }
}

impl From<PixelFormat> for ImagePixelFormat {
    fn from(value: PixelFormat) -> Self {
        match value {
            PixelFormat::Rgb24 => Self::Rgb24,
            PixelFormat::Bgr24 => Self::Bgr24,
        }
    }
}

#[derive(Debug, Clone, Copy)]
/// Data type for image view.
pub struct ImageView<'a> {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// The pixel format value.
    pub pixel_format: ImagePixelFormat,
    /// Underlying data buffer.
    pub data: &'a [u8],
    /// The stride value.
    pub stride: usize,
}

impl<'a> ImageView<'a> {
    /// Returns packed.
    pub fn packed(
        width: u32,
        height: u32,
        pixel_format: ImagePixelFormat,
        data: &'a [u8],
    ) -> Result<Self> {
        Self::new(
            width,
            height,
            pixel_format,
            data,
            width as usize * pixel_format.bytes_per_pixel(),
        )
    }

    /// Creates a new value.
    pub fn new(
        width: u32,
        height: u32,
        pixel_format: ImagePixelFormat,
        data: &'a [u8],
        stride: usize,
    ) -> Result<Self> {
        let image = Self {
            width,
            height,
            pixel_format,
            data,
            stride,
        };
        image.validate()?;
        Ok(image)
    }

    /// Builds this value from video frame.
    pub fn from_video_frame(frame: &VideoFrame<'a>) -> Result<Self> {
        Self::new(
            frame.width,
            frame.height,
            frame.pixel_format.into(),
            frame.data,
            frame.stride,
        )
    }

    /// Validates this value.
    pub fn validate(self) -> Result<()> {
        if self.width == 0 || self.height == 0 {
            return Err(DetectError::InvalidDimensions {
                width: self.width,
                height: self.height,
            });
        }
        let packed_width = self.width as usize * self.pixel_format.bytes_per_pixel();
        if self.stride < packed_width {
            return Err(DetectError::InvalidFrameBuffer {
                expected: packed_width,
                actual: self.stride,
            });
        }
        let expected = self.stride * self.height as usize;
        if self.data.len() < expected {
            return Err(DetectError::InvalidFrameBuffer {
                expected,
                actual: self.data.len(),
            });
        }
        Ok(())
    }

    /// Returns compact len.
    pub fn compact_len(self) -> usize {
        self.width as usize * self.height as usize * self.pixel_format.bytes_per_pixel()
    }

    /// Returns pixel RGB.
    pub fn pixel_rgb(self, x: u32, y: u32) -> [u8; 3] {
        let bpp = self.pixel_format.bytes_per_pixel();
        let offset = y as usize * self.stride + x as usize * bpp;
        match self.pixel_format {
            ImagePixelFormat::Rgb24 => [
                self.data[offset],
                self.data[offset + 1],
                self.data[offset + 2],
            ],
            ImagePixelFormat::Bgr24 => [
                self.data[offset + 2],
                self.data[offset + 1],
                self.data[offset],
            ],
            ImagePixelFormat::Gray8 => {
                let value = self.data[offset];
                [value, value, value]
            }
        }
    }

    /// Returns luma.
    pub fn luma(self, x: u32, y: u32) -> u8 {
        let [red, green, blue] = self.pixel_rgb(x, y);
        (0.299 * red as f32 + 0.587 * green as f32 + 0.114 * blue as f32).round() as u8
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for owned image.
pub struct OwnedImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// The pixel format value.
    pub pixel_format: ImagePixelFormat,
    /// Underlying data buffer.
    pub data: Vec<u8>,
    /// The stride value.
    pub stride: usize,
}

impl OwnedImage {
    /// Returns new RGB.
    pub fn new_rgb(width: u32, height: u32, data: Vec<u8>) -> Result<Self> {
        Self::new(
            width,
            height,
            ImagePixelFormat::Rgb24,
            data,
            width as usize * ImagePixelFormat::Rgb24.bytes_per_pixel(),
        )
    }

    /// Returns new BGR.
    pub fn new_bgr(width: u32, height: u32, data: Vec<u8>) -> Result<Self> {
        Self::new(
            width,
            height,
            ImagePixelFormat::Bgr24,
            data,
            width as usize * ImagePixelFormat::Bgr24.bytes_per_pixel(),
        )
    }

    /// Returns new gray.
    pub fn new_gray(width: u32, height: u32, data: Vec<u8>) -> Result<Self> {
        Self::new(
            width,
            height,
            ImagePixelFormat::Gray8,
            data,
            width as usize * ImagePixelFormat::Gray8.bytes_per_pixel(),
        )
    }

    /// Creates a new value.
    pub fn new(
        width: u32,
        height: u32,
        pixel_format: ImagePixelFormat,
        data: Vec<u8>,
        stride: usize,
    ) -> Result<Self> {
        let image = Self {
            width,
            height,
            pixel_format,
            data,
            stride,
        };
        image.as_view().validate()?;
        Ok(image)
    }

    /// Borrows this value as a view.
    pub fn as_view(&self) -> ImageView<'_> {
        ImageView {
            width: self.width,
            height: self.height,
            pixel_format: self.pixel_format,
            data: &self.data,
            stride: self.stride,
        }
    }

    /// Builds this value from video frame.
    pub fn from_video_frame(frame: &VideoFrame<'_>) -> Result<Self> {
        compact_image(&ImageView::new(
            frame.width,
            frame.height,
            frame.pixel_format.into(),
            frame.data,
            frame.stride,
        )?)
    }
}

#[derive(Debug, Clone, Copy)]
/// Data type for image batch view.
pub struct ImageBatchView<'a> {
    /// The batch size value.
    pub batch_size: usize,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// The pixel format value.
    pub pixel_format: ImagePixelFormat,
    /// Underlying data buffer.
    pub data: &'a [u8],
    /// The row stride value.
    pub row_stride: usize,
    /// The image stride value.
    pub image_stride: usize,
}

impl<'a> ImageBatchView<'a> {
    /// Returns packed.
    pub fn packed(
        batch_size: usize,
        width: u32,
        height: u32,
        pixel_format: ImagePixelFormat,
        data: &'a [u8],
    ) -> Result<Self> {
        let row_stride = width as usize * pixel_format.bytes_per_pixel();
        Self::new(
            batch_size,
            width,
            height,
            pixel_format,
            data,
            row_stride,
            row_stride * height as usize,
        )
    }

    /// Creates a new value.
    pub fn new(
        batch_size: usize,
        width: u32,
        height: u32,
        pixel_format: ImagePixelFormat,
        data: &'a [u8],
        row_stride: usize,
        image_stride: usize,
    ) -> Result<Self> {
        let batch = Self {
            batch_size,
            width,
            height,
            pixel_format,
            data,
            row_stride,
            image_stride,
        };
        batch.validate()?;
        Ok(batch)
    }

    /// Validates this value.
    pub fn validate(self) -> Result<()> {
        if self.batch_size == 0 {
            return Err(DetectError::InvalidArgument(
                "image batches must contain at least one image".to_string(),
            ));
        }
        let packed_width = self.width as usize * self.pixel_format.bytes_per_pixel();
        if self.row_stride < packed_width {
            return Err(DetectError::InvalidFrameBuffer {
                expected: packed_width,
                actual: self.row_stride,
            });
        }
        let packed_image = self.row_stride * self.height as usize;
        if self.image_stride < packed_image {
            return Err(DetectError::InvalidFrameBuffer {
                expected: packed_image,
                actual: self.image_stride,
            });
        }
        let expected = self.image_stride * self.batch_size;
        if self.data.len() < expected {
            return Err(DetectError::InvalidFrameBuffer {
                expected,
                actual: self.data.len(),
            });
        }
        Ok(())
    }

    /// Returns image.
    pub fn image(self, index: usize) -> Result<ImageView<'a>> {
        if index >= self.batch_size {
            return Err(DetectError::InvalidArgument(format!(
                "image batch index {index} is out of bounds for batch of size {}",
                self.batch_size
            )));
        }
        let start = index * self.image_stride;
        let end = start + self.image_stride;
        ImageView::new(
            self.width,
            self.height,
            self.pixel_format,
            &self.data[start..end],
            self.row_stride,
        )
    }

    /// Returns compact len.
    pub fn compact_len(self) -> usize {
        self.batch_size
            * self.width as usize
            * self.height as usize
            * self.pixel_format.bytes_per_pixel()
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for owned image batch.
pub struct OwnedImageBatch {
    /// The batch size value.
    pub batch_size: usize,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// The pixel format value.
    pub pixel_format: ImagePixelFormat,
    /// Underlying data buffer.
    pub data: Vec<u8>,
    /// The row stride value.
    pub row_stride: usize,
    /// The image stride value.
    pub image_stride: usize,
}

impl OwnedImageBatch {
    /// Returns packed.
    pub fn packed(
        batch_size: usize,
        width: u32,
        height: u32,
        pixel_format: ImagePixelFormat,
        data: Vec<u8>,
    ) -> Result<Self> {
        let row_stride = width as usize * pixel_format.bytes_per_pixel();
        Self::new(
            batch_size,
            width,
            height,
            pixel_format,
            data,
            row_stride,
            row_stride * height as usize,
        )
    }

    /// Creates a new value.
    pub fn new(
        batch_size: usize,
        width: u32,
        height: u32,
        pixel_format: ImagePixelFormat,
        data: Vec<u8>,
        row_stride: usize,
        image_stride: usize,
    ) -> Result<Self> {
        let batch = Self {
            batch_size,
            width,
            height,
            pixel_format,
            data,
            row_stride,
            image_stride,
        };
        batch.as_view().validate()?;
        Ok(batch)
    }

    /// Builds this value from images.
    pub fn from_images(images: &[OwnedImage]) -> Result<Self> {
        if images.is_empty() {
            return Err(DetectError::InvalidArgument(
                "image batch must contain at least one image".to_string(),
            ));
        }
        let first = &images[0];
        let row_stride = first.width as usize * first.pixel_format.bytes_per_pixel();
        let image_stride = row_stride * first.height as usize;
        let mut data = Vec::with_capacity(image_stride * images.len());
        for image in images {
            if image.width != first.width
                || image.height != first.height
                || image.pixel_format != first.pixel_format
            {
                return Err(DetectError::InvalidArgument(
                    "all images in a batch must share width, height, and pixel format".to_string(),
                ));
            }
            let compact = compact_image(&image.as_view())?;
            data.extend_from_slice(&compact.data);
        }
        Self::new(
            images.len(),
            first.width,
            first.height,
            first.pixel_format,
            data,
            row_stride,
            image_stride,
        )
    }

    /// Borrows this value as a view.
    pub fn as_view(&self) -> ImageBatchView<'_> {
        ImageBatchView {
            batch_size: self.batch_size,
            width: self.width,
            height: self.height,
            pixel_format: self.pixel_format,
            data: &self.data,
            row_stride: self.row_stride,
            image_stride: self.image_stride,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Data type for RGB mean.
pub struct RgbMean {
    /// The red value.
    pub red: f32,
    /// The green value.
    pub green: f32,
    /// The blue value.
    pub blue: f32,
}

/// Returns compact image.
pub fn compact_image(image: &ImageView<'_>) -> Result<OwnedImage> {
    image.validate()?;
    let bpp = image.pixel_format.bytes_per_pixel();
    let row_len = image.width as usize * bpp;
    let mut data = Vec::with_capacity(image.compact_len());
    for y in 0..image.height as usize {
        let start = y * image.stride;
        data.extend_from_slice(&image.data[start..start + row_len]);
    }
    OwnedImage::new(image.width, image.height, image.pixel_format, data, row_len)
}

/// Returns mean RGB.
pub fn mean_rgb(image: &ImageView<'_>) -> Result<RgbMean> {
    image.validate()?;
    let mut sum = [0_u64; 3];
    for y in 0..image.height {
        for x in 0..image.width {
            let pixel = image.pixel_rgb(x, y);
            sum[0] += pixel[0] as u64;
            sum[1] += pixel[1] as u64;
            sum[2] += pixel[2] as u64;
        }
    }
    let pixels = (image.width as f32) * (image.height as f32);
    Ok(RgbMean {
        red: sum[0] as f32 / pixels,
        green: sum[1] as f32 / pixels,
        blue: sum[2] as f32 / pixels,
    })
}

/// Returns luma histogram.
pub fn luma_histogram(image: &ImageView<'_>, bins: usize) -> Result<Vec<u64>> {
    image.validate()?;
    if bins == 0 || bins > 256 {
        return Err(DetectError::InvalidArgument(
            "histogram bins must be in the range 1..=256".to_string(),
        ));
    }
    let mut histogram = vec![0_u64; bins];
    for y in 0..image.height {
        for x in 0..image.width {
            let bin = (image.luma(x, y) as usize * bins).min(255 * bins) / 256;
            histogram[bin.min(bins - 1)] += 1;
        }
    }
    Ok(histogram)
}

/// Returns mask tensor from luma.
pub fn mask_tensor_from_luma(image: &ImageView<'_>) -> Result<F32Tensor> {
    image.validate()?;
    let mut values = Vec::with_capacity(image.width as usize * image.height as usize);
    for y in 0..image.height {
        for x in 0..image.width {
            values.push(image.luma(x, y) as f32 / 255.0);
        }
    }
    F32Tensor::from_dims([image.height as usize, image.width as usize], values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_luma_histogram() {
        let image = OwnedImage::new_rgb(2, 1, vec![0, 0, 0, 255, 255, 255]).unwrap();
        assert_eq!(luma_histogram(&image.as_view(), 2).unwrap(), vec![1, 1]);
    }

    #[test]
    fn compacts_padded_rows() {
        let view =
            ImageView::new(1, 2, ImagePixelFormat::Rgb24, &[1, 2, 3, 0, 4, 5, 6, 0], 4).unwrap();
        let compact = compact_image(&view).unwrap();
        assert_eq!(compact.stride, 3);
        assert_eq!(compact.data, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn packed_view_uses_tight_stride() {
        let view = ImageView::packed(2, 1, ImagePixelFormat::Gray8, &[0, 255]).unwrap();
        assert_eq!(view.stride, 2);
    }

    #[test]
    fn batches_images_from_existing_owned_images() {
        let first = OwnedImage::new_rgb(1, 1, vec![0, 0, 0]).unwrap();
        let second = OwnedImage::new_rgb(1, 1, vec![255, 255, 255]).unwrap();

        let batch = OwnedImageBatch::from_images(&[first, second]).unwrap();
        assert_eq!(batch.batch_size, 2);
        assert_eq!(
            batch.as_view().image(1).unwrap().pixel_rgb(0, 0),
            [255, 255, 255]
        );
    }

    #[test]
    fn converts_luma_into_mask_tensor() {
        let image = OwnedImage::new_rgb(1, 1, vec![255, 255, 255]).unwrap();
        let mask = mask_tensor_from_luma(&image.as_view()).unwrap();
        assert_eq!(mask.shape().dimensions(), &[1, 1]);
        assert_eq!(mask.values(), &[1.0]);
    }
}
