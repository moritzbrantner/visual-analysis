#![doc = include_str!("../README.md")]

pub mod contracts;
pub mod operations;
pub mod surface;
use image::imageops::FilterType;
use image::{DynamicImage, GrayImage, RgbImage};
use image_analysis_core::{
    compact_image, ImageBatchView, ImagePixelFormat, ImageView, OwnedImage, OwnedImageBatch,
};
use math_geometry_2d::RectU32;
use math_linear::Kernel2d;
use video_analysis_core::{DetectError, Result};

/// Default perceptual hash output size in bits per axis.
pub const PERCEPTUAL_HASH_SIZE: u32 = 8;

/// Default perceptual hash input scaling factor before DCT.
pub const PERCEPTUAL_HASH_HIGHFREQ_FACTOR: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Variants describing model tensor channel order.
pub enum ChannelOrder {
    /// RGB channel order.
    Rgb,
    /// BGR channel order.
    Bgr,
}

#[derive(Debug, Clone, PartialEq)]
/// Image preprocessing settings for local model tensors.
pub struct ImageModelPreprocessing {
    /// Model input width.
    pub input_width: u32,
    /// Model input height.
    pub input_height: u32,
    /// Pixel rescale factor applied before normalization.
    pub rescale_factor: f32,
    /// Channel mean.
    pub mean: [f32; 3],
    /// Channel standard deviation.
    pub std: [f32; 3],
    /// Tensor channel order.
    pub channel_order: ChannelOrder,
}

impl Default for ImageModelPreprocessing {
    fn default() -> Self {
        Self {
            input_width: 640,
            input_height: 640,
            rescale_factor: 1.0 / 255.0,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
            channel_order: ChannelOrder::Rgb,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Channel-first f32 tensor for one image.
pub struct ImageModelTensor {
    /// Channel count.
    pub channels: usize,
    /// Tensor width.
    pub width: u32,
    /// Tensor height.
    pub height: u32,
    /// Channel-first tensor values.
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
/// Channel-first f32 tensor for an image batch.
pub struct ImageModelBatchTensor {
    /// Batch size.
    pub batch_size: usize,
    /// Channel count.
    pub channels: usize,
    /// Tensor width.
    pub width: u32,
    /// Tensor height.
    pub height: u32,
    /// Batch tensor values.
    pub values: Vec<f32>,
}

/// Compatibility alias for callers migrating from older ONNX-specific names.
pub type OnnxImagePreprocessing = ImageModelPreprocessing;
/// Compatibility alias for callers migrating from older ONNX-specific names.
pub type OnnxImageTensor = ImageModelTensor;
/// Compatibility alias for callers migrating from older ONNX-specific names.
pub type OnnxImageBatchTensor = ImageModelBatchTensor;

/// Returns a model-ready tensor for one image.
pub fn image_to_model_tensor(
    image: &ImageView<'_>,
    options: &ImageModelPreprocessing,
) -> Result<ImageModelTensor> {
    validate_image_model_preprocessing(options)?;
    image.validate()?;
    let width = image.width;
    let height = image.height;
    let pixels = width as usize * height as usize;
    let mut values = vec![0.0_f32; 3 * pixels];
    for y in 0..height {
        for x in 0..width {
            let rgb = image.pixel_rgb(x, y);
            let channels = match options.channel_order {
                ChannelOrder::Rgb => [rgb[0], rgb[1], rgb[2]],
                ChannelOrder::Bgr => [rgb[2], rgb[1], rgb[0]],
            };
            let pixel_index = y as usize * width as usize + x as usize;
            for channel in 0..3 {
                let value = channels[channel] as f32 * options.rescale_factor;
                values[channel * pixels + pixel_index] =
                    (value - options.mean[channel]) / options.std[channel];
            }
        }
    }
    Ok(ImageModelTensor {
        channels: 3,
        width,
        height,
        values,
    })
}

/// Returns a resized and model-ready tensor for one image.
pub fn preprocess_image_for_model(
    image: &ImageView<'_>,
    options: &ImageModelPreprocessing,
) -> Result<ImageModelTensor> {
    validate_image_model_preprocessing(options)?;
    image.validate()?;
    let resized = if image.width == options.input_width && image.height == options.input_height {
        compact_image(image)?
    } else {
        resize_nearest(image, options.input_width, options.input_height)?
    };
    image_to_model_tensor(&resized.as_view(), options)
}

/// Returns a model-ready tensor for an image batch.
pub fn image_batch_to_model_tensor(
    batch: ImageBatchView<'_>,
    options: &ImageModelPreprocessing,
) -> Result<ImageModelBatchTensor> {
    validate_image_model_preprocessing(options)?;
    batch.validate()?;
    let pixels_per_image = batch.width as usize * batch.height as usize;
    let mut values = Vec::with_capacity(batch.batch_size * 3 * pixels_per_image);
    for index in 0..batch.batch_size {
        let image = batch.image(index)?;
        let tensor = image_to_model_tensor(&image, options)?;
        values.extend(tensor.values);
    }
    Ok(ImageModelBatchTensor {
        batch_size: batch.batch_size,
        channels: 3,
        width: batch.width,
        height: batch.height,
        values,
    })
}

/// Returns a resized and model-ready tensor for an image batch.
pub fn preprocess_image_batch_for_model(
    batch: ImageBatchView<'_>,
    options: &ImageModelPreprocessing,
) -> Result<ImageModelBatchTensor> {
    validate_image_model_preprocessing(options)?;
    batch.validate()?;
    if batch.width == options.input_width && batch.height == options.input_height {
        return image_batch_to_model_tensor(batch, options);
    }

    let mut resized = Vec::with_capacity(batch.batch_size);
    for index in 0..batch.batch_size {
        let image = batch.image(index)?;
        resized.push(resize_nearest(
            &image,
            options.input_width,
            options.input_height,
        )?);
    }
    let resized_batch = OwnedImageBatch::from_images(&resized)?;
    image_batch_to_model_tensor(resized_batch.as_view(), options)
}

/// Returns model preprocessing settings from a Hugging Face-style preprocessor config.
pub fn image_model_preprocessing_from_config(
    config: &serde_json::Value,
) -> Result<ImageModelPreprocessing> {
    let mut options = ImageModelPreprocessing::default();
    if let Some(size) = config.get("size") {
        if let Some(shortest) = get_u32(size, "shortest_edge") {
            options.input_width = shortest;
            options.input_height = shortest;
        }
        if let Some(width) = get_u32(size, "width") {
            options.input_width = width;
        }
        if let Some(height) = get_u32(size, "height") {
            options.input_height = height;
        }
    }
    if let Some(value) = config
        .get("rescale_factor")
        .and_then(serde_json::Value::as_f64)
    {
        options.rescale_factor = value as f32;
    }
    if let Some(values) = get_f32_array(config, "image_mean") {
        options.mean = values;
    }
    if let Some(values) = get_f32_array(config, "image_std") {
        options.std = values;
    }
    if let Some(order) = config
        .get("channel_order")
        .and_then(serde_json::Value::as_str)
    {
        options.channel_order = match order {
            "rgb" | "RGB" => ChannelOrder::Rgb,
            "bgr" | "BGR" => ChannelOrder::Bgr,
            other => {
                return Err(DetectError::InvalidArgument(format!(
                    "unsupported model channel order `{other}`"
                )))
            }
        };
    }
    Ok(options)
}

/// Compatibility wrapper for older preprocessing config helper names.
pub fn preprocessing_from_config(config: &serde_json::Value) -> Result<ImageModelPreprocessing> {
    image_model_preprocessing_from_config(config)
}

/// Validates model preprocessing settings.
pub fn validate_image_model_preprocessing(options: &ImageModelPreprocessing) -> Result<()> {
    if options.input_width == 0 || options.input_height == 0 {
        return Err(DetectError::InvalidDimensions {
            width: options.input_width,
            height: options.input_height,
        });
    }
    if !options.rescale_factor.is_finite() || options.rescale_factor <= 0.0 {
        return Err(DetectError::InvalidArgument(
            "image model rescale factor must be finite and greater than zero".to_string(),
        ));
    }
    if options
        .mean
        .iter()
        .chain(options.std.iter())
        .any(|value| !value.is_finite())
        || options.std.contains(&0.0)
    {
        return Err(DetectError::InvalidArgument(
            "image model mean/std values must be finite and std values must be non-zero"
                .to_string(),
        ));
    }
    Ok(())
}

fn get_u32(value: &serde_json::Value, key: &str) -> Option<u32> {
    value
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
}

fn get_f32_array(value: &serde_json::Value, key: &str) -> Option<[f32; 3]> {
    let array = value.get(key)?.as_array()?;
    if array.len() != 3 {
        return None;
    }
    Some([
        array[0].as_f64()? as f32,
        array[1].as_f64()? as f32,
        array[2].as_f64()? as f32,
    ])
}

/// Computes a compact DCT-based perceptual hash for an RGB image.
pub fn perceptual_hash_rgb(image: &RgbImage) -> String {
    let image_size = PERCEPTUAL_HASH_SIZE * PERCEPTUAL_HASH_HIGHFREQ_FACTOR;
    let luma = DynamicImage::ImageRgb8(image.clone())
        .grayscale()
        .resize_exact(image_size, image_size, FilterType::Lanczos3)
        .to_luma8();
    perceptual_hash_luma(&luma, PERCEPTUAL_HASH_SIZE)
}

/// Computes a compact DCT-based perceptual hash for a luma image.
pub fn perceptual_hash_luma(luma: &GrayImage, hash_size: u32) -> String {
    let rows = luma.height() as usize;
    let cols = luma.width() as usize;
    let hash_size = hash_size as usize;
    let pixels = luma
        .pixels()
        .map(|pixel| f64::from(pixel[0]))
        .collect::<Vec<_>>();
    let dct_rows = dct_axis(&pixels, rows, cols, 0);
    let dct = dct_axis(&dct_rows, rows, cols, 1);

    let mut low_frequency = Vec::with_capacity(hash_size * hash_size);
    for y in 0..hash_size {
        for x in 0..hash_size {
            let value = dct[y * cols + x];
            low_frequency.push(if value.abs() < 1.0e-6 { 0.0 } else { value });
        }
    }
    let mut sorted = low_frequency.clone();
    let threshold = median(&mut sorted);
    let bits = low_frequency
        .into_iter()
        .map(|value| value > threshold)
        .collect::<Vec<_>>();
    binary_hash_to_hex(&bits)
}

/// Returns the Hamming distance between two compact hexadecimal hashes.
pub fn hash_distance(left: &str, right: &str) -> std::result::Result<u32, String> {
    let left = parse_hex_hash(left)?;
    let right = parse_hex_hash(right)?;
    Ok((left ^ right).count_ones())
}

/// Returns whether two compact hexadecimal hashes are within `max_distance`.
pub fn is_near_duplicate(
    left: &str,
    right: &str,
    max_distance: u32,
) -> std::result::Result<bool, String> {
    Ok(hash_distance(left, right)? <= max_distance)
}

/// Compatibility-only image region type.
///
/// Prefer [`RectU32`] for new shared 2D APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageRegion {
    /// The x value.
    pub x: u32,
    /// The y value.
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl ImageRegion {
    /// Creates a new value.
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Result<Self> {
        let region = Self {
            x,
            y,
            width,
            height,
        };
        let _: RectU32 = region.into();
        region.to_rect().validate()?;
        Ok(region)
    }

    /// Converts this value to rect.
    pub fn to_rect(self) -> RectU32 {
        RectU32 {
            x: self.x,
            y: self.y,
            width: self.width,
            height: self.height,
        }
    }

    /// Builds this value from rect.
    pub fn from_rect(rect: RectU32) -> Self {
        Self {
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height,
        }
    }
}

impl From<ImageRegion> for RectU32 {
    fn from(value: ImageRegion) -> Self {
        value.to_rect()
    }
}

impl TryFrom<RectU32> for ImageRegion {
    type Error = DetectError;

    fn try_from(value: RectU32) -> Result<Self> {
        value.validate()?;
        Ok(Self::from_rect(value))
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Variants describing image operation.
pub enum ImageOperation {
    /// The crop variant.
    Crop(ImageRegion),
    /// The resize nearest variant.
    ResizeNearest {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
    },
    /// The box blur variant.
    BoxBlur {
        /// Blur radius in pixels.
        radius: u32,
    },
    /// The grayscale variant.
    Grayscale,
    /// The invert variant.
    Invert,
    /// The brightness contrast variant.
    BrightnessContrast {
        /// Brightness offset in channel values.
        brightness: i16,
        /// Contrast multiplier around channel midpoint.
        contrast: f32,
    },
    /// The saturation variant.
    Saturation {
        /// Saturation multiplier.
        saturation: f32,
    },
    /// The threshold variant.
    Threshold {
        /// The level value for this variant.
        level: u8,
    },
    /// The flip horizontal variant.
    FlipHorizontal,
    /// The flip vertical variant.
    FlipVertical,
    /// The rotate 90 variant.
    Rotate90 {
        /// Number of clockwise quarter turns.
        clockwise_turns: u8,
    },
    /// Adds deterministic high-frequency blue-noise-like channel variation.
    BlueNoise {
        /// Maximum absolute channel delta.
        amount: u8,
        /// Seed for deterministic noise generation.
        seed: u64,
    },
    /// Adds deterministic shot-noise-like Poisson variation.
    PoissonNoise {
        /// Expected photon count represented by a full-intensity channel.
        scale: f32,
        /// Seed for deterministic noise generation.
        seed: u64,
    },
    /// The convolve3x3 variant.
    Convolve3x3 {
        /// The kernel value for this variant.
        kernel: [f32; 9],
        /// The divisor value for this variant.
        divisor: f32,
        /// The bias value for this variant.
        bias: f32,
    },
}

/// Blend modes for compositing images without requiring alpha pixel formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    /// Source-over normal blend.
    Normal,
    /// Multiply blend.
    Multiply,
    /// Screen blend.
    Screen,
    /// Additive blend with clamping.
    Add,
    /// Absolute channel difference blend.
    Difference,
}

/// Placement and opacity settings for compositing an overlay image onto a base image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompositeSpec {
    /// Overlay x offset in base-image pixels.
    pub x: i32,
    /// Overlay y offset in base-image pixels.
    pub y: i32,
    /// Overall opacity in [0, 1].
    pub opacity: f32,
    /// Blend mode.
    pub blend_mode: BlendMode,
}

impl CompositeSpec {
    /// Creates a validated composite spec.
    pub fn new(x: i32, y: i32, opacity: f32, blend_mode: BlendMode) -> Result<Self> {
        let spec = Self {
            x,
            y,
            opacity,
            blend_mode,
        };
        spec.validate()?;
        Ok(spec)
    }

    fn validate(self) -> Result<()> {
        if self.opacity.is_finite() && (0.0..=1.0).contains(&self.opacity) {
            Ok(())
        } else {
            Err(DetectError::InvalidArgument(
                "composite opacity must be finite and in [0, 1]".to_string(),
            ))
        }
    }
}

impl ImageOperation {
    /// Returns crop rect.
    pub fn crop_rect(region: RectU32) -> Self {
        Self::Crop(ImageRegion::from_rect(region))
    }

    /// Returns convolve kernel2d.
    pub fn convolve_kernel2d(kernel: Kernel2d, divisor: f32, bias: f32) -> Result<Self> {
        Ok(Self::Convolve3x3 {
            kernel: kernel.as_array_3x3()?,
            divisor,
            bias,
        })
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
/// Data type for image processor.
pub struct ImageProcessor {
    operations: Vec<ImageOperation>,
}

impl ImageProcessor {
    /// Creates a new value.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns operation.
    pub fn operation(mut self, operation: ImageOperation) -> Self {
        self.operations.push(operation);
        self
    }

    /// Returns operations.
    pub fn operations(&self) -> &[ImageOperation] {
        &self.operations
    }

    /// Returns process.
    pub fn process(&self, image: &ImageView<'_>) -> Result<OwnedImage> {
        let mut current = image_analysis_core::compact_image(image)?;
        for operation in &self.operations {
            current = apply_operation(&current.as_view(), operation)?;
        }
        Ok(current)
    }
}

/// Returns apply operation.
pub fn apply_operation(image: &ImageView<'_>, operation: &ImageOperation) -> Result<OwnedImage> {
    match operation {
        ImageOperation::Crop(region) => crop_image(image, *region),
        ImageOperation::ResizeNearest { width, height } => resize_nearest(image, *width, *height),
        ImageOperation::BoxBlur { radius } => box_blur_image(image, *radius),
        ImageOperation::Grayscale => grayscale_image(image),
        ImageOperation::Invert => invert_image(image),
        ImageOperation::BrightnessContrast {
            brightness,
            contrast,
        } => brightness_contrast_image(image, *brightness, *contrast),
        ImageOperation::Saturation { saturation } => saturation_image(image, *saturation),
        ImageOperation::Threshold { level } => threshold_image(image, *level),
        ImageOperation::FlipHorizontal => flip_horizontal_image(image),
        ImageOperation::FlipVertical => flip_vertical_image(image),
        ImageOperation::Rotate90 { clockwise_turns } => rotate_image_90(image, *clockwise_turns),
        ImageOperation::BlueNoise { amount, seed } => add_blue_noise(image, *amount, *seed),
        ImageOperation::PoissonNoise { scale, seed } => add_poisson_noise(image, *scale, *seed),
        ImageOperation::Convolve3x3 {
            kernel,
            divisor,
            bias,
        } => convolve_3x3(image, *kernel, *divisor, *bias),
    }
}

/// Returns crop image.
pub fn crop_image(image: &ImageView<'_>, region: ImageRegion) -> Result<OwnedImage> {
    crop_image_rect(image, region.into())
}

/// Returns crop image rect.
pub fn crop_image_rect(image: &ImageView<'_>, region: RectU32) -> Result<OwnedImage> {
    image.validate()?;
    validate_region(image, region)?;
    let bpp = image.pixel_format.bytes_per_pixel();
    let stride = region.width as usize * bpp;
    let mut data = vec![0_u8; stride * region.height as usize];
    for y in 0..region.height {
        let src_start = (region.y + y) as usize * image.stride + region.x as usize * bpp;
        let dst_start = y as usize * stride;
        data[dst_start..dst_start + stride]
            .copy_from_slice(&image.data[src_start..src_start + stride]);
    }
    OwnedImage::new(
        region.width,
        region.height,
        image.pixel_format,
        data,
        stride,
    )
}

/// Returns resize nearest.
pub fn resize_nearest(image: &ImageView<'_>, width: u32, height: u32) -> Result<OwnedImage> {
    image.validate()?;
    if width == 0 || height == 0 {
        return Err(DetectError::InvalidDimensions { width, height });
    }
    let bpp = image.pixel_format.bytes_per_pixel();
    let stride = width as usize * bpp;
    let mut data = vec![0_u8; stride * height as usize];
    for y in 0..height {
        let src_y = (y as u64 * image.height as u64 / height as u64) as u32;
        for x in 0..width {
            let src_x = (x as u64 * image.width as u64 / width as u64) as u32;
            write_pixel(
                &mut data,
                stride,
                image.pixel_format,
                x,
                y,
                image.pixel_rgb(src_x, src_y),
            );
        }
    }
    OwnedImage::new(width, height, image.pixel_format, data, stride)
}

/// Returns box blur image.
pub fn box_blur_image(image: &ImageView<'_>, radius: u32) -> Result<OwnedImage> {
    if radius == 0 {
        return compact_image(image);
    }
    map_pixels(image, image.pixel_format, |source, x, y| {
        let x0 = x.saturating_sub(radius);
        let y0 = y.saturating_sub(radius);
        let x1 = (x + radius).min(source.width - 1);
        let y1 = (y + radius).min(source.height - 1);
        let mut sum = [0u32; 3];
        let mut count = 0u32;
        for sample_y in y0..=y1 {
            for sample_x in x0..=x1 {
                let pixel = source.pixel_rgb(sample_x, sample_y);
                sum[0] += pixel[0] as u32;
                sum[1] += pixel[1] as u32;
                sum[2] += pixel[2] as u32;
                count += 1;
            }
        }
        [
            (sum[0] / count) as u8,
            (sum[1] / count) as u8,
            (sum[2] / count) as u8,
        ]
    })
}

/// Returns grayscale image.
pub fn grayscale_image(image: &ImageView<'_>) -> Result<OwnedImage> {
    map_pixels(image, ImagePixelFormat::Gray8, |source, x, y| {
        [source.luma(x, y); 3]
    })
}

/// Returns invert image.
pub fn invert_image(image: &ImageView<'_>) -> Result<OwnedImage> {
    map_pixels(image, image.pixel_format, |source, x, y| {
        let [red, green, blue] = source.pixel_rgb(x, y);
        [255 - red, 255 - green, 255 - blue]
    })
}

/// Returns brightness contrast image.
pub fn brightness_contrast_image(
    image: &ImageView<'_>,
    brightness: i16,
    contrast: f32,
) -> Result<OwnedImage> {
    if !contrast.is_finite() {
        return Err(DetectError::InvalidArgument(
            "contrast must be finite".to_string(),
        ));
    }
    map_pixels(image, image.pixel_format, |source, x, y| {
        let [red, green, blue] = source.pixel_rgb(x, y);
        [
            adjust_channel(red, brightness, contrast),
            adjust_channel(green, brightness, contrast),
            adjust_channel(blue, brightness, contrast),
        ]
    })
}

/// Returns saturation adjusted image.
pub fn saturation_image(image: &ImageView<'_>, saturation: f32) -> Result<OwnedImage> {
    if !saturation.is_finite() {
        return Err(DetectError::InvalidArgument(
            "saturation must be finite".to_string(),
        ));
    }
    map_pixels(image, image.pixel_format, |source, x, y| {
        let [red, green, blue] = source.pixel_rgb(x, y);
        let luma = source.luma(x, y) as f32;
        [
            clamp_u8(luma + (red as f32 - luma) * saturation),
            clamp_u8(luma + (green as f32 - luma) * saturation),
            clamp_u8(luma + (blue as f32 - luma) * saturation),
        ]
    })
}

/// Returns threshold image.
pub fn threshold_image(image: &ImageView<'_>, level: u8) -> Result<OwnedImage> {
    map_pixels(image, ImagePixelFormat::Gray8, |source, x, y| {
        let value = if source.luma(x, y) >= level { 255 } else { 0 };
        [value, value, value]
    })
}

/// Returns horizontally flipped image.
pub fn flip_horizontal_image(image: &ImageView<'_>) -> Result<OwnedImage> {
    map_pixels(image, image.pixel_format, |source, x, y| {
        source.pixel_rgb(source.width - 1 - x, y)
    })
}

/// Returns vertically flipped image.
pub fn flip_vertical_image(image: &ImageView<'_>) -> Result<OwnedImage> {
    map_pixels(image, image.pixel_format, |source, x, y| {
        source.pixel_rgb(x, source.height - 1 - y)
    })
}

/// Returns image rotated by clockwise quarter turns.
pub fn rotate_image_90(image: &ImageView<'_>, clockwise_turns: u8) -> Result<OwnedImage> {
    image.validate()?;
    let turns = clockwise_turns % 4;
    if turns == 0 {
        return compact_image(image);
    }
    let (width, height) = if turns == 2 {
        (image.width, image.height)
    } else {
        (image.height, image.width)
    };
    let bpp = image.pixel_format.bytes_per_pixel();
    let stride = width as usize * bpp;
    let mut data = vec![0_u8; stride * height as usize];
    for y in 0..height {
        for x in 0..width {
            let (source_x, source_y) = match turns {
                1 => (y, image.height - 1 - x),
                2 => (image.width - 1 - x, image.height - 1 - y),
                3 => (image.width - 1 - y, x),
                _ => unreachable!(),
            };
            write_pixel(
                &mut data,
                stride,
                image.pixel_format,
                x,
                y,
                image.pixel_rgb(source_x, source_y),
            );
        }
    }
    OwnedImage::new(width, height, image.pixel_format, data, stride)
}

/// Adds deterministic high-frequency blue-noise-like variation to each channel.
pub fn add_blue_noise(image: &ImageView<'_>, amount: u8, seed: u64) -> Result<OwnedImage> {
    if amount == 0 {
        return compact_image(image);
    }
    map_pixels(image, image.pixel_format, |source, x, y| {
        let pixel = source.pixel_rgb(x, y);
        [
            add_blue_noise_channel(pixel[0], amount, seed, x, y, 0),
            add_blue_noise_channel(pixel[1], amount, seed, x, y, 1),
            add_blue_noise_channel(pixel[2], amount, seed, x, y, 2),
        ]
    })
}

/// Adds deterministic shot-noise-like Poisson variation to each channel.
///
/// `scale` is the expected sample count for a full-intensity channel. Larger
/// values produce lower relative noise.
pub fn add_poisson_noise(image: &ImageView<'_>, scale: f32, seed: u64) -> Result<OwnedImage> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(DetectError::InvalidArgument(
            "poisson noise scale must be finite and greater than zero".to_string(),
        ));
    }
    map_pixels(image, image.pixel_format, |source, x, y| {
        let pixel = source.pixel_rgb(x, y);
        [
            add_poisson_noise_channel(pixel[0], scale, seed, x, y, 0),
            add_poisson_noise_channel(pixel[1], scale, seed, x, y, 1),
            add_poisson_noise_channel(pixel[2], scale, seed, x, y, 2),
        ]
    })
}

/// Returns convolve 3x3.
pub fn convolve_3x3(
    image: &ImageView<'_>,
    kernel: [f32; 9],
    divisor: f32,
    bias: f32,
) -> Result<OwnedImage> {
    convolve_3x3_kernel(image, &Kernel2d::from(kernel), divisor, bias)
}

/// Returns convolve 3x3 kernel.
pub fn convolve_3x3_kernel(
    image: &ImageView<'_>,
    kernel: &Kernel2d,
    divisor: f32,
    bias: f32,
) -> Result<OwnedImage> {
    if !divisor.is_finite() || divisor == 0.0 || !bias.is_finite() {
        return Err(DetectError::InvalidArgument(
            "convolution divisor must be finite and non-zero, and bias must be finite".to_string(),
        ));
    }
    let kernel = kernel.as_array_3x3()?;
    map_pixels(image, image.pixel_format, |source, x, y| {
        let mut output = [0.0_f32; 3];
        for ky in 0..3 {
            for kx in 0..3 {
                let sx = (x as i64 + kx as i64 - 1).clamp(0, source.width as i64 - 1) as u32;
                let sy = (y as i64 + ky as i64 - 1).clamp(0, source.height as i64 - 1) as u32;
                let pixel = source.pixel_rgb(sx, sy);
                let weight = kernel[ky * 3 + kx];
                output[0] += pixel[0] as f32 * weight;
                output[1] += pixel[1] as f32 * weight;
                output[2] += pixel[2] as f32 * weight;
            }
        }
        [
            clamp_u8(output[0] / divisor + bias),
            clamp_u8(output[1] / divisor + bias),
            clamp_u8(output[2] / divisor + bias),
        ]
    })
}

/// Returns sharpen image.
pub fn sharpen_image(image: &ImageView<'_>) -> Result<OwnedImage> {
    convolve_3x3_kernel(image, &Kernel2d::sharpen_3x3(), 1.0, 0.0)
}

/// Composites an overlay image onto a base image using opacity, optional mask, and blend mode.
pub fn composite_image(
    base: &ImageView<'_>,
    overlay: &ImageView<'_>,
    mask: Option<ImageView<'_>>,
    spec: CompositeSpec,
) -> Result<OwnedImage> {
    base.validate()?;
    overlay.validate()?;
    spec.validate()?;
    if let Some(mask) = mask {
        mask.validate()?;
        if mask.width != overlay.width || mask.height != overlay.height {
            return Err(DetectError::InvalidArgument(
                "composite mask dimensions must match overlay dimensions".to_string(),
            ));
        }
    }

    let mut output = compact_image(base)?;
    let x0 = spec.x.max(0) as u32;
    let y0 = spec.y.max(0) as u32;
    let x1 = (spec.x as i64 + overlay.width as i64)
        .min(base.width as i64)
        .max(0) as u32;
    let y1 = (spec.y as i64 + overlay.height as i64)
        .min(base.height as i64)
        .max(0) as u32;
    if x0 >= x1 || y0 >= y1 || spec.opacity == 0.0 {
        return Ok(output);
    }

    for y in y0..y1 {
        for x in x0..x1 {
            let overlay_x = (x as i64 - spec.x as i64) as u32;
            let overlay_y = (y as i64 - spec.y as i64) as u32;
            let mask_alpha = mask
                .map(|mask| mask.luma(overlay_x, overlay_y) as f32 / 255.0)
                .unwrap_or(1.0);
            let alpha = spec.opacity * mask_alpha;
            if alpha <= 0.0 {
                continue;
            }
            let base_rgb = base.pixel_rgb(x, y);
            let overlay_rgb = overlay.pixel_rgb(overlay_x, overlay_y);
            let blended = blend_rgb(base_rgb, overlay_rgb, spec.blend_mode);
            let rgb = [
                clamp_u8(base_rgb[0] as f32 * (1.0 - alpha) + blended[0] as f32 * alpha),
                clamp_u8(base_rgb[1] as f32 * (1.0 - alpha) + blended[1] as f32 * alpha),
                clamp_u8(base_rgb[2] as f32 * (1.0 - alpha) + blended[2] as f32 * alpha),
            ];
            write_pixel(
                &mut output.data,
                output.stride,
                output.pixel_format,
                x,
                y,
                rgb,
            );
        }
    }
    Ok(output)
}

fn map_pixels(
    image: &ImageView<'_>,
    output_format: ImagePixelFormat,
    mut map: impl FnMut(ImageView<'_>, u32, u32) -> [u8; 3],
) -> Result<OwnedImage> {
    image.validate()?;
    let bpp = output_format.bytes_per_pixel();
    let stride = image.width as usize * bpp;
    let mut data = vec![0_u8; stride * image.height as usize];
    for y in 0..image.height {
        for x in 0..image.width {
            write_pixel(&mut data, stride, output_format, x, y, map(*image, x, y));
        }
    }
    OwnedImage::new(image.width, image.height, output_format, data, stride)
}

fn write_pixel(
    data: &mut [u8],
    stride: usize,
    format: ImagePixelFormat,
    x: u32,
    y: u32,
    rgb: [u8; 3],
) {
    let offset = y as usize * stride + x as usize * format.bytes_per_pixel();
    match format {
        ImagePixelFormat::Rgb24 => data[offset..offset + 3].copy_from_slice(&rgb),
        ImagePixelFormat::Bgr24 => {
            data[offset..offset + 3].copy_from_slice(&[rgb[2], rgb[1], rgb[0]])
        }
        ImagePixelFormat::Gray8 => data[offset] = rgb[0],
    }
}

fn validate_region(image: &ImageView<'_>, region: RectU32) -> Result<()> {
    region.validate()?;
    let max_x = region.max_x()?;
    let max_y = region.max_y()?;
    if max_x > image.width || max_y > image.height {
        return Err(DetectError::InvalidArgument(
            "crop region must be contained by the image".to_string(),
        ));
    }
    Ok(())
}

fn adjust_channel(value: u8, brightness: i16, contrast: f32) -> u8 {
    clamp_u8((value as f32 - 128.0) * contrast + 128.0 + brightness as f32)
}

fn blend_rgb(base: [u8; 3], overlay: [u8; 3], mode: BlendMode) -> [u8; 3] {
    let mut output = [0_u8; 3];
    for channel in 0..3 {
        let base = base[channel] as u16;
        let overlay = overlay[channel] as u16;
        output[channel] = match mode {
            BlendMode::Normal => overlay as u8,
            BlendMode::Multiply => ((base * overlay) / 255) as u8,
            BlendMode::Screen => (255 - ((255 - base) * (255 - overlay)) / 255) as u8,
            BlendMode::Add => (base + overlay).min(255) as u8,
            BlendMode::Difference => base.abs_diff(overlay) as u8,
        };
    }
    output
}

fn add_blue_noise_channel(value: u8, amount: u8, seed: u64, x: u32, y: u32, channel: u8) -> u8 {
    let x = i64::from(x);
    let y = i64::from(y);
    let center = signed_noise(seed, x, y, channel, 0);
    let neighbor_mean = (signed_noise(seed, x - 1, y, channel, 0)
        + signed_noise(seed, x + 1, y, channel, 0)
        + signed_noise(seed, x, y - 1, channel, 0)
        + signed_noise(seed, x, y + 1, channel, 0))
        * 0.25;
    let high_frequency = (center - neighbor_mean).clamp(-1.0, 1.0);
    clamp_u8(value as f32 + high_frequency * amount as f32)
}

fn add_poisson_noise_channel(value: u8, scale: f32, seed: u64, x: u32, y: u32, channel: u8) -> u8 {
    if value == 0 {
        return 0;
    }
    let lambda = value as f32 / 255.0 * scale;
    let sampled = sample_poisson(lambda, seed, x, y, channel);
    clamp_u8(sampled / scale * 255.0)
}

fn sample_poisson(lambda: f32, seed: u64, x: u32, y: u32, channel: u8) -> f32 {
    if lambda <= 64.0 {
        let limit = (-lambda).exp();
        let mut product = 1.0_f32;
        let mut count = 0_u32;
        loop {
            product *= uniform01(seed, i64::from(x), i64::from(y), channel, count);
            if product <= limit {
                return count as f32;
            }
            count += 1;
        }
    }

    let u1 = uniform01(seed, i64::from(x), i64::from(y), channel, 0).max(f32::MIN_POSITIVE);
    let u2 = uniform01(seed, i64::from(x), i64::from(y), channel, 1);
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    (lambda + lambda.sqrt() * z).round().max(0.0)
}

fn signed_noise(seed: u64, x: i64, y: i64, channel: u8, sample: u32) -> f32 {
    uniform01(seed, x, y, channel, sample) * 2.0 - 1.0
}

fn uniform01(seed: u64, x: i64, y: i64, channel: u8, sample: u32) -> f32 {
    let state = seed
        ^ (x as u64).wrapping_mul(0x9e3779b97f4a7c15)
        ^ (y as u64).wrapping_mul(0xbf58476d1ce4e5b9)
        ^ u64::from(channel).wrapping_mul(0x94d049bb133111eb)
        ^ u64::from(sample).wrapping_mul(0xd6e8feb86659fd93);
    let mixed = splitmix64(state);
    (((mixed >> 32) as u32) as f32 + 0.5) / (u32::MAX as f32 + 1.0)
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn clamp_u8(value: f32) -> u8 {
    value.round().clamp(0.0, 255.0) as u8
}

fn parse_hex_hash(value: &str) -> std::result::Result<u64, String> {
    if value.len() != 16 {
        return Err("image hash must be a 16 character hexadecimal string".to_string());
    }
    u64::from_str_radix(value, 16)
        .map_err(|_| "image hash must be a 16 character hexadecimal string".to_string())
}

fn dct_axis(values: &[f64], rows: usize, cols: usize, axis: usize) -> Vec<f64> {
    let mut output = vec![0.0; values.len()];
    if axis == 0 {
        for freq_y in 0..rows {
            for x in 0..cols {
                let mut sum = 0.0;
                for y in 0..rows {
                    let angle = std::f64::consts::PI * freq_y as f64 * (2 * y + 1) as f64
                        / (2 * rows) as f64;
                    sum += values[y * cols + x] * angle.cos();
                }
                output[freq_y * cols + x] = 2.0 * sum;
            }
        }
    } else {
        for y in 0..rows {
            for freq_x in 0..cols {
                let mut sum = 0.0;
                for x in 0..cols {
                    let angle = std::f64::consts::PI * freq_x as f64 * (2 * x + 1) as f64
                        / (2 * cols) as f64;
                    sum += values[y * cols + x] * angle.cos();
                }
                output[y * cols + freq_x] = 2.0 * sum;
            }
        }
    }
    output
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|left, right| left.total_cmp(right));
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

fn binary_hash_to_hex(bits: &[bool]) -> String {
    let mut hash = 0_u64;
    for bit in bits {
        hash <<= 1;
        if *bit {
            hash |= 1;
        }
    }
    let width = bits.len().div_ceil(4);
    format!("{hash:0width$x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn image() -> OwnedImage {
        OwnedImage::new(
            2,
            2,
            ImagePixelFormat::Rgb24,
            vec![0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255],
            6,
        )
        .unwrap()
    }

    #[test]
    fn grayscales_to_single_channel() {
        let gray = grayscale_image(&image().as_view()).unwrap();
        assert_eq!(gray.pixel_format, ImagePixelFormat::Gray8);
        assert_eq!(gray.data.len(), 4);
    }

    #[test]
    fn processor_chains_operations() {
        let processed = ImageProcessor::new()
            .operation(ImageOperation::Crop(ImageRegion {
                x: 0,
                y: 0,
                width: 1,
                height: 2,
            }))
            .operation(ImageOperation::ResizeNearest {
                width: 2,
                height: 2,
            })
            .process(&image().as_view())
            .unwrap();
        assert_eq!((processed.width, processed.height), (2, 2));
    }

    #[test]
    fn editor_style_operations_transform_pixels() {
        let flipped = flip_horizontal_image(&image().as_view()).unwrap();
        assert_eq!(&flipped.data[0..3], &[255, 0, 0]);

        let rotated = rotate_image_90(&image().as_view(), 1).unwrap();
        assert_eq!((rotated.width, rotated.height), (2, 2));
        assert_eq!(&rotated.data[0..3], &[0, 255, 0]);

        let bright = brightness_contrast_image(&image().as_view(), 10, 1.0).unwrap();
        assert_eq!(&bright.data[0..3], &[10, 10, 10]);

        let saturated = saturation_image(&image().as_view(), 0.0).unwrap();
        assert_eq!(&saturated.data[6..9], &[150, 150, 150]);

        let blurred = box_blur_image(&image().as_view(), 1).unwrap();
        assert_eq!(&blurred.data[0..3], &[63, 63, 63]);
    }

    #[test]
    fn composite_supports_mask_opacity_and_negative_offsets() {
        let base = OwnedImage::new_rgb(2, 2, vec![10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
            .unwrap();
        let overlay = OwnedImage::new_rgb(
            2,
            2,
            vec![110, 10, 10, 110, 10, 10, 110, 10, 10, 110, 10, 10],
        )
        .unwrap();
        let mask = OwnedImage::new_gray(2, 2, vec![0, 255, 255, 255]).unwrap();
        let output = composite_image(
            &base.as_view(),
            &overlay.as_view(),
            Some(mask.as_view()),
            CompositeSpec::new(-1, 0, 0.5, BlendMode::Normal).unwrap(),
        )
        .unwrap();

        assert_eq!(&output.data[0..3], &[60, 10, 10]);
        assert_eq!(&output.data[3..6], &[10, 10, 10]);

        let bad_mask = OwnedImage::new_gray(1, 1, vec![255]).unwrap();
        assert!(composite_image(
            &base.as_view(),
            &overlay.as_view(),
            Some(bad_mask.as_view()),
            CompositeSpec::new(0, 0, 1.0, BlendMode::Normal).unwrap(),
        )
        .is_err());
    }

    #[test]
    fn shared_rect_and_kernel_helpers_round_trip() {
        let cropped =
            crop_image_rect(&image().as_view(), RectU32::new(0, 0, 2, 2).unwrap()).unwrap();
        assert_eq!(cropped.width, 2);
        let operation =
            ImageOperation::convolve_kernel2d(Kernel2d::identity_3x3(), 1.0, 0.0).unwrap();
        match operation {
            ImageOperation::Convolve3x3 { kernel, .. } => assert_eq!(kernel[4], 1.0),
            _ => panic!("expected convolve operation"),
        }
    }

    #[test]
    fn blue_noise_is_seeded_and_preserves_dimensions() {
        let source = image();
        let first = add_blue_noise(&source.as_view(), 12, 7).unwrap();
        let second = add_blue_noise(&source.as_view(), 12, 7).unwrap();
        assert_eq!(first, second);
        assert_eq!((first.width, first.height), (2, 2));
        assert_eq!(first.pixel_format, ImagePixelFormat::Rgb24);
        assert_ne!(first.data, source.data);

        let unchanged = add_blue_noise(&source.as_view(), 0, 7).unwrap();
        assert_eq!(unchanged, source);
    }

    #[test]
    fn poisson_noise_is_seeded_and_validates_scale() {
        let source = image();
        let first = add_poisson_noise(&source.as_view(), 32.0, 11).unwrap();
        let second = add_poisson_noise(&source.as_view(), 32.0, 11).unwrap();
        assert_eq!(first, second);
        assert_eq!(&first.data[0..3], &[0, 0, 0]);
        assert_ne!(first.data, source.data);
        assert!(add_poisson_noise(&source.as_view(), 0.0, 11).is_err());
    }

    #[test]
    fn perceptual_hash_is_compact_hex() {
        let image = ImageBuffer::from_pixel(32, 32, Rgb([128, 64, 32]));
        assert_eq!(perceptual_hash_rgb(&image).len(), 16);
    }

    #[test]
    fn hash_distance_counts_bits() {
        assert_eq!(
            hash_distance("0000000000000000", "ffffffffffffffff").unwrap(),
            64
        );
        assert!(is_near_duplicate("0000000000000000", "0000000000000001", 1).unwrap());
        assert!(!is_near_duplicate("0000000000000000", "0000000000000001", 0).unwrap());
    }
}
