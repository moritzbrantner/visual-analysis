#![doc = include_str!("../README.md")]

pub mod contracts;
pub mod operations;
pub mod surface;
use math_geometry_2d::RectU32;
use math_linear::Kernel2d;
use video_analysis_core::{
    BoundingBox, DetectError, OwnedVideoFrame, PixelFormat, Result, VideoFrame,
};

#[derive(Debug, Clone, PartialEq)]
/// Variants describing frame edit.
pub enum FrameEdit {
    /// The crop variant.
    Crop(BoundingBox),
    /// The resize nearest variant.
    ResizeNearest {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
    },
    /// The box blur variant.
    BoxBlur {
        /// The radius value for this variant.
        radius: u32,
    },
    /// The grayscale variant.
    Grayscale,
    /// The invert variant.
    Invert,
    /// The flip horizontal variant.
    FlipHorizontal,
    /// The flip vertical variant.
    FlipVertical,
    /// The rotate 90 variant.
    Rotate90 {
        /// Number of clockwise quarter turns.
        clockwise_turns: u8,
    },
    /// The fade to color variant.
    FadeToColor {
        /// Target RGB color.
        color: [u8; 3],
        /// Blend amount in [0, 1].
        amount: f32,
    },
    /// The brightness contrast variant.
    BrightnessContrast {
        /// The brightness value for this variant.
        brightness: i16,
        /// The contrast value for this variant.
        contrast: f32,
    },
    /// The filter3x3 variant.
    Filter3x3 {
        /// The kernel value for this variant.
        kernel: [f32; 9],
        /// The divisor value for this variant.
        divisor: f32,
        /// The bias value for this variant.
        bias: f32,
    },
}

#[derive(Debug, Clone, Default, PartialEq)]
/// Data type for frame editor.
pub struct FrameEditor {
    edits: Vec<FrameEdit>,
}

impl FrameEditor {
    /// Creates a new value.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns edit.
    pub fn edit(mut self, edit: FrameEdit) -> Self {
        self.edits.push(edit);
        self
    }

    /// Returns crop.
    pub fn crop(self, region: BoundingBox) -> Self {
        self.edit(FrameEdit::Crop(region))
    }

    /// Returns crop rect.
    pub fn crop_rect(self, region: RectU32) -> Result<Self> {
        Ok(self.crop(region.try_into()?))
    }

    /// Returns resize nearest.
    pub fn resize_nearest(self, width: u32, height: u32) -> Self {
        self.edit(FrameEdit::ResizeNearest { width, height })
    }

    /// Returns box blur.
    pub fn box_blur(self, radius: u32) -> Self {
        self.edit(FrameEdit::BoxBlur { radius })
    }

    /// Returns grayscale.
    pub fn grayscale(self) -> Self {
        self.edit(FrameEdit::Grayscale)
    }

    /// Returns invert.
    pub fn invert(self) -> Self {
        self.edit(FrameEdit::Invert)
    }

    /// Returns flip horizontal.
    pub fn flip_horizontal(self) -> Self {
        self.edit(FrameEdit::FlipHorizontal)
    }

    /// Returns flip vertical.
    pub fn flip_vertical(self) -> Self {
        self.edit(FrameEdit::FlipVertical)
    }

    /// Returns rotate 90.
    pub fn rotate_90(self, clockwise_turns: u8) -> Self {
        self.edit(FrameEdit::Rotate90 { clockwise_turns })
    }

    /// Returns fade to color.
    pub fn fade_to_color(self, color: [u8; 3], amount: f32) -> Self {
        self.edit(FrameEdit::FadeToColor { color, amount })
    }

    /// Returns brightness contrast.
    pub fn brightness_contrast(self, brightness: i16, contrast: f32) -> Self {
        self.edit(FrameEdit::BrightnessContrast {
            brightness,
            contrast,
        })
    }

    /// Returns filter 3x3.
    pub fn filter_3x3(self, kernel: [f32; 9], divisor: f32, bias: f32) -> Self {
        self.edit(FrameEdit::Filter3x3 {
            kernel,
            divisor,
            bias,
        })
    }

    /// Returns filter 3x3 kernel.
    pub fn filter_3x3_kernel(self, kernel: Kernel2d, divisor: f32, bias: f32) -> Result<Self> {
        Ok(self.filter_3x3(kernel.as_array_3x3()?, divisor, bias))
    }

    /// Returns edits.
    pub fn edits(&self) -> &[FrameEdit] {
        &self.edits
    }

    /// Returns apply.
    pub fn apply(&self, frame: &VideoFrame<'_>) -> Result<OwnedVideoFrame> {
        let mut current = compact_frame(frame);
        for edit in &self.edits {
            current = apply_edit(&current.as_frame(), edit)?;
        }
        Ok(current)
    }
}

/// Returns crop frame.
pub fn crop_frame(frame: &VideoFrame<'_>, region: BoundingBox) -> Result<OwnedVideoFrame> {
    crop_frame_rect(frame, region.into())
}

/// Returns crop frame rect.
pub fn crop_frame_rect(frame: &VideoFrame<'_>, region: RectU32) -> Result<OwnedVideoFrame> {
    validate_region(frame, region)?;
    let pixel_format = frame.pixel_format;
    let stride = region.width as usize * 3;
    let mut data = vec![0; stride * region.height as usize];
    for y in 0..region.height {
        let src_start = (region.y + y) as usize * frame.stride + region.x as usize * 3;
        let dst_start = y as usize * stride;
        data[dst_start..dst_start + stride]
            .copy_from_slice(&frame.data[src_start..src_start + stride]);
    }
    Ok(OwnedVideoFrame {
        position: frame.position,
        width: region.width,
        height: region.height,
        pixel_format,
        data,
        stride,
    })
}

/// Returns resize nearest frame.
pub fn resize_nearest_frame(
    frame: &VideoFrame<'_>,
    width: u32,
    height: u32,
) -> Result<OwnedVideoFrame> {
    if width == 0 || height == 0 {
        return Err(DetectError::InvalidDimensions { width, height });
    }
    let stride = width as usize * 3;
    let mut data = vec![0; stride * height as usize];
    for y in 0..height {
        let src_y = (y as u64 * frame.height as u64 / height as u64) as u32;
        for x in 0..width {
            let src_x = (x as u64 * frame.width as u64 / width as u64) as u32;
            write_native_pixel(
                &mut data,
                stride,
                frame.pixel_format,
                x,
                y,
                frame.pixel_rgb(src_x, src_y),
            );
        }
    }
    Ok(OwnedVideoFrame {
        position: frame.position,
        width,
        height,
        pixel_format: frame.pixel_format,
        data,
        stride,
    })
}

/// Returns box blur frame.
pub fn box_blur_frame(frame: &VideoFrame<'_>, radius: u32) -> Result<OwnedVideoFrame> {
    if radius == 0 {
        return Ok(compact_frame(frame));
    }
    map_pixels(frame, |x, y| {
        let x0 = x.saturating_sub(radius);
        let y0 = y.saturating_sub(radius);
        let x1 = (x + radius).min(frame.width - 1);
        let y1 = (y + radius).min(frame.height - 1);
        let mut sum = [0u32; 3];
        let mut count = 0u32;
        for sample_y in y0..=y1 {
            for sample_x in x0..=x1 {
                let pixel = frame.pixel_rgb(sample_x, sample_y);
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

/// Returns grayscale frame.
pub fn grayscale_frame(frame: &VideoFrame<'_>) -> Result<OwnedVideoFrame> {
    map_pixels(frame, |x, y| {
        let [red, green, blue] = frame.pixel_rgb(x, y);
        let luma = (0.299 * red as f32 + 0.587 * green as f32 + 0.114 * blue as f32).round() as u8;
        [luma, luma, luma]
    })
}

/// Returns invert frame.
pub fn invert_frame(frame: &VideoFrame<'_>) -> Result<OwnedVideoFrame> {
    map_pixels(frame, |x, y| {
        let [red, green, blue] = frame.pixel_rgb(x, y);
        [255 - red, 255 - green, 255 - blue]
    })
}

/// Returns horizontally flipped frame.
pub fn flip_horizontal_frame(frame: &VideoFrame<'_>) -> Result<OwnedVideoFrame> {
    map_pixels(frame, |x, y| frame.pixel_rgb(frame.width - 1 - x, y))
}

/// Returns vertically flipped frame.
pub fn flip_vertical_frame(frame: &VideoFrame<'_>) -> Result<OwnedVideoFrame> {
    map_pixels(frame, |x, y| frame.pixel_rgb(x, frame.height - 1 - y))
}

/// Returns frame rotated by clockwise quarter turns.
pub fn rotate_frame_90(frame: &VideoFrame<'_>, clockwise_turns: u8) -> Result<OwnedVideoFrame> {
    let turns = clockwise_turns % 4;
    if turns == 0 {
        return Ok(compact_frame(frame));
    }
    let (width, height) = if turns == 2 {
        (frame.width, frame.height)
    } else {
        (frame.height, frame.width)
    };
    let stride = width as usize * 3;
    let mut data = vec![0; stride * height as usize];
    for y in 0..height {
        for x in 0..width {
            let (source_x, source_y) = match turns {
                1 => (y, frame.height - 1 - x),
                2 => (frame.width - 1 - x, frame.height - 1 - y),
                3 => (frame.width - 1 - y, x),
                _ => unreachable!(),
            };
            write_native_pixel(
                &mut data,
                stride,
                frame.pixel_format,
                x,
                y,
                frame.pixel_rgb(source_x, source_y),
            );
        }
    }
    Ok(OwnedVideoFrame {
        position: frame.position,
        width,
        height,
        pixel_format: frame.pixel_format,
        data,
        stride,
    })
}

/// Returns frame faded toward a color.
pub fn fade_frame_to_color(
    frame: &VideoFrame<'_>,
    color: [u8; 3],
    amount: f32,
) -> Result<OwnedVideoFrame> {
    if !amount.is_finite() || !(0.0..=1.0).contains(&amount) {
        return Err(DetectError::InvalidArgument(
            "fade amount must be finite and in [0, 1]".to_string(),
        ));
    }
    map_pixels(frame, |x, y| {
        let [red, green, blue] = frame.pixel_rgb(x, y);
        [
            clamp_u8(red as f32 * (1.0 - amount) + color[0] as f32 * amount),
            clamp_u8(green as f32 * (1.0 - amount) + color[1] as f32 * amount),
            clamp_u8(blue as f32 * (1.0 - amount) + color[2] as f32 * amount),
        ]
    })
}

/// Returns brightness contrast frame.
pub fn brightness_contrast_frame(
    frame: &VideoFrame<'_>,
    brightness: i16,
    contrast: f32,
) -> Result<OwnedVideoFrame> {
    if !contrast.is_finite() {
        return Err(DetectError::InvalidArgument(
            "contrast must be finite".to_string(),
        ));
    }
    map_pixels(frame, |x, y| {
        let [red, green, blue] = frame.pixel_rgb(x, y);
        [
            adjust_channel(red, brightness, contrast),
            adjust_channel(green, brightness, contrast),
            adjust_channel(blue, brightness, contrast),
        ]
    })
}

/// Returns filter 3x3 frame.
pub fn filter_3x3_frame(
    frame: &VideoFrame<'_>,
    kernel: [f32; 9],
    divisor: f32,
    bias: f32,
) -> Result<OwnedVideoFrame> {
    filter_3x3_frame_kernel(frame, &Kernel2d::from(kernel), divisor, bias)
}

/// Returns filter 3x3 frame kernel.
pub fn filter_3x3_frame_kernel(
    frame: &VideoFrame<'_>,
    kernel: &Kernel2d,
    divisor: f32,
    bias: f32,
) -> Result<OwnedVideoFrame> {
    if !divisor.is_finite() || divisor == 0.0 || !bias.is_finite() {
        return Err(DetectError::InvalidArgument(
            "filter divisor must be finite and non-zero, and bias must be finite".to_string(),
        ));
    }
    let kernel = kernel.as_array_3x3()?;
    map_pixels(frame, |x, y| {
        let mut output = [0.0f32; 3];
        for ky in 0..3 {
            for kx in 0..3 {
                let sample_x =
                    clamp_i64(x as i64 + kx as i64 - 1, 0, frame.width as i64 - 1) as u32;
                let sample_y =
                    clamp_i64(y as i64 + ky as i64 - 1, 0, frame.height as i64 - 1) as u32;
                let pixel = frame.pixel_rgb(sample_x, sample_y);
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

/// Returns sharpen frame.
pub fn sharpen_frame(frame: &VideoFrame<'_>) -> Result<OwnedVideoFrame> {
    filter_3x3_frame_kernel(frame, &Kernel2d::sharpen_3x3(), 1.0, 0.0)
}

/// Returns edge detect frame.
pub fn edge_detect_frame(frame: &VideoFrame<'_>) -> Result<OwnedVideoFrame> {
    filter_3x3_frame_kernel(frame, &Kernel2d::edge_3x3(), 1.0, 0.0)
}

fn apply_edit(frame: &VideoFrame<'_>, edit: &FrameEdit) -> Result<OwnedVideoFrame> {
    match edit {
        FrameEdit::Crop(region) => crop_frame(frame, *region),
        FrameEdit::ResizeNearest { width, height } => resize_nearest_frame(frame, *width, *height),
        FrameEdit::BoxBlur { radius } => box_blur_frame(frame, *radius),
        FrameEdit::Grayscale => grayscale_frame(frame),
        FrameEdit::Invert => invert_frame(frame),
        FrameEdit::FlipHorizontal => flip_horizontal_frame(frame),
        FrameEdit::FlipVertical => flip_vertical_frame(frame),
        FrameEdit::Rotate90 { clockwise_turns } => rotate_frame_90(frame, *clockwise_turns),
        FrameEdit::FadeToColor { color, amount } => fade_frame_to_color(frame, *color, *amount),
        FrameEdit::BrightnessContrast {
            brightness,
            contrast,
        } => brightness_contrast_frame(frame, *brightness, *contrast),
        FrameEdit::Filter3x3 {
            kernel,
            divisor,
            bias,
        } => filter_3x3_frame(frame, *kernel, *divisor, *bias),
    }
}

/// Source timeline span in seconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeSpan {
    /// Inclusive start in seconds.
    pub start_seconds: f64,
    /// Exclusive end in seconds.
    pub end_seconds: f64,
}

impl TimeSpan {
    /// Creates a validated time span.
    pub fn new(start_seconds: f64, end_seconds: f64) -> Result<Self> {
        let span = Self {
            start_seconds,
            end_seconds,
        };
        span.validate()?;
        Ok(span)
    }

    /// Returns duration seconds.
    pub fn duration_seconds(self) -> f64 {
        self.end_seconds - self.start_seconds
    }

    fn validate(self) -> Result<()> {
        if !self.start_seconds.is_finite()
            || !self.end_seconds.is_finite()
            || self.start_seconds < 0.0
            || self.end_seconds <= self.start_seconds
        {
            return Err(DetectError::InvalidArgument(
                "time span must be finite, non-negative, and have end greater than start"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

/// Timeline clip placement.
#[derive(Debug, Clone, PartialEq)]
pub struct TimelineClip {
    /// Source media identifier.
    pub source_id: String,
    /// Source start in seconds.
    pub source_start_seconds: f64,
    /// Source end in seconds.
    pub source_end_seconds: f64,
    /// Timeline start in seconds.
    pub timeline_start_seconds: f64,
}

impl TimelineClip {
    /// Creates a validated timeline clip.
    pub fn new(
        source_id: impl Into<String>,
        source_start_seconds: f64,
        source_end_seconds: f64,
        timeline_start_seconds: f64,
    ) -> Result<Self> {
        let clip = Self {
            source_id: source_id.into(),
            source_start_seconds,
            source_end_seconds,
            timeline_start_seconds,
        };
        clip.validate()?;
        Ok(clip)
    }

    /// Returns duration seconds.
    pub fn duration_seconds(&self) -> f64 {
        self.source_end_seconds - self.source_start_seconds
    }

    /// Returns timeline end seconds.
    pub fn timeline_end_seconds(&self) -> f64 {
        self.timeline_start_seconds + self.duration_seconds()
    }

    fn validate(&self) -> Result<()> {
        if self.source_id.trim().is_empty() {
            return Err(DetectError::InvalidArgument(
                "timeline clip source_id must not be empty".to_string(),
            ));
        }
        TimeSpan::new(self.source_start_seconds, self.source_end_seconds)?;
        if !self.timeline_start_seconds.is_finite() || self.timeline_start_seconds < 0.0 {
            return Err(DetectError::InvalidArgument(
                "timeline_start_seconds must be finite and non-negative".to_string(),
            ));
        }
        Ok(())
    }
}

/// Timeline transition kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionKind {
    /// Hard cut.
    Cut,
    /// Crossfade.
    CrossFade,
}

/// Timeline transition.
#[derive(Debug, Clone, PartialEq)]
pub struct TransitionSpec {
    /// Source clip index.
    pub from_clip: usize,
    /// Destination clip index.
    pub to_clip: usize,
    /// Transition duration in seconds.
    pub duration_seconds: f64,
    /// Transition kind.
    pub kind: TransitionKind,
}

impl TransitionSpec {
    /// Creates a validated transition.
    pub fn new(
        from_clip: usize,
        to_clip: usize,
        duration_seconds: f64,
        kind: TransitionKind,
    ) -> Result<Self> {
        let transition = Self {
            from_clip,
            to_clip,
            duration_seconds,
            kind,
        };
        transition.validate_for_clip_count(usize::MAX)?;
        Ok(transition)
    }

    fn validate_for_clip_count(&self, clip_count: usize) -> Result<()> {
        if self.from_clip == self.to_clip
            || self.from_clip >= clip_count
            || self.to_clip >= clip_count
        {
            return Err(DetectError::InvalidArgument(
                "transition clip indexes must refer to two different clips".to_string(),
            ));
        }
        if !self.duration_seconds.is_finite() || self.duration_seconds < 0.0 {
            return Err(DetectError::InvalidArgument(
                "transition duration must be finite and non-negative".to_string(),
            ));
        }
        if matches!(self.kind, TransitionKind::CrossFade) && self.duration_seconds == 0.0 {
            return Err(DetectError::InvalidArgument(
                "crossfade transition duration must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }
}

/// Subtitle overlay timed text.
#[derive(Debug, Clone, PartialEq)]
pub struct SubtitleOverlay {
    /// Subtitle start in seconds.
    pub start_seconds: f64,
    /// Subtitle end in seconds.
    pub end_seconds: f64,
    /// Subtitle text.
    pub text: String,
}

impl SubtitleOverlay {
    /// Creates a validated subtitle overlay.
    pub fn new(start_seconds: f64, end_seconds: f64, text: impl Into<String>) -> Result<Self> {
        let subtitle = Self {
            start_seconds,
            end_seconds,
            text: text.into(),
        };
        subtitle.validate()?;
        Ok(subtitle)
    }

    fn validate(&self) -> Result<()> {
        TimeSpan::new(self.start_seconds, self.end_seconds)?;
        if self.text.trim().is_empty() {
            return Err(DetectError::InvalidArgument(
                "subtitle text must not be empty".to_string(),
            ));
        }
        Ok(())
    }
}

/// Deterministic edit decision list.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EditDecisionList {
    /// Timeline clips.
    pub clips: Vec<TimelineClip>,
    /// Timeline transitions.
    pub transitions: Vec<TransitionSpec>,
    /// Subtitle overlays.
    pub subtitles: Vec<SubtitleOverlay>,
}

impl EditDecisionList {
    /// Returns total timeline duration.
    pub fn duration_seconds(&self) -> f64 {
        self.clips
            .iter()
            .map(TimelineClip::timeline_end_seconds)
            .fold(0.0, f64::max)
    }
}

/// Builds a deterministic cut plan from ordered source intervals.
pub fn build_cut_plan(
    source_id: impl Into<String>,
    intervals: &[TimeSpan],
) -> Result<EditDecisionList> {
    let source_id = source_id.into();
    let mut clips = Vec::with_capacity(intervals.len());
    let mut timeline_start = 0.0;
    let mut previous_source_end = 0.0;
    for interval in intervals {
        interval.validate()?;
        if interval.start_seconds < previous_source_end {
            return Err(DetectError::InvalidArgument(
                "cut intervals must be ordered by source time".to_string(),
            ));
        }
        clips.push(TimelineClip::new(
            source_id.clone(),
            interval.start_seconds,
            interval.end_seconds,
            timeline_start,
        )?);
        timeline_start += interval.duration_seconds();
        previous_source_end = interval.end_seconds;
    }
    Ok(EditDecisionList {
        clips,
        transitions: Vec::new(),
        subtitles: Vec::new(),
    })
}

/// Builds a deterministic concat plan from clips and transitions.
pub fn build_concat_plan(
    clips: Vec<TimelineClip>,
    transitions: Vec<TransitionSpec>,
) -> Result<EditDecisionList> {
    validate_clips(&clips)?;
    for transition in &transitions {
        transition.validate_for_clip_count(clips.len())?;
    }
    Ok(EditDecisionList {
        clips,
        transitions,
        subtitles: Vec::new(),
    })
}

/// Builds a deterministic subtitle plan.
pub fn build_subtitle_plan(
    clips: Vec<TimelineClip>,
    subtitles: Vec<SubtitleOverlay>,
) -> Result<EditDecisionList> {
    validate_clips(&clips)?;
    for subtitle in &subtitles {
        subtitle.validate()?;
    }
    Ok(EditDecisionList {
        clips,
        transitions: Vec::new(),
        subtitles,
    })
}

fn validate_clips(clips: &[TimelineClip]) -> Result<()> {
    let mut previous_start = None;
    for clip in clips {
        clip.validate()?;
        if let Some(previous) = previous_start {
            if clip.timeline_start_seconds < previous {
                return Err(DetectError::InvalidArgument(
                    "timeline clips must have monotonic timeline starts".to_string(),
                ));
            }
        }
        previous_start = Some(clip.timeline_start_seconds);
    }
    Ok(())
}

fn compact_frame(frame: &VideoFrame<'_>) -> OwnedVideoFrame {
    let stride = frame.width as usize * 3;
    let mut data = vec![0; stride * frame.height as usize];
    for y in 0..frame.height {
        let src_start = y as usize * frame.stride;
        let dst_start = y as usize * stride;
        data[dst_start..dst_start + stride]
            .copy_from_slice(&frame.data[src_start..src_start + stride]);
    }
    OwnedVideoFrame {
        position: frame.position,
        width: frame.width,
        height: frame.height,
        pixel_format: frame.pixel_format,
        data,
        stride,
    }
}

fn map_pixels(
    frame: &VideoFrame<'_>,
    mut f: impl FnMut(u32, u32) -> [u8; 3],
) -> Result<OwnedVideoFrame> {
    let stride = frame.width as usize * 3;
    let mut data = vec![0; stride * frame.height as usize];
    for y in 0..frame.height {
        for x in 0..frame.width {
            let rgb = f(x, y);
            write_native_pixel(&mut data, stride, frame.pixel_format, x, y, rgb);
        }
    }
    Ok(OwnedVideoFrame {
        position: frame.position,
        width: frame.width,
        height: frame.height,
        pixel_format: frame.pixel_format,
        data,
        stride,
    })
}

fn write_native_pixel(
    data: &mut [u8],
    stride: usize,
    pixel_format: PixelFormat,
    x: u32,
    y: u32,
    rgb: [u8; 3],
) {
    let i = y as usize * stride + x as usize * 3;
    match pixel_format {
        PixelFormat::Rgb24 => {
            data[i] = rgb[0];
            data[i + 1] = rgb[1];
            data[i + 2] = rgb[2];
        }
        PixelFormat::Bgr24 => {
            data[i] = rgb[2];
            data[i + 1] = rgb[1];
            data[i + 2] = rgb[0];
        }
    }
}

fn validate_region(frame: &VideoFrame<'_>, region: RectU32) -> Result<()> {
    region.validate()?;
    let x1 = region.x.checked_add(region.width).ok_or_else(|| {
        DetectError::InvalidArgument("crop region exceeds frame boundary".to_string())
    })?;
    let y1 = region.y.checked_add(region.height).ok_or_else(|| {
        DetectError::InvalidArgument("crop region exceeds frame boundary".to_string())
    })?;
    if x1 > frame.width || y1 > frame.height {
        return Err(DetectError::InvalidArgument(
            "crop region exceeds frame boundary".to_string(),
        ));
    }
    Ok(())
}

fn adjust_channel(value: u8, brightness: i16, contrast: f32) -> u8 {
    clamp_u8((value as f32 - 128.0) * contrast + 128.0 + brightness as f32)
}

fn clamp_u8(value: f32) -> u8 {
    value.round().clamp(0.0, 255.0) as u8
}

fn clamp_i64(value: i64, min: i64, max: i64) -> i64 {
    value.max(min).min(max)
}

#[cfg(test)]
mod tests {
    use num_rational::Rational64;
    use video_analysis_core::{FramePosition, OwnedVideoFrame};

    use super::*;

    fn frame() -> OwnedVideoFrame {
        OwnedVideoFrame {
            position: FramePosition::from_frame_index(0, Rational64::new(30, 1)),
            width: 3,
            height: 2,
            pixel_format: PixelFormat::Rgb24,
            data: vec![
                255, 0, 0, 0, 255, 0, 0, 0, 255, 10, 20, 30, 40, 50, 60, 70, 80, 90,
            ],
            stride: 9,
        }
    }

    #[test]
    fn crop_extracts_region() {
        let cropped =
            crop_frame_rect(&frame().as_frame(), RectU32::new(1, 0, 2, 1).unwrap()).unwrap();

        assert_eq!(cropped.width, 2);
        assert_eq!(cropped.height, 1);
        assert_eq!(cropped.data, vec![0, 255, 0, 0, 0, 255]);
    }

    #[test]
    fn grayscale_converts_pixels() {
        let gray = grayscale_frame(&frame().as_frame()).unwrap();

        assert_eq!(&gray.data[0..3], &[76, 76, 76]);
    }

    #[test]
    fn editor_chains_operations() {
        let edited = FrameEditor::new()
            .crop_rect(RectU32::new(0, 0, 2, 1).unwrap())
            .unwrap()
            .invert()
            .apply(&frame().as_frame())
            .unwrap();

        assert_eq!(edited.width, 2);
        assert_eq!(edited.data, vec![0, 255, 255, 255, 0, 255]);
    }

    #[test]
    fn blur_averages_neighbors() {
        let blurred = box_blur_frame(&frame().as_frame(), 1).unwrap();

        assert_eq!(&blurred.data[0..3], &[76, 81, 22]);
    }

    #[test]
    fn shared_kernel_helper_builds_filter_edit() {
        let edited = FrameEditor::new()
            .filter_3x3_kernel(Kernel2d::identity_3x3(), 1.0, 0.0)
            .unwrap()
            .apply(&frame().as_frame())
            .unwrap();
        assert_eq!(edited.width, frame().width);
    }

    #[test]
    fn frame_parity_operations_transform_pixels() {
        let resized = resize_nearest_frame(&frame().as_frame(), 2, 1).unwrap();
        assert_eq!((resized.width, resized.height), (2, 1));

        let flipped = flip_horizontal_frame(&frame().as_frame()).unwrap();
        assert_eq!(&flipped.data[0..3], &[0, 0, 255]);

        let vertical = flip_vertical_frame(&frame().as_frame()).unwrap();
        assert_eq!(&vertical.data[0..3], &[10, 20, 30]);

        let rotated = rotate_frame_90(&frame().as_frame(), 1).unwrap();
        assert_eq!((rotated.width, rotated.height), (2, 3));
        assert_eq!(&rotated.data[0..3], &[10, 20, 30]);

        let faded = fade_frame_to_color(&frame().as_frame(), [0, 0, 0], 0.5).unwrap();
        assert_eq!(&faded.data[0..3], &[128, 0, 0]);
    }

    #[test]
    fn edit_decision_builders_validate_timeline_data() {
        let cut = build_cut_plan(
            "source",
            &[
                TimeSpan::new(0.0, 1.0).unwrap(),
                TimeSpan::new(2.0, 4.0).unwrap(),
            ],
        )
        .unwrap();
        assert_eq!(cut.clips.len(), 2);
        assert_eq!(cut.clips[1].timeline_start_seconds, 1.0);
        assert_eq!(cut.duration_seconds(), 3.0);

        let clips = vec![
            TimelineClip::new("a", 0.0, 1.0, 0.0).unwrap(),
            TimelineClip::new("b", 0.0, 1.0, 1.0).unwrap(),
        ];
        let concat = build_concat_plan(
            clips.clone(),
            vec![TransitionSpec::new(0, 1, 0.25, TransitionKind::CrossFade).unwrap()],
        )
        .unwrap();
        assert_eq!(concat.transitions.len(), 1);

        let subtitles = build_subtitle_plan(
            clips,
            vec![SubtitleOverlay::new(0.0, 1.0, "hello").unwrap()],
        )
        .unwrap();
        assert_eq!(subtitles.subtitles.len(), 1);

        assert!(build_cut_plan(
            "source",
            &[
                TimeSpan::new(2.0, 3.0).unwrap(),
                TimeSpan::new(1.0, 2.0).unwrap(),
            ],
        )
        .is_err());
    }
}
