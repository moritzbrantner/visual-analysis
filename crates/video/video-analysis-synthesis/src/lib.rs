#![doc = include_str!("../README.md")]

pub mod surface;
use std::collections::BTreeMap;

use data_inversion_core::{Generated, InformationFidelity, InversionMethod, InversionTrace};
use num_rational::Rational64;
use video_analysis_core::{
    BoundingBox, DetectError, FramePosition, Observation, OwnedVideoFrame, PixelFormat, Result,
};

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
    /// Creates a new value.
    pub const fn new(red: u8, green: u8, blue: u8) -> Self {
        Self { red, green, blue }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Data type for synthesized region.
pub struct SynthesizedRegion {
    /// The region value.
    pub region: BoundingBox,
    /// The color value.
    pub color: RgbColor,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for frame synthesis spec.
pub struct FrameSynthesisSpec {
    /// The frame index value.
    pub frame_index: u64,
    /// The background value.
    pub background: RgbColor,
    /// The regions value.
    pub regions: Vec<SynthesizedRegion>,
}

impl FrameSynthesisSpec {
    /// Creates a new value.
    pub fn new(frame_index: u64, background: RgbColor) -> Self {
        Self {
            frame_index,
            background,
            regions: Vec::new(),
        }
    }

    /// Returns region.
    pub fn region(mut self, region: SynthesizedRegion) -> Self {
        self.regions.push(region);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Data type for video synthesis config.
pub struct VideoSynthesisConfig {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// The FPS value.
    pub fps: Rational64,
    /// The background value.
    pub background: RgbColor,
}

impl VideoSynthesisConfig {
    /// Creates a new value.
    pub fn new(width: u32, height: u32, fps: Rational64) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(DetectError::InvalidDimensions { width, height });
        }
        if *fps.numer() <= 0 || *fps.denom() <= 0 {
            return Err(invalid_argument("fps must be positive"));
        }
        Ok(Self {
            width,
            height,
            fps,
            background: RgbColor::new(16, 18, 20),
        })
    }

    /// Returns background.
    pub fn background(mut self, background: RgbColor) -> Self {
        self.background = background;
        self
    }
}

impl Default for VideoSynthesisConfig {
    fn default() -> Self {
        Self::new(640, 360, Rational64::new(30, 1))
            .expect("default video synthesis config is valid")
    }
}

/// Returns frame from spec.
pub fn frame_from_spec(
    spec: &FrameSynthesisSpec,
    config: VideoSynthesisConfig,
) -> Result<Generated<OwnedVideoFrame>> {
    let mut data = vec![0_u8; config.width as usize * config.height as usize * 3];
    for y in 0..config.height {
        for x in 0..config.width {
            set_pixel(&mut data, config.width, x, y, spec.background);
        }
    }
    for region in &spec.regions {
        draw_region_outline(&mut data, config.width, config.height, region);
    }
    let frame = OwnedVideoFrame {
        position: FramePosition::from_frame_index(spec.frame_index, config.fps),
        width: config.width,
        height: config.height,
        pixel_format: PixelFormat::Rgb24,
        data,
        stride: config.width as usize * 3,
    };
    let trace = InversionTrace::new(
        "frame_synthesis_spec",
        "owned_video_frame",
        InformationFidelity::Heuristic,
    )
    .note(
        "pixels",
        InversionMethod::Template,
        "frame pixels are generated from background and region outlines",
    );
    Ok(Generated::new(frame, trace))
}

/// Returns storyboard from observations.
pub fn storyboard_from_observations(
    observations: &[Observation],
    config: VideoSynthesisConfig,
) -> Result<Generated<Vec<OwnedVideoFrame>>> {
    if observations.is_empty() {
        return Err(invalid_argument("observations must not be empty"));
    }
    let mut by_frame = BTreeMap::<u64, Vec<&Observation>>::new();
    for observation in observations {
        let frame_index = observation
            .frame
            .map(|position| position.frame_index)
            .or_else(|| {
                observation
                    .timestamp
                    .map(|timestamp| seconds_to_frame(timestamp.seconds(), config.fps))
            })
            .unwrap_or(0);
        by_frame.entry(frame_index).or_default().push(observation);
    }

    let mut frames = Vec::with_capacity(by_frame.len());
    let mut trace = InversionTrace::new(
        "observations",
        "owned_video_frames",
        InformationFidelity::Heuristic,
    )
    .assumption("labels are encoded as deterministic colors")
    .note(
        "regions",
        InversionMethod::Preserved,
        "observation regions are drawn as outlines when present",
    )
    .note(
        "missing_regions",
        InversionMethod::Inferred,
        "observations without regions are drawn as timeline markers",
    );

    for (frame_index, frame_observations) in by_frame {
        let mut spec = FrameSynthesisSpec::new(frame_index, config.background);
        for (index, observation) in frame_observations.iter().enumerate() {
            let color = color_for_observation(observation);
            let region = observation.region.unwrap_or_else(|| {
                marker_region(index, frame_observations.len(), config.width, config.height)
            });
            spec.regions.push(SynthesizedRegion { region, color });
        }
        let generated = frame_from_spec(&spec, config)?;
        trace.merge_notes(&generated.trace);
        frames.push(generated.value);
    }

    Ok(Generated::new(frames, trace))
}

fn marker_region(index: usize, count: usize, width: u32, height: u32) -> BoundingBox {
    let count = count.max(1) as u32;
    let marker_width = (width / count).max(1);
    let x = (index as u32).saturating_mul(marker_width).min(width - 1);
    let y = height.saturating_sub(6);
    BoundingBox {
        x,
        y,
        width: marker_width.min(width - x),
        height: 4.min(height - y),
    }
}

fn draw_region_outline(data: &mut [u8], width: u32, height: u32, synthesized: &SynthesizedRegion) {
    let x0 = synthesized.region.x.min(width - 1);
    let y0 = synthesized.region.y.min(height - 1);
    let x1 = synthesized
        .region
        .x
        .saturating_add(synthesized.region.width)
        .saturating_sub(1)
        .min(width - 1);
    let y1 = synthesized
        .region
        .y
        .saturating_add(synthesized.region.height)
        .saturating_sub(1)
        .min(height - 1);

    for x in x0..=x1 {
        set_pixel(data, width, x, y0, synthesized.color);
        set_pixel(data, width, x, y1, synthesized.color);
    }
    for y in y0..=y1 {
        set_pixel(data, width, x0, y, synthesized.color);
        set_pixel(data, width, x1, y, synthesized.color);
    }
}

fn set_pixel(data: &mut [u8], width: u32, x: u32, y: u32, color: RgbColor) {
    let offset = (y as usize * width as usize + x as usize) * 3;
    data[offset] = color.red;
    data[offset + 1] = color.green;
    data[offset + 2] = color.blue;
}

fn color_for_observation(observation: &Observation) -> RgbColor {
    let key = observation
        .label
        .as_deref()
        .or(observation.text.as_deref())
        .or(observation.track_id.as_deref())
        .unwrap_or(&observation.analyzer);
    let hash = fnv1a(key.as_bytes());
    RgbColor::new(
        80 + (hash & 0x7f) as u8,
        80 + ((hash >> 8) & 0x7f) as u8,
        80 + ((hash >> 16) & 0x7f) as u8,
    )
}

fn fnv1a(bytes: &[u8]) -> u32 {
    let mut hash = 0x811c9dc5_u32;
    for byte in bytes {
        hash ^= *byte as u32;
        hash = hash.wrapping_mul(0x01000193);
    }
    hash
}

fn seconds_to_frame(seconds: f64, fps: Rational64) -> u64 {
    let fps = *fps.numer() as f64 / *fps.denom() as f64;
    (seconds.max(0.0) * fps).round() as u64
}

fn invalid_argument(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use video_analysis_core::{ObservationKind, Timebase, Timestamp};

    #[test]
    fn creates_frame_from_regions() {
        let spec = FrameSynthesisSpec::new(2, RgbColor::new(0, 0, 0)).region(SynthesizedRegion {
            region: BoundingBox::new(1, 1, 2, 2).unwrap(),
            color: RgbColor::new(255, 0, 0),
        });
        let generated = frame_from_spec(
            &spec,
            VideoSynthesisConfig::new(4, 4, Rational64::new(24, 1)).unwrap(),
        )
        .unwrap();
        assert_eq!(generated.value.position.frame_index, 2);
        assert!(generated.value.data.contains(&255));
    }

    #[test]
    fn creates_storyboard_from_observations() {
        let observation = Observation::new("detector", ObservationKind::Object)
            .label("person")
            .at_timestamp(Timestamp::new(1, Timebase::new(1, 1)));
        let generated = storyboard_from_observations(
            &[observation],
            VideoSynthesisConfig::new(8, 8, Rational64::new(2, 1)).unwrap(),
        )
        .unwrap();
        assert_eq!(generated.value.len(), 1);
        assert_eq!(generated.value[0].position.frame_index, 2);
    }
}
