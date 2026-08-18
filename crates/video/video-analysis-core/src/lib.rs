#![doc = include_str!("../README.md")]

pub mod runtime;
pub mod surface;
pub use audio_contracts::{
    AudioAnalysis, AudioAnalysisResult, AudioAnalyzer, AudioBuffer, AudioFrame, AudioPipeline,
    AudioPipelineBuilder, OwnedAudioFrame,
};
pub use media_core::{
    AnalysisEvent, AudioSampleFormat, DetectError, PixelFormat, Result, Timebase, Timestamp,
};
pub use text_core::{
    OwnedTextSegment, TextAnalysis, TextAnalysisResult, TextAnalyzer, TextPipeline,
    TextPipelineBuilder, TextSegment,
};

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::ops::{Add, Sub};

use num_rational::Rational64;
use num_traits::ToPrimitive;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// Data type for frame position.
pub struct FramePosition {
    /// The frame index value.
    pub frame_index: u64,
    /// Timestamp associated with this value.
    pub timestamp: Timestamp,
}

impl FramePosition {
    /// Builds this value from frame index.
    pub fn from_frame_index(frame_index: u64, fps: Rational64) -> Self {
        let den = *fps.numer() as i32;
        let num = *fps.denom() as i32;
        Self {
            frame_index,
            timestamp: Timestamp::new(frame_index as i64, Timebase::new(num, den)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// Data type for frame timecode.
pub struct FrameTimecode {
    /// The frame index value.
    pub frame_index: u64,
    /// The FPS value.
    pub fps: Rational64,
}

impl FrameTimecode {
    /// Builds this value from frames.
    pub fn from_frames(frame_index: u64, fps: Rational64) -> Self {
        Self { frame_index, fps }
    }

    /// Builds a PTS-backed frame timecode without changing the legacy
    /// `FrameTimecode` layout.
    pub fn from_pts(
        frame_index: u64,
        fps: Rational64,
        pts: i64,
        timebase: Timebase,
    ) -> TimestampedFrameTimecode {
        TimestampedFrameTimecode {
            timecode: Self::from_frames(frame_index, fps),
            timestamp: Timestamp::new(pts, timebase),
        }
    }

    /// Builds this value from seconds.
    pub fn from_seconds(seconds: f64, fps: Rational64) -> Result<Self> {
        if seconds < 0.0 {
            return Err(DetectError::InvalidArgument(
                "seconds must be greater than or equal to zero".to_string(),
            ));
        }
        let fps_float = fps.to_f64().ok_or_else(|| {
            DetectError::InvalidArgument("frame rate cannot be represented".to_string())
        })?;
        Ok(Self {
            frame_index: (seconds * fps_float).round() as u64,
            fps,
        })
    }

    /// Parses parse.
    pub fn parse(input: &str, fps: Rational64) -> Result<Self> {
        if input.chars().all(|c| c.is_ascii_digit()) {
            let frame_index = input.parse::<u64>().map_err(|err| {
                DetectError::InvalidArgument(format!("invalid frame number `{input}`: {err}"))
            })?;
            return Ok(Self { frame_index, fps });
        }
        if !input.contains(':') {
            let seconds = input.parse::<f64>().map_err(|err| {
                DetectError::InvalidArgument(format!("invalid seconds `{input}`: {err}"))
            })?;
            return Self::from_seconds(seconds, fps);
        }

        let parts: Vec<&str> = input.split(':').collect();
        if parts.len() != 3 {
            return Err(DetectError::InvalidArgument(format!(
                "invalid timecode `{input}`"
            )));
        }
        let hours = parts[0].parse::<f64>().map_err(|err| {
            DetectError::InvalidArgument(format!("invalid hours in `{input}`: {err}"))
        })?;
        let minutes = parts[1].parse::<f64>().map_err(|err| {
            DetectError::InvalidArgument(format!("invalid minutes in `{input}`: {err}"))
        })?;
        let seconds = parts[2].parse::<f64>().map_err(|err| {
            DetectError::InvalidArgument(format!("invalid seconds in `{input}`: {err}"))
        })?;
        Self::from_seconds((hours * 3600.0) + (minutes * 60.0) + seconds, fps)
    }

    /// Returns seconds.
    pub fn seconds(self) -> f64 {
        self.frame_index as f64 / self.fps.to_f64().unwrap_or(1.0)
    }

    /// Returns timecode.
    pub fn timecode(self, precision: usize) -> String {
        let factor = 10_f64.powi(precision as i32);
        let total = (self.seconds() * factor).round() / factor;
        let hours = (total / 3600.0).floor() as u64;
        let minutes = ((total - hours as f64 * 3600.0) / 60.0).floor() as u64;
        let seconds = total - hours as f64 * 3600.0 - minutes as f64 * 60.0;
        if precision == 0 {
            format!("{hours:02}:{minutes:02}:{:02}", seconds.round() as u64)
        } else {
            let whole = seconds.floor() as u64;
            let frac = ((seconds - whole as f64) * factor).round() as u64;
            format!("{hours:02}:{minutes:02}:{whole:02}.{frac:0precision$}")
        }
    }

    /// Returns position.
    pub fn position(self) -> FramePosition {
        FramePosition::from_frame_index(self.frame_index, self.fps)
    }

    /// Returns PTS metadata when this value came from a timestamped wrapper.
    ///
    /// Plain `FrameTimecode` values remain frame-rate-derived and therefore do
    /// not carry source PTS metadata.
    pub const fn pts(self) -> Option<i64> {
        None
    }

    /// Returns source timebase metadata when this value came from a timestamped
    /// wrapper.
    ///
    /// Plain `FrameTimecode` values remain frame-rate-derived and therefore do
    /// not carry source timebase metadata.
    pub const fn time_base(self) -> Option<Timebase> {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// Frame timecode paired with source PTS/timebase metadata for VFR-aware
/// workflows.
pub struct TimestampedFrameTimecode {
    /// The legacy frame/fps timecode view.
    pub timecode: FrameTimecode,
    /// The source timestamp attached to this frame.
    pub timestamp: Timestamp,
}

impl TimestampedFrameTimecode {
    /// Returns the frame index.
    pub const fn frame_index(self) -> u64 {
        self.timecode.frame_index
    }

    /// Returns the display frame rate.
    pub const fn fps(self) -> Rational64 {
        self.timecode.fps
    }

    /// Returns source PTS metadata.
    pub const fn pts(self) -> Option<i64> {
        Some(self.timestamp.pts)
    }

    /// Returns source timebase metadata.
    pub const fn time_base(self) -> Option<Timebase> {
        Some(self.timestamp.timebase)
    }

    /// Returns source timestamp seconds.
    pub fn seconds(self) -> f64 {
        self.timestamp.seconds()
    }

    /// Returns display timecode using the legacy frame/fps conversion.
    pub fn timecode(self, precision: usize) -> String {
        self.timecode.timecode(precision)
    }

    /// Returns a frame position preserving the source timestamp.
    pub const fn position(self) -> FramePosition {
        FramePosition {
            frame_index: self.timecode.frame_index,
            timestamp: self.timestamp,
        }
    }
}

impl Add<u64> for FrameTimecode {
    type Output = FrameTimecode;

    fn add(self, rhs: u64) -> Self::Output {
        Self {
            frame_index: self.frame_index + rhs,
            fps: self.fps,
        }
    }
}

impl Sub<u64> for FrameTimecode {
    type Output = FrameTimecode;

    fn sub(self, rhs: u64) -> Self::Output {
        Self {
            frame_index: self.frame_index.saturating_sub(rhs),
            fps: self.fps,
        }
    }
}

impl fmt::Display for FrameTimecode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.timecode(3))
    }
}

#[derive(Debug, Clone, Copy)]
/// Data type for video frame.
pub struct VideoFrame<'a> {
    /// The position value.
    pub position: FramePosition,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// The pixel format value.
    pub pixel_format: PixelFormat,
    /// Underlying data buffer.
    pub data: &'a [u8],
    /// The stride value.
    pub stride: usize,
}

impl<'a> VideoFrame<'a> {
    /// Returns rgb24.
    pub fn rgb24(position: FramePosition, width: u32, height: u32, data: &'a [u8]) -> Result<Self> {
        Self::packed(
            position,
            width,
            height,
            PixelFormat::Rgb24,
            data,
            width as usize * 3,
        )
    }

    /// Returns bgr24.
    pub fn bgr24(position: FramePosition, width: u32, height: u32, data: &'a [u8]) -> Result<Self> {
        Self::packed(
            position,
            width,
            height,
            PixelFormat::Bgr24,
            data,
            width as usize * 3,
        )
    }

    /// Returns packed.
    pub fn packed(
        position: FramePosition,
        width: u32,
        height: u32,
        pixel_format: PixelFormat,
        data: &'a [u8],
        stride: usize,
    ) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(DetectError::InvalidDimensions { width, height });
        }
        let expected = stride * height as usize;
        if data.len() < expected {
            return Err(DetectError::InvalidFrameBuffer {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            position,
            width,
            height,
            pixel_format,
            data,
            stride,
        })
    }

    /// Returns pixel RGB.
    pub fn pixel_rgb(&self, x: u32, y: u32) -> [u8; 3] {
        let i = y as usize * self.stride + x as usize * 3;
        match self.pixel_format {
            PixelFormat::Rgb24 => [self.data[i], self.data[i + 1], self.data[i + 2]],
            PixelFormat::Bgr24 => [self.data[i + 2], self.data[i + 1], self.data[i]],
        }
    }

    /// Returns pixel count.
    pub fn pixel_count(&self) -> usize {
        self.width as usize * self.height as usize
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for scene.
pub struct Scene {
    /// The start value.
    pub start: FramePosition,
    /// The end value.
    pub end: FramePosition,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for cut.
pub struct Cut {
    /// The position value.
    pub position: FramePosition,
    /// The detector value.
    pub detector: &'static str,
    /// Score assigned to this value.
    pub score: Option<f32>,
}

/// Trait for metrics sink implementations.
pub trait MetricsSink {
    /// Sets metric.
    fn set_metric(&mut self, frame_index: u64, key: &str, value: f64);
}

#[derive(Debug, Default, Clone, PartialEq)]
/// Data type for metrics store.
pub struct MetricsStore {
    rows: BTreeMap<u64, BTreeMap<String, f64>>,
    keys: BTreeSet<String>,
}

impl MetricsStore {
    /// Returns rows.
    pub fn rows(&self) -> &BTreeMap<u64, BTreeMap<String, f64>> {
        &self.rows
    }

    /// Returns keys.
    pub fn keys(&self) -> impl Iterator<Item = &str> + '_ {
        self.keys.iter().map(String::as_str)
    }

    /// Returns get.
    pub fn get(&self, frame_index: u64, key: &str) -> Option<f64> {
        self.rows
            .get(&frame_index)
            .and_then(|row| row.get(key))
            .copied()
    }
}

impl MetricsSink for MetricsStore {
    fn set_metric(&mut self, frame_index: u64, key: &str, value: f64) {
        self.keys.insert(key.to_string());
        self.rows
            .entry(frame_index)
            .or_default()
            .insert(key.to_string(), value);
    }
}

/// Trait for scene detector implementations.
pub trait SceneDetector {
    /// Returns name.
    fn name(&self) -> &'static str;
    /// Returns metric keys.
    fn metric_keys(&self) -> &'static [&'static str];
    /// Returns event buffer len.
    fn event_buffer_len(&self) -> usize {
        0
    }
    /// Returns process frame.
    fn process_frame(
        &mut self,
        frame: &VideoFrame<'_>,
        metrics: Option<&mut dyn MetricsSink>,
    ) -> Result<Vec<Cut>>;
    /// Returns finish.
    fn finish(
        &mut self,
        _last_position: FramePosition,
        _metrics: Option<&mut dyn MetricsSink>,
    ) -> Result<Vec<Cut>> {
        Ok(Vec::new())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Variants describing flash filter mode.
pub enum FlashFilterMode {
    /// The merge variant.
    Merge,
    /// The suppress variant.
    Suppress,
}


#[derive(Debug, Clone, Copy)]
/// Compatibility weights converted explicitly into the canonical scene contract.
pub struct ContentWeights {
    /// Hue contribution.
    pub delta_hue: f32,
    /// Saturation contribution.
    pub delta_sat: f32,
    /// Luminance contribution.
    pub delta_lum: f32,
    /// Edge contribution.
    pub delta_edges: f32,
}

impl Default for ContentWeights {
    fn default() -> Self {
        Self {
            delta_hue: 1.0,
            delta_sat: 1.0,
            delta_lum: 1.0,
            delta_edges: 0.0,
        }
    }
}

impl ContentWeights {
    /// Compatibility preset for luminance-only scoring.
    pub const LUMA_ONLY: Self = Self {
        delta_hue: 0.0,
        delta_sat: 0.0,
        delta_lum: 1.0,
        delta_edges: 0.0,
    };

    fn canonical(self) -> scenedetect_core::ContentWeights {
        scenedetect_core::ContentWeights {
            hue: self.delta_hue as f64,
            saturation: self.delta_sat as f64,
            luminance: self.delta_lum as f64,
            edges: self.delta_edges as f64,
        }
    }
}

#[derive(Debug, Clone)]
/// Compatibility adapter backed by the canonical `scenedetect-core` detector.
pub struct ContentDetector {
    threshold: f32,
    weights: ContentWeights,
    luma_only: bool,
    min_scene_len: u64,
    filter_mode: FlashFilterMode,
    frames: Vec<scenedetect_core::Frame>,
    emitted: BTreeSet<u64>,
}

impl Default for ContentDetector {
    fn default() -> Self {
        Self::new(27.0, 15)
    }
}

impl ContentDetector {
    /// Metric keys retained by the visual compatibility surface.
    pub const METRIC_KEYS: &'static [&'static str] = &[
        "content_val",
        "delta_hue",
        "delta_sat",
        "delta_lum",
        "delta_edges",
    ];

    /// Creates a canonical-detector compatibility adapter.
    pub fn new(threshold: f32, min_scene_len: u64) -> Self {
        Self {
            threshold,
            weights: ContentWeights::default(),
            luma_only: false,
            min_scene_len,
            filter_mode: FlashFilterMode::Merge,
            frames: Vec::new(),
            emitted: BTreeSet::new(),
        }
    }

    /// Sets compatibility weights before conversion to canonical weights.
    pub fn with_weights(mut self, weights: ContentWeights) -> Self {
        self.weights = weights;
        self
    }

    /// Selects canonical luminance-only scoring.
    pub fn luma_only(mut self, value: bool) -> Self {
        self.luma_only = value;
        self
    }

    /// Retains the legacy validation seam; kernel policy belongs to the canonical owner.
    pub fn kernel_size(self, value: Option<usize>) -> Result<Self> {
        if let Some(size) = value {
            if size < 3 || size % 2 == 0 {
                return Err(DetectError::InvalidArgument(
                    "kernel_size must be an odd integer >= 3".to_string(),
                ));
            }
        }
        Ok(self)
    }

    /// Maps the visual flash-filter choice to canonical minimum-length policy.
    pub fn filter_mode(mut self, mode: FlashFilterMode, min_scene_len: u64) -> Self {
        self.filter_mode = mode;
        self.min_scene_len = min_scene_len;
        self
    }

    fn canonical_config(&self) -> scenedetect_core::DetectorConfig {
        scenedetect_core::DetectorConfig::Content(scenedetect_core::ContentDetectorConfig {
            threshold: self.threshold as f64,
            weights: self.weights.canonical(),
            luma_only: self.luma_only,
        })
    }

    fn canonical_options(&self) -> scenedetect_core::DetectionOptions {
        scenedetect_core::DetectionOptions {
            min_scene_len: self.min_scene_len,
            min_scene_len_policy: match self.filter_mode {
                FlashFilterMode::Merge => scenedetect_core::MinSceneLenPolicy::MergeLast,
                FlashFilterMode::Suppress => scenedetect_core::MinSceneLenPolicy::Suppress,
            },
        }
    }
}

impl SceneDetector for ContentDetector {
    fn name(&self) -> &'static str {
        "content"
    }

    fn metric_keys(&self) -> &'static [&'static str] {
        Self::METRIC_KEYS
    }

    fn process_frame(
        &mut self,
        frame: &VideoFrame<'_>,
        mut metrics: Option<&mut dyn MetricsSink>,
    ) -> Result<Vec<Cut>> {
        let rgb = (0..frame.height)
            .flat_map(|y| (0..frame.width).flat_map(move |x| frame.pixel_rgb(x, y)))
            .collect();
        self.frames.push(scenedetect_core::Frame {
            index: scenedetect_core::FrameIndex(frame.position.frame_index),
            width: frame.width,
            height: frame.height,
            rgb,
        });
        let timebase = frame.position.timestamp.timebase;
        let fps = if timebase.num == 0 {
            1.0
        } else {
            timebase.den as f64 / timebase.num as f64
        };
        let detected = scenedetect_core::detect_frames(
            self.canonical_config(),
            scenedetect_core::FrameRate(fps),
            &self.frames,
            self.canonical_options(),
        )
        .map_err(|error| DetectError::Source(error.to_string()))?;
        if let Some(sink) = metrics.as_mut() {
            if let Some(row) = detected.stats.rows.last() {
                for (key, value) in &row.metrics {
                    sink.set_metric(row.frame.0, key, *value);
                }
            }
        }
        Ok(detected
            .scene_list
            .scenes
            .into_iter()
            .map(|scene| scene.start.0)
            .filter(|start| *start != 0 && self.emitted.insert(*start))
            .map(|frame_index| Cut {
                position: position_like(frame.position, frame_index),
                detector: self.name(),
                score: None,
            })
            .collect())
    }
}

/// Trait for video source implementations.
pub trait VideoSource {
    /// Returns next frame.
    fn next_frame(&mut self) -> Result<Option<OwnedVideoFrame>>;
    /// Returns frame rate.
    fn frame_rate(&self) -> Rational64;
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Data type for owned video frame.
pub struct OwnedVideoFrame {
    /// The position value.
    pub position: FramePosition,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// The pixel format value.
    pub pixel_format: PixelFormat,
    /// Underlying data buffer.
    pub data: Vec<u8>,
    /// The stride value.
    pub stride: usize,
}

impl OwnedVideoFrame {
    /// Borrows this value as a frame.
    pub fn as_frame(&self) -> VideoFrame<'_> {
        VideoFrame {
            position: self.position,
            width: self.width,
            height: self.height,
            pixel_format: self.pixel_format,
            data: &self.data,
            stride: self.stride,
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
/// Data type for detection result.
pub struct DetectionResult {
    /// The scenes value.
    pub scenes: Vec<Scene>,
    /// The cuts value.
    pub cuts: Vec<Cut>,
    /// The metrics value.
    pub metrics: MetricsStore,
    /// The frames processed value.
    pub frames_processed: u64,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for frame analysis.
pub struct FrameAnalysis {
    /// The position value.
    pub position: FramePosition,
    /// The cuts value.
    pub cuts: Vec<Cut>,
    /// The frames processed value.
    pub frames_processed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
/// Data type for bounding box.
pub struct BoundingBox {
    /// The x value.
    pub x: u32,
    /// The y value.
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl BoundingBox {
    /// Creates a new value.
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(DetectError::InvalidArgument(
                "bounding box must have non-zero width and height".to_string(),
            ));
        }
        Ok(Self {
            x,
            y,
            width,
            height,
        })
    }
}

impl From<BoundingBox> for math_geometry_2d::RectU32 {
    fn from(value: BoundingBox) -> Self {
        Self {
            x: value.x,
            y: value.y,
            width: value.width,
            height: value.height,
        }
    }
}

impl TryFrom<math_geometry_2d::RectU32> for BoundingBox {
    type Error = DetectError;

    fn try_from(value: math_geometry_2d::RectU32) -> std::result::Result<Self, Self::Error> {
        value.validate()?;
        Self::new(value.x, value.y, value.width, value.height)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Variants describing observation kind.
pub enum ObservationKind {
    /// The text variant.
    Text,
    /// The face variant.
    Face,
    /// The object variant.
    Object,
    /// The scene variant.
    Scene,
    /// The custom variant.
    Custom(String),
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for observation.
pub struct Observation {
    /// Timestamp associated with this value.
    pub timestamp: Option<Timestamp>,
    /// The frame value.
    pub frame: Option<FramePosition>,
    /// The scene index value.
    pub scene_index: Option<u64>,
    /// The analyzer value.
    pub analyzer: String,
    /// The kind value.
    pub kind: ObservationKind,
    /// Label assigned to this value.
    pub label: Option<String>,
    /// Text content for this value.
    pub text: Option<String>,
    /// Score assigned to this value.
    pub score: Option<f32>,
    /// The region value.
    pub region: Option<BoundingBox>,
    /// The track identifier value.
    pub track_id: Option<String>,
    /// The attributes value.
    pub attributes: BTreeMap<String, String>,
}

impl Observation {
    /// Creates a new value.
    pub fn new(analyzer: impl Into<String>, kind: ObservationKind) -> Self {
        Self {
            timestamp: None,
            frame: None,
            scene_index: None,
            analyzer: analyzer.into(),
            kind,
            label: None,
            text: None,
            score: None,
            region: None,
            track_id: None,
            attributes: BTreeMap::new(),
        }
    }

    /// Returns at frame.
    pub fn at_frame(mut self, position: FramePosition) -> Self {
        self.timestamp = Some(position.timestamp);
        self.frame = Some(position);
        self
    }

    /// Returns at timestamp.
    pub fn at_timestamp(mut self, timestamp: Timestamp) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    /// Returns in scene.
    pub fn in_scene(mut self, scene_index: u64) -> Self {
        self.scene_index = Some(scene_index);
        self
    }

    /// Returns label.
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Returns text.
    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    /// Returns score.
    pub fn score(mut self, score: f32) -> Self {
        self.score = Some(score);
        self
    }

    /// Returns region.
    pub fn region(mut self, region: BoundingBox) -> Self {
        self.region = Some(region);
        self
    }

    /// Returns track identifier.
    pub fn track_id(mut self, track_id: impl Into<String>) -> Self {
        self.track_id = Some(track_id.into());
        self
    }

    /// Returns attribute.
    pub fn attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// Converts this value to text segment.
    pub fn to_text_segment(&self, segment_index: u64) -> Option<OwnedTextSegment> {
        let text = self.text.as_ref()?;
        let mut segment = OwnedTextSegment::new(segment_index, text.clone());
        if let Some(timestamp) = self.timestamp {
            segment = segment.timestamp(timestamp);
        }
        if let Some(language) = self.attributes.get("language") {
            segment = segment.language(language.clone());
        }
        Some(segment)
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
/// Data type for video analysis result.
pub struct VideoAnalysisResult {
    /// The observations value.
    pub observations: Vec<Observation>,
    /// The frames processed value.
    pub frames_processed: u64,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for video frame analysis.
pub struct VideoFrameAnalysis {
    /// The position value.
    pub position: FramePosition,
    /// The observations value.
    pub observations: Vec<Observation>,
    /// The frames processed value.
    pub frames_processed: u64,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for scene analysis.
pub struct SceneAnalysis {
    /// The scene index value.
    pub scene_index: u64,
    /// The scene value.
    pub scene: Scene,
    /// The observations value.
    pub observations: Vec<Observation>,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for realtime video frame analysis.
pub struct RealtimeVideoFrameAnalysis {
    /// The position value.
    pub position: FramePosition,
    /// The scene value.
    pub scene: FrameAnalysis,
    /// The observations value.
    pub observations: Vec<Observation>,
    /// The completed scenes value.
    pub completed_scenes: Vec<SceneAnalysis>,
    /// The frames processed value.
    pub frames_processed: u64,
}

#[derive(Debug, Default, Clone, PartialEq)]
/// Data type for realtime video analysis result.
pub struct RealtimeVideoAnalysisResult {
    /// The detection value.
    pub detection: DetectionResult,
    /// The observations value.
    pub observations: Vec<Observation>,
    /// The scenes value.
    pub scenes: Vec<SceneAnalysis>,
    /// The frames processed value.
    pub frames_processed: u64,
}

/// Data type for scene pipeline.
pub struct ScenePipeline {
    detectors: Vec<Box<dyn SceneDetector>>,
    start_in_scene: bool,
    crop: Option<CropRegion>,
    auto_downscale_min_width: Option<u32>,
    state: ScenePipelineState,
}

#[derive(Debug, Default, Clone)]
struct ScenePipelineState {
    cuts: Vec<Cut>,
    metrics: MetricsStore,
    first_position: Option<FramePosition>,
    last_position: Option<FramePosition>,
    frames_processed: u64,
    finished: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Data type for crop region.
pub struct CropRegion {
    /// The x0 value.
    pub x0: u32,
    /// The y0 value.
    pub y0: u32,
    /// The x1 value.
    pub x1: u32,
    /// The y1 value.
    pub y1: u32,
}

impl CropRegion {
    /// Creates a new value.
    pub fn new(x0: u32, y0: u32, x1: u32, y1: u32) -> Result<Self> {
        if x0 == x1 || y0 == y1 {
            return Err(DetectError::InvalidArgument(
                "crop region must have non-zero width and height".to_string(),
            ));
        }
        Ok(Self { x0, y0, x1, y1 })
    }
}

impl ScenePipeline {
    /// Returns builder.
    pub fn builder() -> ScenePipelineBuilder {
        ScenePipelineBuilder::default()
    }

    /// Returns detect.
    pub fn detect<S: VideoSource>(&mut self, source: &mut S) -> Result<DetectionResult> {
        self.reset();
        while let Some(frame) = source.next_frame()? {
            self.process_frame(frame)?;
        }
        self.finish_detection()
    }

    /// Returns process frame.
    pub fn process_frame(&mut self, frame: OwnedVideoFrame) -> Result<FrameAnalysis> {
        let frame = self.prepare_frame(frame)?;
        self.process_frame_ref(&frame.as_frame())
    }

    /// Returns process frame ref.
    pub fn process_frame_ref(&mut self, frame: &VideoFrame<'_>) -> Result<FrameAnalysis> {
        if self.state.finished {
            return Err(DetectError::InvalidArgument(
                "cannot process frames after finish_detection; call reset first".to_string(),
            ));
        }
        self.validate_frame_options(frame)?;
        self.state.first_position.get_or_insert(frame.position);
        self.state.last_position = Some(frame.position);

        let mut cuts = Vec::new();
        for detector in &mut self.detectors {
            let mut new_cuts = detector.process_frame(frame, Some(&mut self.state.metrics))?;
            cuts.append(&mut new_cuts);
        }
        let cuts = self.record_cuts(cuts);
        self.state.frames_processed += 1;

        Ok(FrameAnalysis {
            position: frame.position,
            cuts,
            frames_processed: self.state.frames_processed,
        })
    }

    /// Returns finish detection.
    pub fn finish_detection(&mut self) -> Result<DetectionResult> {
        if !self.state.finished {
            if let Some(last) = self.state.last_position {
                let mut cuts = Vec::new();
                for detector in &mut self.detectors {
                    let mut new_cuts = detector.finish(last, Some(&mut self.state.metrics))?;
                    cuts.append(&mut new_cuts);
                }
                self.record_cuts(cuts);
            }
            self.state.finished = true;
        }

        let scenes = if let (Some(start), Some(end)) =
            (self.state.first_position, self.state.last_position)
        {
            scenes_from_cuts(&self.state.cuts, start, end, self.start_in_scene)
        } else {
            Vec::new()
        };

        Ok(DetectionResult {
            scenes,
            cuts: self.state.cuts.clone(),
            metrics: self.state.metrics.clone(),
            frames_processed: self.state.frames_processed,
        })
    }

    /// Returns reset.
    pub fn reset(&mut self) {
        self.state = ScenePipelineState::default();
    }

    /// Returns metrics.
    pub fn metrics(&self) -> &MetricsStore {
        &self.state.metrics
    }

    /// Returns cuts.
    pub fn cuts(&self) -> &[Cut] {
        &self.state.cuts
    }

    /// Returns frames processed.
    pub fn frames_processed(&self) -> u64 {
        self.state.frames_processed
    }

    fn record_cuts(&mut self, mut cuts: Vec<Cut>) -> Vec<Cut> {
        cuts.sort_by_key(|cut| cut.position.frame_index);
        cuts.dedup_by_key(|cut| cut.position.frame_index);
        let mut accepted = Vec::new();
        for cut in cuts {
            if self
                .state
                .cuts
                .iter()
                .any(|existing| existing.position.frame_index == cut.position.frame_index)
            {
                continue;
            }
            self.state.cuts.push(cut.clone());
            accepted.push(cut);
        }
        self.state.cuts.sort_by_key(|cut| cut.position.frame_index);
        accepted
    }

    fn prepare_frame(&self, frame: OwnedVideoFrame) -> Result<OwnedVideoFrame> {
        self.validate_frame_options(&frame.as_frame())?;
        Ok(frame)
    }

    fn validate_frame_options(&self, frame: &VideoFrame<'_>) -> Result<()> {
        let _ = self.auto_downscale_min_width;
        if let Some(crop) = self.crop {
            if crop.x0 >= frame.width || crop.y0 >= frame.height {
                return Err(DetectError::InvalidArgument(
                    "crop starts outside frame boundary".to_string(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Default)]
/// Data type for scene pipeline builder.
pub struct ScenePipelineBuilder {
    detectors: Vec<Box<dyn SceneDetector>>,
    start_in_scene: bool,
    crop: Option<CropRegion>,
    auto_downscale_min_width: Option<u32>,
}

impl ScenePipelineBuilder {
    /// Returns detector.
    pub fn detector<D: SceneDetector + 'static>(mut self, detector: D) -> Self {
        self.detectors.push(Box::new(detector));
        self
    }

    /// Returns start in scene.
    pub fn start_in_scene(mut self, value: bool) -> Self {
        self.start_in_scene = value;
        self
    }

    /// Returns crop.
    pub fn crop(mut self, value: Option<CropRegion>) -> Self {
        self.crop = value;
        self
    }

    /// Returns auto downscale min width.
    pub fn auto_downscale_min_width(mut self, value: u32) -> Self {
        self.auto_downscale_min_width = Some(value);
        self
    }

    /// Returns build.
    pub fn build(self) -> Result<ScenePipeline> {
        if self.detectors.is_empty() {
            return Err(DetectError::InvalidArgument(
                "at least one detector is required".to_string(),
            ));
        }
        Ok(ScenePipeline {
            detectors: self.detectors,
            start_in_scene: self.start_in_scene,
            crop: self.crop,
            auto_downscale_min_width: self.auto_downscale_min_width,
            state: ScenePipelineState::default(),
        })
    }
}

/// Trait for video analyzer implementations.
pub trait VideoAnalyzer {
    /// Returns name.
    fn name(&self) -> &str;

    /// Returns process frame.
    fn process_frame(&mut self, frame: &VideoFrame<'_>) -> Result<Vec<Observation>>;

    /// Returns finish.
    fn finish(&mut self, _last_position: Option<FramePosition>) -> Result<Vec<Observation>> {
        Ok(Vec::new())
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for sampled video analyzer.
pub struct SampledVideoAnalyzer<A> {
    inner: A,
    every: u64,
}

impl<A> SampledVideoAnalyzer<A> {
    /// Creates a new value.
    pub fn new(inner: A, every: u64) -> Self {
        Self {
            inner,
            every: every.max(1),
        }
    }

    /// Returns inner.
    pub fn inner(&self) -> &A {
        &self.inner
    }

    /// Returns inner mut.
    pub fn inner_mut(&mut self) -> &mut A {
        &mut self.inner
    }

    /// Returns every.
    pub fn every(&self) -> u64 {
        self.every
    }
}

impl<A: VideoAnalyzer> VideoAnalyzer for SampledVideoAnalyzer<A> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn process_frame(&mut self, frame: &VideoFrame<'_>) -> Result<Vec<Observation>> {
        if frame.position.frame_index.is_multiple_of(self.every) {
            self.inner.process_frame(frame)
        } else {
            Ok(Vec::new())
        }
    }

    fn finish(&mut self, last_position: Option<FramePosition>) -> Result<Vec<Observation>> {
        self.inner.finish(last_position)
    }
}

/// Data type for video analysis pipeline.
pub struct VideoAnalysisPipeline {
    analyzers: Vec<Box<dyn VideoAnalyzer>>,
    state: VideoAnalysisPipelineState,
}

#[derive(Debug, Default, Clone)]
struct VideoAnalysisPipelineState {
    observations: Vec<Observation>,
    last_position: Option<FramePosition>,
    frames_processed: u64,
    finished: bool,
}

impl VideoAnalysisPipeline {
    /// Returns builder.
    pub fn builder() -> VideoAnalysisPipelineBuilder {
        VideoAnalysisPipelineBuilder::default()
    }

    /// Returns process frame.
    pub fn process_frame(&mut self, frame: OwnedVideoFrame) -> Result<VideoFrameAnalysis> {
        self.process_frame_ref(&frame.as_frame())
    }

    /// Returns process frame ref.
    pub fn process_frame_ref(&mut self, frame: &VideoFrame<'_>) -> Result<VideoFrameAnalysis> {
        if self.state.finished {
            return Err(DetectError::InvalidArgument(
                "cannot process video frames after finish_analysis; call reset first".to_string(),
            ));
        }
        self.state.last_position = Some(frame.position);

        let mut observations = Vec::new();
        for analyzer in &mut self.analyzers {
            let mut new_observations = analyzer.process_frame(frame)?;
            for observation in &mut new_observations {
                if observation.timestamp.is_none() {
                    observation.timestamp = Some(frame.position.timestamp);
                }
                if observation.frame.is_none() {
                    observation.frame = Some(frame.position);
                }
            }
            observations.append(&mut new_observations);
        }
        self.state.observations.extend(observations.iter().cloned());
        self.state.frames_processed += 1;

        Ok(VideoFrameAnalysis {
            position: frame.position,
            observations,
            frames_processed: self.state.frames_processed,
        })
    }

    /// Returns finish analysis.
    pub fn finish_analysis(&mut self) -> Result<VideoAnalysisResult> {
        if !self.state.finished {
            let mut observations = Vec::new();
            for analyzer in &mut self.analyzers {
                let mut new_observations = analyzer.finish(self.state.last_position)?;
                observations.append(&mut new_observations);
            }
            self.state.observations.extend(observations);
            self.state.finished = true;
        }
        Ok(VideoAnalysisResult {
            observations: self.state.observations.clone(),
            frames_processed: self.state.frames_processed,
        })
    }

    /// Returns reset.
    pub fn reset(&mut self) {
        self.state = VideoAnalysisPipelineState::default();
    }

    /// Returns observations.
    pub fn observations(&self) -> &[Observation] {
        &self.state.observations
    }

    /// Returns frames processed.
    pub fn frames_processed(&self) -> u64 {
        self.state.frames_processed
    }
}

#[derive(Default)]
/// Data type for video analysis pipeline builder.
pub struct VideoAnalysisPipelineBuilder {
    analyzers: Vec<Box<dyn VideoAnalyzer>>,
}

impl VideoAnalysisPipelineBuilder {
    /// Returns analyzer.
    pub fn analyzer<A: VideoAnalyzer + 'static>(mut self, analyzer: A) -> Self {
        self.analyzers.push(Box::new(analyzer));
        self
    }

    /// Returns build.
    pub fn build(self) -> Result<VideoAnalysisPipeline> {
        if self.analyzers.is_empty() {
            return Err(DetectError::InvalidArgument(
                "at least one video analyzer is required".to_string(),
            ));
        }
        Ok(VideoAnalysisPipeline {
            analyzers: self.analyzers,
            state: VideoAnalysisPipelineState::default(),
        })
    }
}

/// Data type for realtime video pipeline.
pub struct RealtimeVideoPipeline {
    scene_pipeline: ScenePipeline,
    video_pipeline: Option<VideoAnalysisPipeline>,
    state: RealtimeVideoPipelineState,
}

#[derive(Debug, Default, Clone)]
struct RealtimeVideoPipelineState {
    observations: Vec<Observation>,
    completed_scenes: Vec<SceneAnalysis>,
    open_scene_observations: Vec<Observation>,
    active_scene_start: Option<FramePosition>,
    active_scene_index: u64,
    first_position: Option<FramePosition>,
    last_position: Option<FramePosition>,
    frames_processed: u64,
    finished: bool,
}

impl RealtimeVideoPipeline {
    /// Returns builder.
    pub fn builder() -> RealtimeVideoPipelineBuilder {
        RealtimeVideoPipelineBuilder::default()
    }

    /// Creates a new value.
    pub fn new(
        scene_pipeline: ScenePipeline,
        video_pipeline: Option<VideoAnalysisPipeline>,
    ) -> Self {
        Self {
            scene_pipeline,
            video_pipeline,
            state: RealtimeVideoPipelineState::default(),
        }
    }

    /// Returns process frame.
    pub fn process_frame(&mut self, frame: OwnedVideoFrame) -> Result<RealtimeVideoFrameAnalysis> {
        self.process_frame_ref(&frame.as_frame())
    }

    /// Returns process frame ref.
    pub fn process_frame_ref(
        &mut self,
        frame: &VideoFrame<'_>,
    ) -> Result<RealtimeVideoFrameAnalysis> {
        if self.state.finished {
            return Err(DetectError::InvalidArgument(
                "cannot process video frames after finish_analysis; call reset first".to_string(),
            ));
        }

        self.state.first_position.get_or_insert(frame.position);
        self.state.last_position = Some(frame.position);
        self.state.active_scene_start.get_or_insert(frame.position);

        let scene = self.scene_pipeline.process_frame_ref(frame)?;
        let completed_scenes = self.close_scenes(&scene.cuts);
        let mut observations = if let Some(pipeline) = &mut self.video_pipeline {
            pipeline.process_frame_ref(frame)?.observations
        } else {
            Vec::new()
        };
        self.annotate_observations(&mut observations, frame.position);
        self.state
            .open_scene_observations
            .extend(observations.iter().cloned());
        self.state.observations.extend(observations.iter().cloned());
        self.state.frames_processed += 1;

        Ok(RealtimeVideoFrameAnalysis {
            position: frame.position,
            scene,
            observations,
            completed_scenes,
            frames_processed: self.state.frames_processed,
        })
    }

    /// Returns finish analysis.
    pub fn finish_analysis(&mut self) -> Result<RealtimeVideoAnalysisResult> {
        let detection = self.scene_pipeline.finish_detection()?;
        if !self.state.finished {
            let pending_cuts = detection
                .cuts
                .iter()
                .filter(|cut| {
                    self.state
                        .active_scene_start
                        .map(|start| cut.position.frame_index > start.frame_index)
                        .unwrap_or(false)
                })
                .cloned()
                .collect::<Vec<_>>();
            self.close_scenes(&pending_cuts);

            if let Some(pipeline) = &mut self.video_pipeline {
                let already_observed = self.state.observations.len();
                let result = pipeline.finish_analysis()?;
                let mut final_observations = result
                    .observations
                    .into_iter()
                    .skip(already_observed)
                    .collect::<Vec<_>>();
                self.annotate_observations_without_frame(&mut final_observations);
                self.state
                    .open_scene_observations
                    .extend(final_observations.iter().cloned());
                self.state
                    .observations
                    .extend(final_observations.iter().cloned());
            }
            self.state.finished = true;
        }

        let scenes = self.scene_analyses_for(&detection.scenes);
        Ok(RealtimeVideoAnalysisResult {
            detection,
            observations: self.state.observations.clone(),
            scenes,
            frames_processed: self.state.frames_processed,
        })
    }

    /// Returns reset.
    pub fn reset(&mut self) {
        self.scene_pipeline.reset();
        if let Some(pipeline) = &mut self.video_pipeline {
            pipeline.reset();
        }
        self.state = RealtimeVideoPipelineState::default();
    }

    /// Returns observations.
    pub fn observations(&self) -> &[Observation] {
        &self.state.observations
    }

    /// Returns completed scenes.
    pub fn completed_scenes(&self) -> &[SceneAnalysis] {
        &self.state.completed_scenes
    }

    /// Returns frames processed.
    pub fn frames_processed(&self) -> u64 {
        self.state.frames_processed
    }

    fn close_scenes(&mut self, cuts: &[Cut]) -> Vec<SceneAnalysis> {
        let mut completed = Vec::new();
        for cut in cuts {
            let Some(start) = self.state.active_scene_start else {
                continue;
            };
            if cut.position.frame_index <= start.frame_index {
                continue;
            }
            let scene = Scene {
                start,
                end: cut.position,
            };
            let observations = std::mem::take(&mut self.state.open_scene_observations);
            let analysis = SceneAnalysis {
                scene_index: self.state.active_scene_index,
                scene,
                observations,
            };
            self.state.completed_scenes.push(analysis.clone());
            completed.push(analysis);
            self.state.active_scene_index += 1;
            self.state.active_scene_start = Some(cut.position);
        }
        completed
    }

    fn annotate_observations(&self, observations: &mut [Observation], position: FramePosition) {
        for observation in observations {
            if observation.timestamp.is_none() {
                observation.timestamp = Some(position.timestamp);
            }
            if observation.frame.is_none() {
                observation.frame = Some(position);
            }
            if observation.scene_index.is_none() {
                observation.scene_index = Some(self.state.active_scene_index);
            }
        }
    }

    fn annotate_observations_without_frame(&self, observations: &mut [Observation]) {
        for observation in observations {
            if observation.scene_index.is_none() {
                observation.scene_index = Some(self.state.active_scene_index);
            }
        }
    }

    fn scene_analyses_for(&self, scenes: &[Scene]) -> Vec<SceneAnalysis> {
        scenes
            .iter()
            .enumerate()
            .map(|(index, scene)| {
                let scene_index = index as u64;
                let observations = self
                    .state
                    .observations
                    .iter()
                    .filter(|observation| observation.scene_index == Some(scene_index))
                    .cloned()
                    .collect();
                SceneAnalysis {
                    scene_index,
                    scene: scene.clone(),
                    observations,
                }
            })
            .collect()
    }
}

#[derive(Default)]
/// Data type for realtime video pipeline builder.
pub struct RealtimeVideoPipelineBuilder {
    scene_pipeline: Option<ScenePipeline>,
    scene_builder: ScenePipelineBuilder,
    video_analyzers: Vec<Box<dyn VideoAnalyzer>>,
}

impl RealtimeVideoPipelineBuilder {
    /// Returns scene pipeline.
    pub fn scene_pipeline(mut self, pipeline: ScenePipeline) -> Self {
        self.scene_pipeline = Some(pipeline);
        self
    }

    /// Returns scene detector.
    pub fn scene_detector<D: SceneDetector + 'static>(mut self, detector: D) -> Self {
        self.scene_builder = self.scene_builder.detector(detector);
        self
    }

    /// Returns video analyzer.
    pub fn video_analyzer<A: VideoAnalyzer + 'static>(mut self, analyzer: A) -> Self {
        self.video_analyzers.push(Box::new(analyzer));
        self
    }

    /// Returns start in scene.
    pub fn start_in_scene(mut self, value: bool) -> Self {
        self.scene_builder = self.scene_builder.start_in_scene(value);
        self
    }

    /// Returns crop.
    pub fn crop(mut self, value: Option<CropRegion>) -> Self {
        self.scene_builder = self.scene_builder.crop(value);
        self
    }

    /// Returns auto downscale min width.
    pub fn auto_downscale_min_width(mut self, value: u32) -> Self {
        self.scene_builder = self.scene_builder.auto_downscale_min_width(value);
        self
    }

    /// Returns build.
    pub fn build(self) -> Result<RealtimeVideoPipeline> {
        let scene_pipeline = match self.scene_pipeline {
            Some(pipeline) => pipeline,
            None => self.scene_builder.build()?,
        };
        let video_pipeline = if self.video_analyzers.is_empty() {
            None
        } else {
            Some(VideoAnalysisPipeline {
                analyzers: self.video_analyzers,
                state: VideoAnalysisPipelineState::default(),
            })
        };
        Ok(RealtimeVideoPipeline::new(scene_pipeline, video_pipeline))
    }
}

/// Returns scenes from cuts.
pub fn scenes_from_cuts(
    cuts: &[Cut],
    start: FramePosition,
    last_frame: FramePosition,
    start_in_scene: bool,
) -> Vec<Scene> {
    if cuts.is_empty() && !start_in_scene {
        return Vec::new();
    }
    let mut scenes = Vec::new();
    let mut scene_start = start;
    for cut in cuts {
        if cut.position.frame_index <= scene_start.frame_index {
            continue;
        }
        scenes.push(Scene {
            start: scene_start,
            end: cut.position,
        });
        scene_start = cut.position;
    }
    scenes.push(Scene {
        start: scene_start,
        end: FramePosition {
            frame_index: last_frame.frame_index + 1,
            timestamp: Timestamp::new(last_frame.timestamp.pts + 1, last_frame.timestamp.timebase),
        },
    });
    scenes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fps() -> Rational64 {
        Rational64::new(30, 1)
    }

    fn pos(frame: u64) -> FramePosition {
        FramePosition::from_frame_index(frame, fps())
    }

    fn owned_frame(frame_index: u64) -> OwnedVideoFrame {
        OwnedVideoFrame {
            position: pos(frame_index),
            width: 1,
            height: 1,
            pixel_format: PixelFormat::Rgb24,
            data: vec![0, 0, 0],
            stride: 3,
        }
    }

    fn content_frame(frame_index: u64, rgb: [u8; 3]) -> OwnedVideoFrame {
        let mut data = Vec::new();
        for _ in 0..16 {
            data.extend_from_slice(&rgb);
        }
        OwnedVideoFrame {
            position: pos(frame_index),
            width: 4,
            height: 4,
            pixel_format: PixelFormat::Rgb24,
            data,
            stride: 12,
        }
    }

    fn content_bgr_frame(frame_index: u64, rgb: [u8; 3]) -> OwnedVideoFrame {
        let mut data = Vec::new();
        for _ in 0..16 {
            data.extend_from_slice(&[rgb[2], rgb[1], rgb[0]]);
        }
        OwnedVideoFrame {
            position: pos(frame_index),
            width: 4,
            height: 4,
            pixel_format: PixelFormat::Bgr24,
            data,
            stride: 12,
        }
    }

    #[test]
    fn timecode_formats_and_clamps_subtraction() {
        let tc = FrameTimecode::from_seconds(10.0, fps()).unwrap();
        assert_eq!(tc.frame_index, 300);
        assert_eq!(tc.timecode(3), "00:00:10.000");
        assert_eq!((FrameTimecode::from_frames(5, fps()) - 10).frame_index, 0);
    }

    #[test]
    fn timecode_parse_accepts_frames_seconds_and_hms() {
        assert_eq!(FrameTimecode::parse("42", fps()).unwrap().frame_index, 42);
        assert_eq!(FrameTimecode::parse("2.0", fps()).unwrap().frame_index, 60);
        assert_eq!(
            FrameTimecode::parse("00:01:00.000", fps())
                .unwrap()
                .frame_index,
            1800
        );
    }

    #[test]
    fn scenes_are_empty_without_cuts_unless_start_in_scene() {
        assert!(scenes_from_cuts(&[], pos(0), pos(9), false).is_empty());
        let scenes = scenes_from_cuts(&[], pos(0), pos(9), true);
        assert_eq!(scenes.len(), 1);
        assert_eq!(scenes[0].end.frame_index, 10);
    }

    #[test]
    fn scenes_are_contiguous_from_unique_cuts() {
        let cuts = vec![Cut {
            position: pos(5),
            detector: "test",
            score: None,
        }];
        let scenes = scenes_from_cuts(&cuts, pos(0), pos(9), true);
        assert_eq!(scenes.len(), 2);
        assert_eq!(scenes[0].start.frame_index, 0);
        assert_eq!(scenes[0].end.frame_index, 5);
        assert_eq!(scenes[1].start.frame_index, 5);
        assert_eq!(scenes[1].end.frame_index, 10);
    }

    #[test]
    fn metrics_store_tracks_keys_and_values() {
        let mut metrics = MetricsStore::default();
        metrics.set_metric(7, "content_val", 12.5);
        assert_eq!(metrics.get(7, "content_val"), Some(12.5));
        assert_eq!(metrics.keys().collect::<Vec<_>>(), vec!["content_val"]);
    }

    #[test]
    fn metrics_store_accepts_dynamic_keys() {
        let mut metrics = MetricsStore::default();
        metrics.set_metric(7, "combined.content.raw", 12.5);
        assert_eq!(metrics.get(7, "combined.content.raw"), Some(12.5));
        assert_eq!(
            metrics.keys().collect::<Vec<_>>(),
            vec!["combined.content.raw"]
        );
    }

    #[test]
    fn content_detector_finds_hard_cut() {
        let mut detector = ContentDetector::new(10.0, 1);
        let mut metrics = MetricsStore::default();
        let first = content_frame(0, [0, 0, 0]);
        let second = content_frame(1, [255, 255, 255]);
        assert!(detector
            .process_frame(&first.as_frame(), Some(&mut metrics))
            .unwrap()
            .is_empty());
        let cuts = detector
            .process_frame(&second.as_frame(), Some(&mut metrics))
            .unwrap();
        assert_eq!(cuts.len(), 1);
        assert!(metrics.get(1, "content_val").unwrap() > 10.0);
    }

    #[test]
    fn content_detector_scores_rgb24_and_bgr24_equivalent_frames_equally() {
        let mut rgb_detector = ContentDetector::new(10.0, 1);
        let mut bgr_detector = ContentDetector::new(10.0, 1);
        let mut rgb_metrics = MetricsStore::default();
        let mut bgr_metrics = MetricsStore::default();
        let rgb_first = content_frame(0, [24, 48, 96]);
        let rgb_second = content_frame(1, [160, 32, 224]);
        let bgr_first = content_bgr_frame(0, [24, 48, 96]);
        let bgr_second = content_bgr_frame(1, [160, 32, 224]);

        rgb_detector
            .process_frame(&rgb_first.as_frame(), Some(&mut rgb_metrics))
            .unwrap();
        rgb_detector
            .process_frame(&rgb_second.as_frame(), Some(&mut rgb_metrics))
            .unwrap();
        bgr_detector
            .process_frame(&bgr_first.as_frame(), Some(&mut bgr_metrics))
            .unwrap();
        bgr_detector
            .process_frame(&bgr_second.as_frame(), Some(&mut bgr_metrics))
            .unwrap();

        for key in ["content_val", "delta_hue", "delta_sat", "delta_lum"] {
            assert_eq!(rgb_metrics.get(1, key), bgr_metrics.get(1, key), "{key}");
        }
    }

    #[test]
    fn content_detector_suppresses_flash() {
        let mut detector = ContentDetector::new(10.0, 3).filter_mode(FlashFilterMode::Suppress, 3);
        let frames = [
            content_frame(0, [0, 0, 0]),
            content_frame(1, [255, 255, 255]),
            content_frame(2, [0, 0, 0]),
        ];
        let mut cuts = Vec::new();
        for frame in frames {
            cuts.extend(detector.process_frame(&frame.as_frame(), None).unwrap());
        }

        assert!(cuts.is_empty());
    }

    #[test]
    fn pipeline_processes_frames_incrementally() {
        struct CutOnFrame(u64);

        impl SceneDetector for CutOnFrame {
            fn name(&self) -> &'static str {
                "test"
            }

            fn metric_keys(&self) -> &'static [&'static str] {
                &[]
            }

            fn process_frame(
                &mut self,
                frame: &VideoFrame<'_>,
                _metrics: Option<&mut dyn MetricsSink>,
            ) -> Result<Vec<Cut>> {
                Ok((frame.position.frame_index == self.0)
                    .then(|| Cut {
                        position: frame.position,
                        detector: self.name(),
                        score: Some(1.0),
                    })
                    .into_iter()
                    .collect())
            }
        }

        let mut pipeline = ScenePipeline::builder()
            .detector(CutOnFrame(1))
            .start_in_scene(true)
            .build()
            .unwrap();

        assert!(pipeline
            .process_frame(owned_frame(0))
            .unwrap()
            .cuts
            .is_empty());
        let analysis = pipeline.process_frame(owned_frame(1)).unwrap();
        assert_eq!(analysis.frames_processed, 2);
        assert_eq!(analysis.cuts[0].position.frame_index, 1);

        let result = pipeline.finish_detection().unwrap();
        assert_eq!(result.frames_processed, 2);
        assert_eq!(result.cuts.len(), 1);
        assert_eq!(result.scenes.len(), 2);
    }

    #[test]
    fn video_pipeline_emits_frame_observations() {
        struct OcrAnalyzer;

        impl VideoAnalyzer for OcrAnalyzer {
            fn name(&self) -> &str {
                "ocr"
            }

            fn process_frame(&mut self, frame: &VideoFrame<'_>) -> Result<Vec<Observation>> {
                Ok(vec![Observation::new(self.name(), ObservationKind::Text)
                    .at_frame(frame.position)
                    .text("EXIT")
                    .score(0.9)])
            }
        }

        let mut pipeline = VideoAnalysisPipeline::builder()
            .analyzer(OcrAnalyzer)
            .build()
            .unwrap();

        let analysis = pipeline.process_frame(owned_frame(3)).unwrap();
        assert_eq!(analysis.frames_processed, 1);
        assert_eq!(analysis.observations[0].text.as_deref(), Some("EXIT"));
        assert_eq!(
            analysis.observations[0].to_text_segment(0).unwrap().text,
            "EXIT"
        );
        assert_eq!(pipeline.finish_analysis().unwrap().observations.len(), 1);
    }

    #[test]
    fn sampled_video_analyzer_skips_unsampled_frames() {
        struct CountingVideoAnalyzer {
            frames: Vec<u64>,
        }

        impl VideoAnalyzer for CountingVideoAnalyzer {
            fn name(&self) -> &str {
                "counting"
            }

            fn process_frame(&mut self, frame: &VideoFrame<'_>) -> Result<Vec<Observation>> {
                self.frames.push(frame.position.frame_index);
                Ok(Vec::new())
            }
        }

        let mut analyzer =
            SampledVideoAnalyzer::new(CountingVideoAnalyzer { frames: Vec::new() }, 3);
        for index in 0..7 {
            analyzer
                .process_frame(&owned_frame(index).as_frame())
                .unwrap();
        }

        assert_eq!(analyzer.inner().frames, vec![0, 3, 6]);
    }

    #[test]
    fn realtime_video_pipeline_groups_observations_by_completed_scene() {
        struct CutOnFrame(u64);

        impl SceneDetector for CutOnFrame {
            fn name(&self) -> &'static str {
                "test"
            }

            fn metric_keys(&self) -> &'static [&'static str] {
                &[]
            }

            fn process_frame(
                &mut self,
                frame: &VideoFrame<'_>,
                _metrics: Option<&mut dyn MetricsSink>,
            ) -> Result<Vec<Cut>> {
                Ok((frame.position.frame_index == self.0)
                    .then(|| Cut {
                        position: frame.position,
                        detector: self.name(),
                        score: Some(1.0),
                    })
                    .into_iter()
                    .collect())
            }
        }

        struct ObjectAnalyzer;

        impl VideoAnalyzer for ObjectAnalyzer {
            fn name(&self) -> &str {
                "objects"
            }

            fn process_frame(&mut self, frame: &VideoFrame<'_>) -> Result<Vec<Observation>> {
                Ok(vec![Observation::new(self.name(), ObservationKind::Object)
                    .at_frame(frame.position)
                    .label(format!("frame-{}", frame.position.frame_index))])
            }
        }

        let mut pipeline = RealtimeVideoPipeline::builder()
            .scene_detector(CutOnFrame(2))
            .video_analyzer(ObjectAnalyzer)
            .start_in_scene(true)
            .build()
            .unwrap();

        assert!(pipeline
            .process_frame(owned_frame(0))
            .unwrap()
            .completed_scenes
            .is_empty());
        pipeline.process_frame(owned_frame(1)).unwrap();
        let analysis = pipeline.process_frame(owned_frame(2)).unwrap();

        assert_eq!(analysis.completed_scenes.len(), 1);
        assert_eq!(analysis.completed_scenes[0].scene.start.frame_index, 0);
        assert_eq!(analysis.completed_scenes[0].scene.end.frame_index, 2);
        assert_eq!(analysis.completed_scenes[0].observations.len(), 2);
        assert_eq!(analysis.observations[0].scene_index, Some(1));

        let result = pipeline.finish_analysis().unwrap();
        assert_eq!(result.detection.scenes.len(), 2);
        assert_eq!(result.scenes[0].observations.len(), 2);
        assert_eq!(result.scenes[1].observations.len(), 1);
    }

    #[test]
    fn audio_pipeline_processes_frames_incrementally() {
        struct LoudnessAnalyzer;

        impl AudioAnalyzer for LoudnessAnalyzer {
            fn name(&self) -> &str {
                "loudness"
            }

            fn process_frame(&mut self, frame: &AudioFrame<'_>) -> Result<Vec<AnalysisEvent>> {
                let AudioBuffer::F32(samples) = frame.data else {
                    return Ok(Vec::new());
                };
                let mean = samples.iter().map(|sample| sample.abs()).sum::<f32>()
                    / samples.len().max(1) as f32;
                Ok((mean > 0.5)
                    .then(|| {
                        AnalysisEvent::new(self.name(), "loud")
                            .at_timestamp(frame.timestamp)
                            .score(mean)
                    })
                    .into_iter()
                    .collect())
            }
        }

        let mut pipeline = AudioPipeline::builder()
            .analyzer(LoudnessAnalyzer)
            .build()
            .unwrap();
        let frame = OwnedAudioFrame::new(
            Timestamp::new(0, Timebase::new(1, 48_000)),
            48_000,
            1,
            AudioBuffer::F32(vec![1.0, 0.5]),
        )
        .unwrap();

        let analysis = pipeline.process_frame(frame).unwrap();
        assert_eq!(analysis.frames_processed, 1);
        assert_eq!(analysis.events[0].label, "loud");
        assert_eq!(pipeline.finish_analysis().unwrap().events.len(), 1);
    }

    #[test]
    fn text_pipeline_processes_segments_incrementally() {
        struct KeywordAnalyzer;

        impl TextAnalyzer for KeywordAnalyzer {
            fn name(&self) -> &str {
                "keyword"
            }

            fn process_segment(&mut self, segment: &TextSegment<'_>) -> Result<Vec<AnalysisEvent>> {
                Ok(segment
                    .text
                    .contains("cut")
                    .then(|| {
                        let mut event = AnalysisEvent::new(self.name(), "keyword").score(1.0);
                        if let Some(timestamp) = segment.timestamp {
                            event = event.at_timestamp(timestamp);
                        }
                        event
                    })
                    .into_iter()
                    .collect())
            }
        }

        let mut pipeline = TextPipeline::builder()
            .analyzer(KeywordAnalyzer)
            .build()
            .unwrap();
        let segment = OwnedTextSegment::new(0, "find this cut point");

        let analysis = pipeline.process_segment(segment).unwrap();
        assert_eq!(analysis.segments_processed, 1);
        assert_eq!(analysis.events[0].analyzer, "keyword");
        assert_eq!(pipeline.finish_analysis().unwrap().segments_processed, 1);
    }

    #[test]
    fn crop_rejects_empty_regions() {
        assert!(CropRegion::new(0, 0, 0, 10).is_err());
    }
}
