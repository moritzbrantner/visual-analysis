#![doc = include_str!("../README.md")]

pub mod surface;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use num_rational::Rational64;
use video_analysis_core::{
    AudioAnalysis, AudioAnalysisResult, AudioPipeline, AudioSampleFormat, DetectionResult,
    FrameAnalysis, OwnedAudioFrame, OwnedTextSegment, OwnedVideoFrame, PixelFormat,
    RealtimeVideoAnalysisResult, RealtimeVideoFrameAnalysis, RealtimeVideoPipeline, Result,
    ScenePipeline, TextAnalysis, TextAnalysisResult, TextPipeline, VideoAnalysisPipeline,
    VideoAnalysisResult, VideoFrameAnalysis,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Variants describing source mode.
pub enum SourceMode {
    /// The recorded variant.
    Recorded,
    /// The live variant.
    Live,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for media source info.
pub struct MediaSourceInfo {
    /// The input value.
    pub input: String,
    /// The mode value.
    pub mode: SourceMode,
    /// The video value.
    pub video: Option<VideoStreamInfo>,
    /// The audio value.
    pub audio: Vec<AudioStreamInfo>,
    /// Text content for this value.
    pub text: Vec<TextStreamInfo>,
}

impl MediaSourceInfo {
    /// Returns recorded.
    pub fn recorded(input: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            mode: SourceMode::Recorded,
            video: None,
            audio: Vec::new(),
            text: Vec::new(),
        }
    }

    /// Returns live.
    pub fn live(input: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            mode: SourceMode::Live,
            video: None,
            audio: Vec::new(),
            text: Vec::new(),
        }
    }

    /// Returns this value with video.
    pub fn with_video(mut self, video: VideoStreamInfo) -> Self {
        self.video = Some(video);
        self
    }

    /// Returns this value with audio.
    pub fn with_audio(mut self, audio: AudioStreamInfo) -> Self {
        self.audio.push(audio);
        self
    }

    /// Returns this value with text.
    pub fn with_text(mut self, text: TextStreamInfo) -> Self {
        self.text.push(text);
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for video stream info.
pub struct VideoStreamInfo {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// The frame rate value.
    pub frame_rate: Option<Rational64>,
    /// The pixel format value.
    pub pixel_format: PixelFormat,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Data type for audio stream info.
pub struct AudioStreamInfo {
    /// Sample rate in hertz.
    pub sample_rate: u32,
    /// Number of audio channels.
    pub channels: u16,
    /// The sample format value.
    pub sample_format: AudioSampleFormat,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Data type for text stream info.
pub struct TextStreamInfo {
    /// The format value.
    pub format: TextFormat,
    /// Language tag for this value.
    pub language: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Variants describing text format.
pub enum TextFormat {
    /// The plain variant.
    Plain,
    /// The lines variant.
    Lines,
    /// The transcript variant.
    Transcript,
    /// The subtitles variant.
    Subtitles,
}

#[derive(Debug, Clone, PartialEq)]
/// Variants describing media sample.
pub enum MediaSample {
    /// The video variant.
    Video(OwnedVideoFrame),
    /// The audio variant.
    Audio(OwnedAudioFrame),
    /// The text variant.
    Text(OwnedTextSegment),
}

/// Trait for media source implementations.
pub trait MediaSource {
    /// Returns source info.
    fn source_info(&self) -> &MediaSourceInfo;
    /// Returns next sample.
    fn next_sample(&mut self) -> Result<Option<MediaSample>>;

    /// Returns mode.
    fn mode(&self) -> SourceMode {
        self.source_info().mode
    }

    /// Returns whether is live.
    fn is_live(&self) -> bool {
        self.mode() == SourceMode::Live
    }
}

/// Trait for video frame source implementations.
pub trait VideoFrameSource {
    /// Returns source info.
    fn source_info(&self) -> &MediaSourceInfo;
    /// Returns next video frame.
    fn next_video_frame(&mut self) -> Result<Option<OwnedVideoFrame>>;

    /// Returns frame rate.
    fn frame_rate(&self) -> Option<Rational64> {
        self.source_info()
            .video
            .as_ref()
            .and_then(|video| video.frame_rate)
    }

    /// Returns whether is live.
    fn is_live(&self) -> bool {
        self.source_info().mode == SourceMode::Live
    }
}

/// Trait for audio frame source implementations.
pub trait AudioFrameSource {
    /// Returns source info.
    fn source_info(&self) -> &MediaSourceInfo;
    /// Returns next audio frame.
    fn next_audio_frame(&mut self) -> Result<Option<OwnedAudioFrame>>;

    /// Returns whether is live.
    fn is_live(&self) -> bool {
        self.source_info().mode == SourceMode::Live
    }
}

/// Trait for text segment source implementations.
pub trait TextSegmentSource {
    /// Returns source info.
    fn source_info(&self) -> &MediaSourceInfo;
    /// Returns next text segment.
    fn next_text_segment(&mut self) -> Result<Option<OwnedTextSegment>>;

    /// Returns whether is live.
    fn is_live(&self) -> bool {
        self.source_info().mode == SourceMode::Live
    }
}

/// Returns analyze video source.
pub fn analyze_video_source<S, F>(
    source: &mut S,
    pipeline: &mut ScenePipeline,
    mut on_frame: F,
) -> Result<DetectionResult>
where
    S: VideoFrameSource,
    F: FnMut(&FrameAnalysis) -> Result<()>,
{
    pipeline.reset();
    while let Some(frame) = source.next_video_frame()? {
        let analysis = pipeline.process_frame(frame)?;
        on_frame(&analysis)?;
    }
    pipeline.finish_detection()
}

/// Returns analyze video frames.
pub fn analyze_video_frames<S, F>(
    source: &mut S,
    pipeline: &mut VideoAnalysisPipeline,
    mut on_frame: F,
) -> Result<VideoAnalysisResult>
where
    S: VideoFrameSource,
    F: FnMut(&VideoFrameAnalysis) -> Result<()>,
{
    pipeline.reset();
    while let Some(frame) = source.next_video_frame()? {
        let analysis = pipeline.process_frame(frame)?;
        on_frame(&analysis)?;
    }
    pipeline.finish_analysis()
}

/// Returns analyze realtime video source.
pub fn analyze_realtime_video_source<S, F>(
    source: &mut S,
    pipeline: &mut RealtimeVideoPipeline,
    mut on_frame: F,
) -> Result<RealtimeVideoAnalysisResult>
where
    S: VideoFrameSource,
    F: FnMut(&RealtimeVideoFrameAnalysis) -> Result<()>,
{
    pipeline.reset();
    while let Some(frame) = source.next_video_frame()? {
        let analysis = pipeline.process_frame(frame)?;
        on_frame(&analysis)?;
    }
    pipeline.finish_analysis()
}

/// Returns analyze audio source.
pub fn analyze_audio_source<S, F>(
    source: &mut S,
    pipeline: &mut AudioPipeline,
    mut on_frame: F,
) -> Result<AudioAnalysisResult>
where
    S: AudioFrameSource,
    F: FnMut(&AudioAnalysis) -> Result<()>,
{
    pipeline.reset();
    while let Some(frame) = source.next_audio_frame()? {
        let analysis = pipeline.process_frame(frame)?;
        on_frame(&analysis)?;
    }
    pipeline.finish_analysis()
}

/// Returns analyze text source.
pub fn analyze_text_source<S, F>(
    source: &mut S,
    pipeline: &mut TextPipeline,
    mut on_segment: F,
) -> Result<TextAnalysisResult>
where
    S: TextSegmentSource,
    F: FnMut(&TextAnalysis) -> Result<()>,
{
    pipeline.reset();
    while let Some(segment) = source.next_text_segment()? {
        let analysis = pipeline.process_segment(segment)?;
        on_segment(&analysis)?;
    }
    pipeline.finish_analysis()
}

/// Data type for text line source.
pub struct TextLineSource<R> {
    source_info: MediaSourceInfo,
    reader: R,
    next_segment_index: u64,
    language: Option<String>,
}

impl<R: BufRead> TextLineSource<R> {
    /// Returns recorded.
    pub fn recorded(input: impl Into<String>, reader: R) -> Self {
        Self::new(SourceMode::Recorded, input, reader)
    }

    /// Returns live.
    pub fn live(input: impl Into<String>, reader: R) -> Self {
        Self::new(SourceMode::Live, input, reader)
    }

    /// Returns this value with language.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        let language = language.into();
        self.language = Some(language.clone());
        if let Some(text) = self.source_info.text.first_mut() {
            text.language = Some(language);
        }
        self
    }

    fn new(mode: SourceMode, input: impl Into<String>, reader: R) -> Self {
        let input = input.into();
        let source_info = MediaSourceInfo {
            input,
            mode,
            video: None,
            audio: Vec::new(),
            text: vec![TextStreamInfo {
                format: TextFormat::Lines,
                language: None,
            }],
        };
        Self {
            source_info,
            reader,
            next_segment_index: 0,
            language: None,
        }
    }
}

impl TextLineSource<BufReader<File>> {
    /// Returns open.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        Ok(Self::recorded(
            path.to_string_lossy().into_owned(),
            BufReader::new(file),
        ))
    }
}

impl<R: BufRead> TextSegmentSource for TextLineSource<R> {
    fn source_info(&self) -> &MediaSourceInfo {
        &self.source_info
    }

    fn next_text_segment(&mut self) -> Result<Option<OwnedTextSegment>> {
        let mut line = String::new();
        let bytes = self.reader.read_line(&mut line)?;
        if bytes == 0 {
            return Ok(None);
        }
        let text = line.trim_end_matches(['\r', '\n']).to_string();
        let segment_index = self.next_segment_index;
        self.next_segment_index += 1;
        let mut segment = OwnedTextSegment::new(segment_index, text);
        if let Some(language) = &self.language {
            segment = segment.language(language.clone());
        }
        Ok(Some(segment))
    }
}

impl<R: BufRead> MediaSource for TextLineSource<R> {
    fn source_info(&self) -> &MediaSourceInfo {
        &self.source_info
    }

    fn next_sample(&mut self) -> Result<Option<MediaSample>> {
        self.next_text_segment()
            .map(|segment| segment.map(MediaSample::Text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use video_analysis_core::{AudioBuffer, FramePosition, PixelFormat, Timebase, Timestamp};

    #[test]
    fn audio_frame_reports_samples_per_channel_and_duration() {
        let frame = OwnedAudioFrame {
            timestamp: Timestamp::new(0, Timebase::new(1, 48_000)),
            sample_rate: 48_000,
            channels: 2,
            data: AudioBuffer::F32(vec![0.0; 960]),
        };

        assert_eq!(frame.sample_format(), AudioSampleFormat::F32);
        assert_eq!(frame.samples_per_channel(), 480);
        assert_eq!(frame.duration_seconds(), 0.01);
    }

    #[test]
    fn media_source_info_builders_track_mode_and_streams() {
        let info = MediaSourceInfo::live("rtsp://example/stream").with_video(VideoStreamInfo {
            width: 1920,
            height: 1080,
            frame_rate: Some(Rational64::new(30, 1)),
            pixel_format: PixelFormat::Rgb24,
        });

        assert_eq!(info.mode, SourceMode::Live);
        assert_eq!(info.video.unwrap().width, 1920);
    }

    #[test]
    fn media_sample_can_hold_video_frames() {
        let frame = OwnedVideoFrame {
            position: FramePosition::from_frame_index(0, Rational64::new(30, 1)),
            width: 1,
            height: 1,
            pixel_format: PixelFormat::Rgb24,
            data: vec![0, 0, 0],
            stride: 3,
        };

        assert!(matches!(MediaSample::Video(frame), MediaSample::Video(_)));
    }

    #[test]
    fn text_line_source_yields_segments() {
        let input = std::io::Cursor::new("first\nsecond\n");
        let mut source = TextLineSource::recorded("memory", input).with_language("en");

        let first = source.next_text_segment().unwrap().unwrap();
        let second = source.next_text_segment().unwrap().unwrap();

        assert_eq!(first.segment_index, 0);
        assert_eq!(first.text, "first");
        assert_eq!(first.language.as_deref(), Some("en"));
        assert_eq!(second.segment_index, 1);
        assert!(source.next_text_segment().unwrap().is_none());
    }
}
