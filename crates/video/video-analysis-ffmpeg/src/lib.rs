#![doc = include_str!("../README.md")]

pub mod surface;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};

use num_rational::Rational64;
use thiserror::Error;
use video_analysis_core::{
    AudioBuffer, AudioSampleFormat, DetectError, FramePosition, OwnedAudioFrame, OwnedVideoFrame,
    PixelFormat, Result, Timebase, Timestamp, VideoSource,
};
use video_analysis_ingest::{
    AudioFrameSource, AudioStreamInfo, MediaSample, MediaSource, MediaSourceInfo, SourceMode,
    VideoFrameSource, VideoStreamInfo,
};

#[derive(Debug, Error)]
/// Variants describing FFmpeg error.
pub enum FfmpegError {
    #[error("ffprobe failed for `{input}`: {message}")]
    /// The probe failed variant.
    ProbeFailed {
        /// Input value that triggered this variant.
        input: String,
        /// Diagnostic message for this variant.
        message: String,
    },
    #[error("ffmpeg failed to start for `{input}`: {message}")]
    /// The start failed variant.
    StartFailed {
        /// Input value that triggered this variant.
        input: String,
        /// Diagnostic message for this variant.
        message: String,
    },
    #[error("missing or invalid video metadata: {0}")]
    /// The invalid metadata variant.
    InvalidMetadata(String),
    #[error("invalid audio stream selection {selection:?}: {reason:?}")]
    /// Audio stream selection failed before decode.
    InvalidAudioStreamSelection {
        /// Requested stream selection.
        selection: AudioStreamSelection,
        /// Why the selection failed.
        reason: AudioStreamSelectionErrorReason,
        /// All streams available in the probed input.
        available_streams: MediaStreamInventory,
    },
    #[error("unsupported FFmpeg runtime for decode: {message}")]
    /// The selected runtime cannot perform this decode operation.
    UnsupportedRuntime {
        /// Diagnostic message for this variant.
        message: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// A typed way to identify an audio stream.
pub enum AudioStreamSelection {
    /// Zero-based ordinal among audio streams (`0:a:N` in FFmpeg syntax).
    AudioOrdinal(usize),
    /// Global container stream index, useful for validating inventory-driven choices.
    GlobalStreamIndex(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Reason an audio stream selection could not be resolved.
pub enum AudioStreamSelectionErrorReason {
    /// The input contains no audio streams.
    NoAudioStreams,
    /// The requested ordinal or global index is not present.
    OutOfRange,
    /// The requested global stream exists but is not audio.
    NotAudio,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Media type reported for an FFprobe stream.
pub enum MediaType {
    /// Video stream.
    Video,
    /// Audio stream.
    Audio,
    /// Subtitle stream.
    Subtitle,
    /// Data stream.
    Data,
    /// Attachment stream.
    Attachment,
    /// A stream type not recognized by this crate version.
    Unknown(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Typed FFprobe metadata for one media stream.
pub struct MediaStream {
    /// Global stream index within the input container.
    pub index: u32,
    /// Media type reported by FFprobe.
    pub media_type: MediaType,
    /// Zero-based ordinal among audio streams, when this is an audio stream.
    pub audio_stream_ordinal: Option<usize>,
    /// Codec name, when reported.
    pub codec: Option<String>,
    /// Channel count for audio streams, when reported.
    pub channels: Option<u16>,
    /// Sample rate in hertz for audio streams, when reported.
    pub sample_rate: Option<u32>,
    /// Language tag, when reported.
    pub language: Option<String>,
    /// Whether FFmpeg marks this stream as the default, when reported.
    pub default_disposition: Option<bool>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
/// Typed inventory of every stream reported by FFprobe.
pub struct MediaStreamInventory {
    /// Streams in FFprobe order.
    pub streams: Vec<MediaStream>,
}

/// Resolves and validates an audio stream selection against a probed inventory.
pub fn validate_audio_stream_selection(
    inventory: &MediaStreamInventory,
    selection: AudioStreamSelection,
) -> std::result::Result<&MediaStream, FfmpegError> {
    let selected = match selection {
        AudioStreamSelection::AudioOrdinal(ordinal) => {
            if !inventory
                .streams
                .iter()
                .any(|stream| stream.media_type == MediaType::Audio)
            {
                return Err(invalid_audio_stream_selection(
                    inventory,
                    selection,
                    AudioStreamSelectionErrorReason::NoAudioStreams,
                ));
            }
            inventory
                .streams
                .iter()
                .find(|stream| stream.audio_stream_ordinal == Some(ordinal))
        }
        AudioStreamSelection::GlobalStreamIndex(index) => {
            let stream = inventory
                .streams
                .iter()
                .find(|stream| stream.index == index);
            if stream.is_some_and(|stream| stream.media_type != MediaType::Audio) {
                return Err(invalid_audio_stream_selection(
                    inventory,
                    selection,
                    AudioStreamSelectionErrorReason::NotAudio,
                ));
            }
            stream
        }
    };
    selected.ok_or_else(|| {
        invalid_audio_stream_selection(
            inventory,
            selection,
            AudioStreamSelectionErrorReason::OutOfRange,
        )
    })
}

fn invalid_audio_stream_selection(
    inventory: &MediaStreamInventory,
    selection: AudioStreamSelection,
    reason: AudioStreamSelectionErrorReason,
) -> FfmpegError {
    FfmpegError::InvalidAudioStreamSelection {
        selection,
        reason,
        available_streams: inventory.clone(),
    }
}

fn into_detect_error(error: FfmpegError) -> DetectError {
    match error {
        FfmpegError::UnsupportedRuntime { message } => DetectError::InvalidArgument(message),
        error => DetectError::Source(error.to_string()),
    }
}

#[derive(Debug, Clone)]
/// Data type for video metadata.
pub struct VideoMetadata {
    /// The input value.
    pub input: String,
    /// Filesystem path for this value.
    pub path: Option<PathBuf>,
    /// The mode value.
    pub mode: SourceMode,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// The frame rate value.
    pub frame_rate: Rational64,
    /// Duration in seconds.
    pub duration_seconds: Option<f64>,
}

#[derive(Debug, Clone)]
/// Data type for audio metadata.
pub struct AudioMetadata {
    /// The input value.
    pub input: String,
    /// Filesystem path for this value.
    pub path: Option<PathBuf>,
    /// The mode value.
    pub mode: SourceMode,
    /// Sample rate in hertz.
    pub sample_rate: u32,
    /// Number of audio channels.
    pub channels: u16,
    /// Duration in seconds.
    pub duration_seconds: Option<f64>,
}

#[derive(Debug, Clone, Default)]
/// Runtime backend used by FFmpeg integration APIs.
pub enum FfmpegRuntimeBackend {
    /// Native `ffmpeg-next` backend.
    Native,
    /// External `ffmpeg`/`ffprobe` command backend.
    #[default]
    Command,
}

#[derive(Debug, Clone, Default)]
/// Runtime options for FFmpeg integration APIs.
pub struct FfmpegRuntimeOptions {
    /// Backend selection.
    pub backend: FfmpegRuntimeBackend,
}

impl FfmpegRuntimeOptions {
    /// Returns command runtime options.
    pub fn command() -> Self {
        Self {
            backend: FfmpegRuntimeBackend::Command,
        }
    }

    /// Returns native runtime options.
    pub fn native() -> Self {
        Self {
            backend: FfmpegRuntimeBackend::Native,
        }
    }
}

#[derive(Debug, Clone)]
/// Data type for FFmpeg source options.
pub struct FfmpegSourceOptions {
    /// The mode value.
    pub mode: SourceMode,
    /// The realtime value.
    pub realtime: bool,
    /// The extra input args value.
    pub extra_input_args: Vec<String>,
    /// The extra output args value.
    pub extra_output_args: Vec<String>,
    /// Resize output to this width while preserving aspect ratio.
    pub resize_width: Option<u32>,
    /// Output width override after output-side filters.
    pub output_width: Option<u32>,
    /// Output height override after output-side filters.
    pub output_height: Option<u32>,
    /// Runtime options.
    pub runtime: FfmpegRuntimeOptions,
    /// Packed video pixel format emitted by the rawvideo pipe.
    pub pixel_format: PixelFormat,
}

impl FfmpegSourceOptions {
    /// Returns recorded.
    pub fn recorded() -> Self {
        Self {
            mode: SourceMode::Recorded,
            realtime: false,
            extra_input_args: Vec::new(),
            extra_output_args: Vec::new(),
            resize_width: None,
            output_width: None,
            output_height: None,
            runtime: FfmpegRuntimeOptions::command(),
            pixel_format: PixelFormat::Rgb24,
        }
    }

    /// Returns live.
    pub fn live() -> Self {
        Self {
            mode: SourceMode::Live,
            realtime: true,
            extra_input_args: Vec::new(),
            extra_output_args: Vec::new(),
            resize_width: None,
            output_width: None,
            output_height: None,
            runtime: FfmpegRuntimeOptions::command(),
            pixel_format: PixelFormat::Rgb24,
        }
    }

    /// Returns extra input arg.
    pub fn extra_input_arg(mut self, arg: impl Into<String>) -> Self {
        self.extra_input_args.push(arg.into());
        self
    }

    /// Returns extra output arg.
    pub fn extra_output_arg(mut self, arg: impl Into<String>) -> Self {
        self.extra_output_args.push(arg.into());
        self
    }

    /// Returns resize width.
    pub fn resize_width(mut self, width: u32) -> Self {
        self.resize_width = Some(width);
        self
    }

    /// Returns output dimensions.
    pub fn output_dimensions(mut self, width: u32, height: u32) -> Self {
        self.output_width = Some(width);
        self.output_height = Some(height);
        self
    }

    /// Returns pixel format.
    pub fn pixel_format(mut self, pixel_format: PixelFormat) -> Self {
        self.pixel_format = pixel_format;
        self
    }

    /// Returns runtime.
    pub fn runtime(mut self, runtime: FfmpegRuntimeOptions) -> Self {
        self.runtime = runtime;
        self
    }
}

#[derive(Debug, Clone)]
/// Data type for FFmpeg audio source options.
pub struct FfmpegAudioSourceOptions {
    /// The mode value.
    pub mode: SourceMode,
    /// The realtime value.
    pub realtime: bool,
    /// The samples per chunk value.
    pub samples_per_chunk: usize,
    /// The extra input args value.
    pub extra_input_args: Vec<String>,
    /// Zero-based ordinal of the audio stream to decode.
    ///
    /// When omitted, FFmpeg's existing first-audio-stream behavior is retained.
    pub audio_stream_index: Option<usize>,
    /// Runtime options.
    pub runtime: FfmpegRuntimeOptions,
}

impl FfmpegAudioSourceOptions {
    /// Returns recorded.
    pub fn recorded() -> Self {
        Self {
            mode: SourceMode::Recorded,
            realtime: false,
            samples_per_chunk: 1024,
            extra_input_args: Vec::new(),
            audio_stream_index: None,
            runtime: FfmpegRuntimeOptions::command(),
        }
    }

    /// Returns live.
    pub fn live() -> Self {
        Self {
            mode: SourceMode::Live,
            realtime: true,
            samples_per_chunk: 1024,
            extra_input_args: Vec::new(),
            audio_stream_index: None,
            runtime: FfmpegRuntimeOptions::command(),
        }
    }

    /// Returns samples per chunk.
    pub fn samples_per_chunk(mut self, samples: usize) -> Self {
        self.samples_per_chunk = samples.max(1);
        self
    }

    /// Returns extra input arg.
    pub fn extra_input_arg(mut self, arg: impl Into<String>) -> Self {
        self.extra_input_args.push(arg.into());
        self
    }

    /// Selects a zero-based audio-stream ordinal for decoding.
    pub fn audio_stream_index(mut self, index: usize) -> Self {
        self.audio_stream_index = Some(index);
        self
    }

    /// Returns runtime.
    pub fn runtime(mut self, runtime: FfmpegRuntimeOptions) -> Self {
        self.runtime = runtime;
        self
    }
}

/// Data type for FFmpeg video source.
pub struct FfmpegVideoSource {
    metadata: VideoMetadata,
    source_info: MediaSourceInfo,
    child: Child,
    stdout: ChildStdout,
    next_frame_index: u64,
    frame_size: usize,
}

impl FfmpegVideoSource {
    /// Returns open.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_path_with_options(path, FfmpegSourceOptions::recorded())
    }

    /// Returns open path with options.
    pub fn open_path_with_options(
        path: impl AsRef<Path>,
        options: FfmpegSourceOptions,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let input = path.to_string_lossy().into_owned();
        let metadata = probe_path_with_runtime(&path, options.mode, &options.runtime)
            .map_err(|err| DetectError::Source(err.to_string()))?;
        Self::spawn(input, metadata, options)
    }

    /// Returns open input.
    pub fn open_input(input: impl Into<String>) -> Result<Self> {
        Self::open_input_with_options(input, FfmpegSourceOptions::recorded())
    }

    /// Returns open live.
    pub fn open_live(input: impl Into<String>) -> Result<Self> {
        Self::open_input_with_options(input, FfmpegSourceOptions::live())
    }

    /// Returns open input with options.
    pub fn open_input_with_options(
        input: impl Into<String>,
        options: FfmpegSourceOptions,
    ) -> Result<Self> {
        let input = input.into();
        let metadata = probe_input_with_runtime(&input, None, options.mode, &options.runtime)
            .map_err(|err| DetectError::Source(err.to_string()))?;
        Self::spawn(input, metadata, options)
    }

    fn spawn(input: String, metadata: VideoMetadata, options: FfmpegSourceOptions) -> Result<Self> {
        reject_native_runtime(&options.runtime)?;
        let resized = options.resize_width.map(|width| {
            (
                width,
                scaled_even_height(metadata.width, metadata.height, width),
            )
        });
        let output_width = options
            .output_width
            .or_else(|| resized.map(|(width, _)| width))
            .unwrap_or(metadata.width);
        let output_height = options
            .output_height
            .or_else(|| resized.map(|(_, height)| height))
            .unwrap_or(metadata.height);
        let source_info = MediaSourceInfo {
            input: metadata.input.clone(),
            mode: metadata.mode,
            video: Some(VideoStreamInfo {
                width: output_width,
                height: output_height,
                frame_rate: Some(metadata.frame_rate),
                pixel_format: options.pixel_format,
            }),
            audio: Vec::new(),
            text: Vec::new(),
        };
        let mut command = Command::new("ffmpeg");
        command.arg("-v").arg("error");
        if options.realtime {
            command
                .arg("-fflags")
                .arg("nobuffer")
                .arg("-flags")
                .arg("low_delay");
        }
        for arg in &options.extra_input_args {
            command.arg(arg);
        }
        command
            .arg("-i")
            .arg(&input)
            .arg("-map")
            .arg("0:v:0")
            .arg("-an");
        let resize_filter = options
            .resize_width
            .map(|width| format!("scale={width}:-2"));
        if let Some(filter) = resize_filter {
            command.arg("-vf").arg(filter);
        }
        for arg in &options.extra_output_args {
            command.arg(arg);
        }
        command
            .arg("-f")
            .arg("rawvideo")
            .arg("-pix_fmt")
            .arg(ffmpeg_pixel_format(options.pixel_format))
            .arg("-vsync")
            .arg("0")
            .arg("pipe:1")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = command.spawn().map_err(|err| {
            DetectError::Source(
                FfmpegError::StartFailed {
                    input: input.clone(),
                    message: err.to_string(),
                }
                .to_string(),
            )
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            DetectError::Source("ffmpeg stdout pipe was not available".to_string())
        })?;
        let frame_size = output_width as usize * output_height as usize * 3;
        Ok(Self {
            metadata,
            source_info,
            child,
            stdout,
            next_frame_index: 0,
            frame_size,
        })
    }

    /// Returns metadata.
    pub fn metadata(&self) -> &VideoMetadata {
        &self.metadata
    }

    /// Returns source info.
    pub fn source_info(&self) -> &MediaSourceInfo {
        &self.source_info
    }

    fn read_next_frame(&mut self) -> Result<Option<OwnedVideoFrame>> {
        let mut data = vec![0_u8; self.frame_size];
        let mut offset = 0;
        while offset < self.frame_size {
            match self.stdout.read(&mut data[offset..]) {
                Ok(0) if offset == 0 => return Ok(None),
                Ok(0) => {
                    return Err(DetectError::Source(
                        "ffmpeg ended in the middle of a raw frame".to_string(),
                    ))
                }
                Ok(bytes) => offset += bytes,
                Err(err) if err.kind() == std::io::ErrorKind::Interrupted => {}
                Err(err) => return Err(DetectError::Io(err)),
            }
        }
        let position =
            FramePosition::from_frame_index(self.next_frame_index, self.metadata.frame_rate);
        self.next_frame_index += 1;
        Ok(Some(OwnedVideoFrame {
            position,
            width: self.source_info.video.as_ref().unwrap().width,
            height: self.source_info.video.as_ref().unwrap().height,
            pixel_format: self.source_info.video.as_ref().unwrap().pixel_format,
            data,
            stride: self.source_info.video.as_ref().unwrap().width as usize * 3,
        }))
    }
}

fn ffmpeg_pixel_format(pixel_format: PixelFormat) -> &'static str {
    match pixel_format {
        PixelFormat::Rgb24 => "rgb24",
        PixelFormat::Bgr24 => "bgr24",
    }
}

fn scaled_even_height(input_width: u32, input_height: u32, output_width: u32) -> u32 {
    let scaled = u64::from(input_height) * u64::from(output_width) / u64::from(input_width);
    let even = scaled - (scaled % 2);
    even.max(2) as u32
}

impl Drop for FfmpegVideoSource {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl VideoSource for FfmpegVideoSource {
    fn next_frame(&mut self) -> Result<Option<OwnedVideoFrame>> {
        self.read_next_frame()
    }

    fn frame_rate(&self) -> Rational64 {
        self.metadata.frame_rate
    }
}

impl VideoFrameSource for FfmpegVideoSource {
    fn source_info(&self) -> &MediaSourceInfo {
        &self.source_info
    }

    fn next_video_frame(&mut self) -> Result<Option<OwnedVideoFrame>> {
        self.read_next_frame()
    }
}

impl MediaSource for FfmpegVideoSource {
    fn source_info(&self) -> &MediaSourceInfo {
        &self.source_info
    }

    fn next_sample(&mut self) -> Result<Option<MediaSample>> {
        self.read_next_frame()
            .map(|frame| frame.map(MediaSample::Video))
    }
}

/// Data type for FFmpeg audio source.
pub struct FfmpegAudioSource {
    metadata: AudioMetadata,
    source_info: MediaSourceInfo,
    child: Child,
    stdout: ChildStdout,
    next_sample_index: u64,
    samples_per_chunk: usize,
}

impl FfmpegAudioSource {
    /// Returns open.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_path_with_options(path, FfmpegAudioSourceOptions::recorded())
    }

    /// Returns open path with options.
    pub fn open_path_with_options(
        path: impl AsRef<Path>,
        options: FfmpegAudioSourceOptions,
    ) -> Result<Self> {
        Self::open_path_with_options_checked(path, options).map_err(into_detect_error)
    }

    /// Opens a path while preserving typed FFprobe and stream-selection errors.
    pub fn open_path_with_options_checked(
        path: impl AsRef<Path>,
        options: FfmpegAudioSourceOptions,
    ) -> std::result::Result<Self, FfmpegError> {
        let path = path.as_ref().to_path_buf();
        let input = path.to_string_lossy().into_owned();
        let metadata = probe_selected_audio_input(
            &input,
            Some(path),
            options.mode,
            &options.runtime,
            options.audio_stream_index,
        )?;
        Self::spawn(input, metadata, options)
    }

    /// Returns open input.
    pub fn open_input(input: impl Into<String>) -> Result<Self> {
        Self::open_input_with_options(input, FfmpegAudioSourceOptions::recorded())
    }

    /// Returns open live.
    pub fn open_live(input: impl Into<String>) -> Result<Self> {
        Self::open_input_with_options(input, FfmpegAudioSourceOptions::live())
    }

    /// Returns open input with options.
    pub fn open_input_with_options(
        input: impl Into<String>,
        options: FfmpegAudioSourceOptions,
    ) -> Result<Self> {
        Self::open_input_with_options_checked(input, options).map_err(into_detect_error)
    }

    /// Opens an input while preserving typed FFprobe and stream-selection errors.
    pub fn open_input_with_options_checked(
        input: impl Into<String>,
        options: FfmpegAudioSourceOptions,
    ) -> std::result::Result<Self, FfmpegError> {
        let input = input.into();
        let metadata = probe_selected_audio_input(
            &input,
            None,
            options.mode,
            &options.runtime,
            options.audio_stream_index,
        )?;
        Self::spawn(input, metadata, options)
    }

    fn spawn(
        input: String,
        metadata: AudioMetadata,
        options: FfmpegAudioSourceOptions,
    ) -> std::result::Result<Self, FfmpegError> {
        if matches!(options.runtime.backend, FfmpegRuntimeBackend::Native) {
            return Err(FfmpegError::UnsupportedRuntime {
                message: "ffmpeg-native runtime is available for probing in this build; decode/extract still uses the command backend".to_string(),
            });
        }
        let source_info = MediaSourceInfo {
            input: metadata.input.clone(),
            mode: metadata.mode,
            video: None,
            audio: vec![AudioStreamInfo {
                sample_rate: metadata.sample_rate,
                channels: metadata.channels,
                sample_format: AudioSampleFormat::F32,
            }],
            text: Vec::new(),
        };
        let selected_audio_stream = options.audio_stream_index.unwrap_or(0);
        let mut command = Command::new("ffmpeg");
        command
            .args(build_audio_ffmpeg_args(
                &input,
                &options,
                selected_audio_stream,
            ))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = command.spawn().map_err(|err| FfmpegError::StartFailed {
            input: input.clone(),
            message: err.to_string(),
        })?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| FfmpegError::StartFailed {
                input: input.clone(),
                message: "ffmpeg stdout pipe was not available".to_string(),
            })?;
        Ok(Self {
            metadata,
            source_info,
            child,
            stdout,
            next_sample_index: 0,
            samples_per_chunk: options.samples_per_chunk.max(1),
        })
    }

    /// Returns metadata.
    pub fn metadata(&self) -> &AudioMetadata {
        &self.metadata
    }

    /// Returns source info.
    pub fn source_info(&self) -> &MediaSourceInfo {
        &self.source_info
    }

    fn read_next_audio_frame(&mut self) -> Result<Option<OwnedAudioFrame>> {
        let bytes_per_sample = std::mem::size_of::<f32>();
        let bytes_per_sample_frame = self.metadata.channels as usize * bytes_per_sample;
        let target_bytes = self.samples_per_chunk * bytes_per_sample_frame;
        let mut bytes = vec![0_u8; target_bytes];
        let mut offset = 0;
        while offset < target_bytes {
            match self.stdout.read(&mut bytes[offset..]) {
                Ok(0) if offset == 0 => return Ok(None),
                Ok(0) => break,
                Ok(read) => offset += read,
                Err(err) if err.kind() == std::io::ErrorKind::Interrupted => {}
                Err(err) => return Err(DetectError::Io(err)),
            }
        }
        bytes.truncate(offset - (offset % bytes_per_sample_frame));
        if bytes.is_empty() {
            return Ok(None);
        }
        let samples = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>();
        let timestamp = Timestamp::new(
            self.next_sample_index as i64,
            Timebase::new(1, self.metadata.sample_rate as i32),
        );
        let samples_per_channel = samples.len() / self.metadata.channels as usize;
        self.next_sample_index += samples_per_channel as u64;
        OwnedAudioFrame::new(
            timestamp,
            self.metadata.sample_rate,
            self.metadata.channels,
            AudioBuffer::F32(samples),
        )
        .map(Some)
    }
}

fn build_audio_ffmpeg_args(
    input: &str,
    options: &FfmpegAudioSourceOptions,
    selected_audio_stream: usize,
) -> Vec<String> {
    let mut args = vec!["-v".to_string(), "error".to_string()];
    if options.realtime {
        args.extend(
            ["-fflags", "nobuffer", "-flags", "low_delay"]
                .into_iter()
                .map(str::to_owned),
        );
    }
    args.extend(options.extra_input_args.iter().cloned());
    args.extend([
        "-i".to_string(),
        input.to_string(),
        "-map".to_string(),
        format!("0:a:{selected_audio_stream}"),
        "-vn".to_string(),
        "-f".to_string(),
        "f32le".to_string(),
        "-acodec".to_string(),
        "pcm_f32le".to_string(),
        "pipe:1".to_string(),
    ]);
    args
}

impl Drop for FfmpegAudioSource {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl AudioFrameSource for FfmpegAudioSource {
    fn source_info(&self) -> &MediaSourceInfo {
        &self.source_info
    }

    fn next_audio_frame(&mut self) -> Result<Option<OwnedAudioFrame>> {
        self.read_next_audio_frame()
    }
}

impl MediaSource for FfmpegAudioSource {
    fn source_info(&self) -> &MediaSourceInfo {
        &self.source_info
    }

    fn next_sample(&mut self) -> Result<Option<MediaSample>> {
        self.read_next_audio_frame()
            .map(|frame| frame.map(MediaSample::Audio))
    }
}

/// Returns probe.
pub fn probe(path: impl AsRef<Path>) -> std::result::Result<VideoMetadata, FfmpegError> {
    let path = path.as_ref().to_path_buf();
    probe_path_with_mode(path, SourceMode::Recorded)
}

/// Returns probe using explicit runtime options.
pub fn probe_with_runtime(
    path: impl AsRef<Path>,
    runtime: &FfmpegRuntimeOptions,
) -> std::result::Result<VideoMetadata, FfmpegError> {
    let path = path.as_ref().to_path_buf();
    probe_path_with_runtime(path, SourceMode::Recorded, runtime)
}

/// Returns probe input.
pub fn probe_input(input: impl AsRef<str>) -> std::result::Result<VideoMetadata, FfmpegError> {
    probe_input_with_mode(input.as_ref(), None, SourceMode::Recorded)
}

/// Returns probe audio.
pub fn probe_audio(path: impl AsRef<Path>) -> std::result::Result<AudioMetadata, FfmpegError> {
    let path = path.as_ref().to_path_buf();
    probe_audio_path_with_mode(path, SourceMode::Recorded)
}

/// Returns audio probe using explicit runtime options.
pub fn probe_audio_with_runtime(
    path: impl AsRef<Path>,
    runtime: &FfmpegRuntimeOptions,
) -> std::result::Result<AudioMetadata, FfmpegError> {
    let path = path.as_ref().to_path_buf();
    probe_audio_path_with_runtime(path, SourceMode::Recorded, runtime)
}

/// Returns probe audio input.
pub fn probe_audio_input(
    input: impl AsRef<str>,
) -> std::result::Result<AudioMetadata, FfmpegError> {
    probe_audio_input_with_mode(input.as_ref(), None, SourceMode::Recorded)
}

/// Probes and returns a typed inventory of every stream in a media file.
pub fn probe_streams(
    path: impl AsRef<Path>,
) -> std::result::Result<MediaStreamInventory, FfmpegError> {
    let path = path.as_ref();
    probe_streams_input(path.to_string_lossy())
}

/// Probes and returns a typed inventory of every stream in a media input.
pub fn probe_streams_input(
    input: impl AsRef<str>,
) -> std::result::Result<MediaStreamInventory, FfmpegError> {
    let input = input.as_ref();
    let output = Command::new("ffprobe")
        .arg("-v")
        .arg("error")
        .arg("-show_entries")
        .arg("stream=index,codec_type,codec_name,channels,sample_rate:stream_tags=language:stream_disposition=default")
        .arg("-of")
        .arg("json")
        .arg(input)
        .output()
        .map_err(|err| FfmpegError::ProbeFailed {
            input: input.to_string(),
            message: err.to_string(),
        })?;
    if !output.status.success() {
        return Err(FfmpegError::ProbeFailed {
            input: input.to_string(),
            message: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }
    parse_stream_inventory(&String::from_utf8_lossy(&output.stdout))
}

fn probe_path_with_mode(
    path: impl AsRef<Path>,
    mode: SourceMode,
) -> std::result::Result<VideoMetadata, FfmpegError> {
    probe_path_with_runtime(path, mode, &FfmpegRuntimeOptions::command())
}

fn probe_path_with_runtime(
    path: impl AsRef<Path>,
    mode: SourceMode,
    runtime: &FfmpegRuntimeOptions,
) -> std::result::Result<VideoMetadata, FfmpegError> {
    let path = path.as_ref().to_path_buf();
    let input = path.to_string_lossy().into_owned();
    probe_input_with_runtime(&input, Some(path), mode, runtime)
}

fn probe_audio_path_with_mode(
    path: impl AsRef<Path>,
    mode: SourceMode,
) -> std::result::Result<AudioMetadata, FfmpegError> {
    probe_audio_path_with_runtime(path, mode, &FfmpegRuntimeOptions::command())
}

fn probe_audio_path_with_runtime(
    path: impl AsRef<Path>,
    mode: SourceMode,
    runtime: &FfmpegRuntimeOptions,
) -> std::result::Result<AudioMetadata, FfmpegError> {
    let path = path.as_ref().to_path_buf();
    let input = path.to_string_lossy().into_owned();
    probe_audio_input_with_runtime(&input, Some(path), mode, runtime)
}

fn probe_audio_input_with_runtime(
    input: &str,
    path: Option<PathBuf>,
    mode: SourceMode,
    runtime: &FfmpegRuntimeOptions,
) -> std::result::Result<AudioMetadata, FfmpegError> {
    probe_audio_input_with_runtime_and_ordinal(input, path, mode, runtime, 0)
}

fn probe_audio_input_with_runtime_and_ordinal(
    input: &str,
    path: Option<PathBuf>,
    mode: SourceMode,
    runtime: &FfmpegRuntimeOptions,
    audio_stream_ordinal: usize,
) -> std::result::Result<AudioMetadata, FfmpegError> {
    match runtime.backend {
        FfmpegRuntimeBackend::Command => {
            probe_audio_input_with_mode_and_ordinal(input, path, mode, audio_stream_ordinal)
        }
        FfmpegRuntimeBackend::Native => native_probe_audio_input(input, path, mode),
    }
}

fn probe_selected_audio_input(
    input: &str,
    path: Option<PathBuf>,
    mode: SourceMode,
    runtime: &FfmpegRuntimeOptions,
    audio_stream_index: Option<usize>,
) -> std::result::Result<AudioMetadata, FfmpegError> {
    let Some(audio_stream_index) = audio_stream_index else {
        return probe_audio_input_with_runtime(input, path, mode, runtime);
    };
    if matches!(runtime.backend, FfmpegRuntimeBackend::Native) {
        return native_probe_audio_input(input, path, mode);
    }
    let inventory = probe_streams_input(input)?;
    validate_audio_stream_selection(
        &inventory,
        AudioStreamSelection::AudioOrdinal(audio_stream_index),
    )?;
    probe_audio_input_with_runtime_and_ordinal(input, path, mode, runtime, audio_stream_index)
}

fn probe_input_with_runtime(
    input: &str,
    path: Option<PathBuf>,
    mode: SourceMode,
    runtime: &FfmpegRuntimeOptions,
) -> std::result::Result<VideoMetadata, FfmpegError> {
    match runtime.backend {
        FfmpegRuntimeBackend::Command => probe_input_with_mode(input, path, mode),
        FfmpegRuntimeBackend::Native => native_probe_input(input, path, mode),
    }
}

fn probe_audio_input_with_mode(
    input: &str,
    path: Option<PathBuf>,
    mode: SourceMode,
) -> std::result::Result<AudioMetadata, FfmpegError> {
    probe_audio_input_with_mode_and_ordinal(input, path, mode, 0)
}

fn probe_audio_input_with_mode_and_ordinal(
    input: &str,
    path: Option<PathBuf>,
    mode: SourceMode,
    audio_stream_ordinal: usize,
) -> std::result::Result<AudioMetadata, FfmpegError> {
    let output = Command::new("ffprobe")
        .arg("-v")
        .arg("error")
        .arg("-select_streams")
        .arg(format!("a:{audio_stream_ordinal}"))
        .arg("-show_entries")
        .arg("stream=sample_rate,channels,duration")
        .arg("-of")
        .arg("default=noprint_wrappers=1:nokey=1")
        .arg(input)
        .output()
        .map_err(|err| FfmpegError::ProbeFailed {
            input: input.to_string(),
            message: err.to_string(),
        })?;
    if !output.status.success() {
        return Err(FfmpegError::ProbeFailed {
            input: input.to_string(),
            message: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let mut lines = text.lines();
    let sample_rate = parse_u32(lines.next(), "sample_rate")?;
    let channels = parse_u16(lines.next(), "channels")?;
    if sample_rate == 0 || channels == 0 {
        return Err(FfmpegError::InvalidMetadata(
            "audio sample_rate and channels must be positive".to_string(),
        ));
    }
    let duration_seconds = lines.next().and_then(|value| value.parse::<f64>().ok());
    Ok(AudioMetadata {
        input: input.to_string(),
        path,
        mode,
        sample_rate,
        channels,
        duration_seconds,
    })
}

fn probe_input_with_mode(
    input: &str,
    path: Option<PathBuf>,
    mode: SourceMode,
) -> std::result::Result<VideoMetadata, FfmpegError> {
    let output = Command::new("ffprobe")
        .arg("-v")
        .arg("error")
        .arg("-select_streams")
        .arg("v:0")
        .arg("-show_entries")
        .arg("stream=width,height,avg_frame_rate,duration")
        .arg("-of")
        .arg("default=noprint_wrappers=1:nokey=1")
        .arg(input)
        .output()
        .map_err(|err| FfmpegError::ProbeFailed {
            input: input.to_string(),
            message: err.to_string(),
        })?;
    if !output.status.success() {
        return Err(FfmpegError::ProbeFailed {
            input: input.to_string(),
            message: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let mut lines = text.lines();
    let width = parse_u32(lines.next(), "width")?;
    let height = parse_u32(lines.next(), "height")?;
    let frame_rate = parse_rate(lines.next().unwrap_or_default())?;
    let duration_seconds = lines.next().and_then(|value| value.parse::<f64>().ok());
    Ok(VideoMetadata {
        input: input.to_string(),
        path,
        mode,
        width,
        height,
        frame_rate,
        duration_seconds,
    })
}

/// Returns whether is FFmpeg available.
pub fn is_ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// Returns whether is ffprobe available.
pub fn is_ffprobe_available() -> bool {
    Command::new("ffprobe")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// Returns extract wav.
pub fn extract_wav(
    media_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    sample_rate: u32,
) -> Result<PathBuf> {
    extract_wav_with_runtime(
        media_path,
        output_path,
        sample_rate,
        &FfmpegRuntimeOptions::command(),
    )
}

/// Returns extract wav using explicit runtime options.
pub fn extract_wav_with_runtime(
    media_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    sample_rate: u32,
    runtime: &FfmpegRuntimeOptions,
) -> Result<PathBuf> {
    reject_native_runtime(runtime)?;
    if sample_rate == 0 {
        return Err(DetectError::InvalidArgument(
            "sample_rate must be positive".to_string(),
        ));
    }
    let media_path = media_path.as_ref();
    let output_path = output_path.as_ref();
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let status = Command::new("ffmpeg")
        .arg("-y")
        .arg("-v")
        .arg("error")
        .arg("-i")
        .arg(media_path)
        .arg("-vn")
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg(sample_rate.to_string())
        .arg("-f")
        .arg("wav")
        .arg(output_path)
        .status()?;
    if !status.success() {
        return Err(DetectError::Source(format!(
            "ffmpeg failed to extract {sample_rate} Hz WAV audio"
        )));
    }
    Ok(output_path.to_path_buf())
}

fn reject_native_runtime(runtime: &FfmpegRuntimeOptions) -> Result<()> {
    if matches!(runtime.backend, FfmpegRuntimeBackend::Native) {
        return Err(DetectError::InvalidArgument(
            "ffmpeg-native runtime is available for probing in this build; decode/extract still uses the command backend"
                .to_string(),
        ));
    }
    Ok(())
}

fn native_probe_input(
    input: &str,
    _path: Option<PathBuf>,
    _mode: SourceMode,
) -> std::result::Result<VideoMetadata, FfmpegError> {
    Err(FfmpegError::ProbeFailed {
        input: input.to_string(),
        message: "ffmpeg-native feature is not enabled".to_string(),
    })
}

fn native_probe_audio_input(
    input: &str,
    _path: Option<PathBuf>,
    _mode: SourceMode,
) -> std::result::Result<AudioMetadata, FfmpegError> {
    Err(FfmpegError::ProbeFailed {
        input: input.to_string(),
        message: "ffmpeg-native feature is not enabled".to_string(),
    })
}

#[cfg(feature = "test-utils")]
/// Writes two scene test video.
pub fn write_two_scene_test_video(path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let status = Command::new("ffmpeg")
        .arg("-y")
        .arg("-v")
        .arg("error")
        .arg("-f")
        .arg("lavfi")
        .arg("-i")
        .arg("color=c=black:s=64x64:d=0.5:r=10")
        .arg("-f")
        .arg("lavfi")
        .arg("-i")
        .arg("color=c=white:s=64x64:d=0.5:r=10")
        .arg("-filter_complex")
        .arg("[0:v][1:v]concat=n=2:v=1:a=0")
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg(path)
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(DetectError::Source(
            "ffmpeg failed to generate test video".to_string(),
        ))
    }
}

#[cfg(feature = "test-utils")]
/// Writes vertical test video.
pub fn write_vertical_test_video(path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let status = Command::new("ffmpeg")
        .arg("-y")
        .arg("-v")
        .arg("error")
        .arg("-f")
        .arg("lavfi")
        .arg("-i")
        .arg("testsrc=size=72x128:duration=0.4:rate=10")
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg(path)
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(DetectError::Source(
            "ffmpeg failed to generate vertical test video".to_string(),
        ))
    }
}

#[cfg(feature = "test-utils")]
/// Writes test audio.
pub fn write_test_audio(path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let status = Command::new("ffmpeg")
        .arg("-y")
        .arg("-v")
        .arg("error")
        .arg("-f")
        .arg("lavfi")
        .arg("-i")
        .arg("sine=frequency=1000:duration=0.1:sample_rate=48000")
        .arg("-ac")
        .arg("1")
        .arg(path)
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(DetectError::Source(
            "ffmpeg failed to generate test audio".to_string(),
        ))
    }
}

#[cfg(feature = "test-utils")]
/// Writes a small Matroska file with video and two distinguishable audio streams.
pub fn write_two_audio_stream_test_media(path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let status = Command::new("ffmpeg")
        .arg("-y")
        .arg("-v")
        .arg("error")
        .arg("-f")
        .arg("lavfi")
        .arg("-i")
        .arg("color=c=black:s=16x16:d=0.2:r=10")
        .arg("-f")
        .arg("lavfi")
        .arg("-i")
        .arg("sine=frequency=440:duration=0.2:sample_rate=48000")
        .arg("-f")
        .arg("lavfi")
        .arg("-i")
        .arg("sine=frequency=880:duration=0.2:sample_rate=24000")
        .arg("-map")
        .arg("0:v:0")
        .arg("-map")
        .arg("1:a:0")
        .arg("-map")
        .arg("2:a:0")
        .arg("-c:v")
        .arg("ffv1")
        .arg("-c:a")
        .arg("pcm_s16le")
        .arg("-metadata:s:a:0")
        .arg("language=eng")
        .arg("-metadata:s:a:1")
        .arg("language=deu")
        .arg("-disposition:a:0")
        .arg("default")
        .arg("-disposition:a:1")
        .arg("0")
        .arg(path)
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(DetectError::Source(
            "ffmpeg failed to generate two-audio-stream test media".to_string(),
        ))
    }
}

fn parse_u32(value: Option<&str>, name: &str) -> std::result::Result<u32, FfmpegError> {
    value
        .ok_or_else(|| FfmpegError::InvalidMetadata(format!("missing {name}")))?
        .parse::<u32>()
        .map_err(|err| FfmpegError::InvalidMetadata(format!("invalid {name}: {err}")))
}

fn parse_u16(value: Option<&str>, name: &str) -> std::result::Result<u16, FfmpegError> {
    value
        .ok_or_else(|| FfmpegError::InvalidMetadata(format!("missing {name}")))?
        .parse::<u16>()
        .map_err(|err| FfmpegError::InvalidMetadata(format!("invalid {name}: {err}")))
}

fn parse_rate(value: &str) -> std::result::Result<Rational64, FfmpegError> {
    let Some((num, den)) = value.split_once('/') else {
        return Err(FfmpegError::InvalidMetadata(format!(
            "invalid frame rate `{value}`"
        )));
    };
    let num = num
        .parse::<i64>()
        .map_err(|err| FfmpegError::InvalidMetadata(format!("invalid frame rate: {err}")))?;
    let den = den
        .parse::<i64>()
        .map_err(|err| FfmpegError::InvalidMetadata(format!("invalid frame rate: {err}")))?;
    if num <= 0 || den <= 0 {
        return Err(FfmpegError::InvalidMetadata(
            "frame rate must be positive".to_string(),
        ));
    }
    Ok(Rational64::new(num, den))
}

fn parse_stream_inventory(json: &str) -> std::result::Result<MediaStreamInventory, FfmpegError> {
    let value: serde_json::Value = serde_json::from_str(json)
        .map_err(|err| FfmpegError::InvalidMetadata(format!("invalid ffprobe JSON: {err}")))?;
    let streams = value
        .get("streams")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| FfmpegError::InvalidMetadata("missing streams array".to_string()))?;
    let mut audio_stream_ordinal = 0_usize;
    let streams = streams
        .iter()
        .map(|stream| {
            let index = stream
                .get("index")
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| {
                    FfmpegError::InvalidMetadata("missing or invalid stream index".to_string())
                })?;
            let media_type = match stream
                .get("codec_type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown")
            {
                "video" => MediaType::Video,
                "audio" => MediaType::Audio,
                "subtitle" => MediaType::Subtitle,
                "data" => MediaType::Data,
                "attachment" => MediaType::Attachment,
                other => MediaType::Unknown(other.to_string()),
            };
            let ordinal = if media_type == MediaType::Audio {
                let ordinal = audio_stream_ordinal;
                audio_stream_ordinal += 1;
                Some(ordinal)
            } else {
                None
            };
            let sample_rate = stream
                .get("sample_rate")
                .and_then(serde_json::Value::as_str)
                .and_then(|value| value.parse::<u32>().ok());
            let channels = stream
                .get("channels")
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| u16::try_from(value).ok());
            let language = stream
                .get("tags")
                .and_then(|tags| tags.get("language"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned);
            let default_disposition = stream
                .get("disposition")
                .and_then(|disposition| disposition.get("default"))
                .and_then(serde_json::Value::as_u64)
                .map(|value| value != 0);
            Ok(MediaStream {
                index,
                media_type,
                audio_stream_ordinal: ordinal,
                codec: stream
                    .get("codec_name")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned),
                channels,
                sample_rate,
                language,
                default_disposition,
            })
        })
        .collect::<std::result::Result<Vec<_>, FfmpegError>>()?;
    Ok(MediaStreamInventory { streams })
}

/// Returns ensure parent dir.
pub fn ensure_parent_dir(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

/// Returns touch text.
pub fn touch_text(path: &Path, text: &str) -> std::io::Result<()> {
    ensure_parent_dir(path)?;
    let mut file = std::fs::File::create(path)?;
    file.write_all(text.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_option_builders_select_expected_backends() {
        assert!(matches!(
            FfmpegRuntimeOptions::default().backend,
            FfmpegRuntimeBackend::Command
        ));
        assert!(matches!(
            FfmpegSourceOptions::live()
                .extra_input_arg("-nostdin")
                .extra_output_arg("-vf")
                .extra_output_arg("scale=320:-2")
                .resize_width(320)
                .output_dimensions(320, 180)
                .runtime(FfmpegRuntimeOptions::native())
                .runtime
                .backend,
            FfmpegRuntimeBackend::Native
        ));
        assert_eq!(
            FfmpegSourceOptions::recorded()
                .extra_output_arg("-vf")
                .extra_output_arg("scale=320:-2")
                .extra_output_args,
            vec!["-vf".to_string(), "scale=320:-2".to_string()]
        );
        assert_eq!(
            FfmpegSourceOptions::recorded()
                .resize_width(320)
                .resize_width,
            Some(320)
        );
        assert_eq!(
            FfmpegSourceOptions::recorded()
                .pixel_format(PixelFormat::Bgr24)
                .pixel_format,
            PixelFormat::Bgr24
        );
        assert!(matches!(
            FfmpegAudioSourceOptions::recorded()
                .samples_per_chunk(0)
                .runtime(FfmpegRuntimeOptions::native())
                .runtime
                .backend,
            FfmpegRuntimeBackend::Native
        ));
        assert_eq!(
            FfmpegAudioSourceOptions::recorded()
                .samples_per_chunk(0)
                .samples_per_chunk,
            1
        );
        assert_eq!(
            FfmpegAudioSourceOptions::recorded()
                .audio_stream_index(1)
                .audio_stream_index,
            Some(1)
        );
    }

    #[test]
    fn stream_inventory_parses_all_media_and_assigns_audio_ordinals() {
        let inventory = parse_stream_inventory(
            r#"{"streams":[
                {"index":0,"codec_name":"h264","codec_type":"video","disposition":{"default":1}},
                {"index":1,"codec_name":"aac","codec_type":"audio","sample_rate":"48000","channels":2,"tags":{"language":"eng"},"disposition":{"default":1}},
                {"index":2,"codec_name":"subrip","codec_type":"subtitle"},
                {"index":3,"codec_name":"opus","codec_type":"audio","sample_rate":"24000","channels":1,"tags":{"language":"deu"},"disposition":{"default":0}}
            ]}"#,
        )
        .unwrap();

        assert_eq!(inventory.streams.len(), 4);
        assert_eq!(inventory.streams[0].media_type, MediaType::Video);
        assert_eq!(inventory.streams[0].audio_stream_ordinal, None);
        assert_eq!(inventory.streams[1].audio_stream_ordinal, Some(0));
        assert_eq!(inventory.streams[1].sample_rate, Some(48_000));
        assert_eq!(inventory.streams[1].channels, Some(2));
        assert_eq!(inventory.streams[1].language.as_deref(), Some("eng"));
        assert_eq!(inventory.streams[1].default_disposition, Some(true));
        assert_eq!(inventory.streams[2].media_type, MediaType::Subtitle);
        assert_eq!(inventory.streams[3].audio_stream_ordinal, Some(1));
        assert_eq!(inventory.streams[3].default_disposition, Some(false));
    }

    #[test]
    fn audio_stream_selection_resolves_audio_ordinals() {
        let inventory = parse_stream_inventory(
            r#"{"streams":[
                {"index":0,"codec_type":"video"},
                {"index":1,"codec_type":"audio","sample_rate":"48000","channels":1},
                {"index":2,"codec_type":"audio","sample_rate":"24000","channels":2}
            ]}"#,
        )
        .unwrap();

        let selected =
            validate_audio_stream_selection(&inventory, AudioStreamSelection::AudioOrdinal(1))
                .unwrap();

        assert_eq!(selected.index, 2);
        assert_eq!(selected.audio_stream_ordinal, Some(1));
    }

    #[test]
    fn invalid_audio_stream_selections_are_typed_and_retain_inventory() {
        let inventory = parse_stream_inventory(
            r#"{"streams":[
                {"index":0,"codec_type":"video"},
                {"index":1,"codec_type":"audio","sample_rate":"48000","channels":1}
            ]}"#,
        )
        .unwrap();

        for (selection, expected_reason) in [
            (
                AudioStreamSelection::AudioOrdinal(2),
                AudioStreamSelectionErrorReason::OutOfRange,
            ),
            (
                AudioStreamSelection::GlobalStreamIndex(0),
                AudioStreamSelectionErrorReason::NotAudio,
            ),
        ] {
            let error = validate_audio_stream_selection(&inventory, selection).unwrap_err();
            let FfmpegError::InvalidAudioStreamSelection {
                reason,
                available_streams,
                ..
            } = error
            else {
                panic!("expected typed selection error");
            };
            assert_eq!(reason, expected_reason);
            assert_eq!(available_streams, inventory);
        }

        let no_audio =
            parse_stream_inventory(r#"{"streams":[{"index":0,"codec_type":"video"}]}"#).unwrap();
        let error =
            validate_audio_stream_selection(&no_audio, AudioStreamSelection::AudioOrdinal(0))
                .unwrap_err();
        assert!(matches!(
            error,
            FfmpegError::InvalidAudioStreamSelection {
                reason: AudioStreamSelectionErrorReason::NoAudioStreams,
                ..
            }
        ));
    }

    #[test]
    fn audio_decode_map_is_an_output_option_after_the_input() {
        let options = FfmpegAudioSourceOptions::recorded()
            .extra_input_arg("-nostdin")
            .audio_stream_index(2);
        let args = build_audio_ffmpeg_args("input.mkv", &options, 2);

        let input_position = args.iter().position(|arg| arg == "-i").unwrap();
        let input_path_position = args.iter().position(|arg| arg == "input.mkv").unwrap();
        let map_position = args.iter().position(|arg| arg == "-map").unwrap();
        assert!(args.iter().position(|arg| arg == "-nostdin").unwrap() < input_position);
        assert_eq!(input_path_position, input_position + 1);
        assert!(map_position > input_path_position);
        assert_eq!(args[map_position + 1], "0:a:2");
    }

    #[test]
    fn command_only_operations_reject_native_runtime_before_spawning() {
        let native_extract = extract_wav_with_runtime(
            "missing-input.mp4",
            "missing-output.wav",
            16_000,
            &FfmpegRuntimeOptions::native(),
        )
        .unwrap_err();
        assert!(matches!(native_extract, DetectError::InvalidArgument(_)));

        let zero_sample_rate = extract_wav_with_runtime(
            "missing-input.mp4",
            "missing-output.wav",
            0,
            &FfmpegRuntimeOptions::command(),
        )
        .unwrap_err();
        assert!(matches!(zero_sample_rate, DetectError::InvalidArgument(_)));
    }

    #[test]
    fn native_probe_runtime_reports_unavailable_backend_without_ffprobe() {
        let video = probe_with_runtime("missing-input.mp4", &FfmpegRuntimeOptions::native())
            .unwrap_err()
            .to_string();
        assert!(video.contains("ffmpeg-native feature is not enabled"));

        let audio = probe_audio_with_runtime("missing-input.wav", &FfmpegRuntimeOptions::native())
            .unwrap_err()
            .to_string();
        assert!(audio.contains("ffmpeg-native feature is not enabled"));
    }

    #[test]
    fn resize_height_matches_ffmpeg_scale_negative_two_rounding() {
        assert_eq!(scaled_even_height(720, 1280, 320), 568);
        assert_eq!(scaled_even_height(1920, 1080, 320), 180);
        assert_eq!(scaled_even_height(9, 16, 4), 6);
    }

    #[test]
    #[cfg(feature = "ffmpeg-tests")]
    fn decodes_generated_two_scene_video() {
        use video_analysis_core::VideoSource;

        if !(is_ffmpeg_available() && is_ffprobe_available()) {
            eprintln!("skipping FFmpeg test because ffmpeg/ffprobe is unavailable");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("two-scenes.mp4");
        write_two_scene_test_video(&path).unwrap();
        let mut source = FfmpegVideoSource::open(&path).unwrap();
        assert_eq!(source.metadata().width, 64);
        assert_eq!(source.metadata().height, 64);
        let mut count = 0;
        while source.next_frame().unwrap().is_some() {
            count += 1;
        }
        assert!(
            count >= 8,
            "expected at least 8 decoded frames, got {count}"
        );
    }

    #[test]
    #[cfg(feature = "ffmpeg-tests")]
    fn decodes_generated_vertical_video_with_resize() {
        use video_analysis_core::VideoSource;

        if !(is_ffmpeg_available() && is_ffprobe_available()) {
            eprintln!("skipping FFmpeg test because ffmpeg/ffprobe is unavailable");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vertical.mp4");
        write_vertical_test_video(&path).unwrap();
        let mut source = FfmpegVideoSource::open_path_with_options(
            &path,
            FfmpegSourceOptions::recorded().resize_width(32),
        )
        .unwrap();
        let first = source.next_frame().unwrap().unwrap();
        assert_eq!(first.width, 32);
        assert_eq!(first.height, 56);
        assert_eq!(first.stride, 96);
        assert_eq!(first.data.len(), 32 * 56 * 3);
        let mut count = 1;
        while source.next_frame().unwrap().is_some() {
            count += 1;
        }
        assert!(
            count >= 4,
            "expected at least 4 decoded frames, got {count}"
        );
    }

    #[test]
    #[cfg(feature = "ffmpeg-tests")]
    fn probes_generated_two_audio_stream_inventory() {
        if !(is_ffmpeg_available() && is_ffprobe_available()) {
            eprintln!("skipping FFmpeg test because ffmpeg/ffprobe is unavailable");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("two-audio-streams.mkv");
        write_two_audio_stream_test_media(&path).unwrap();

        let inventory = probe_streams(&path).unwrap();
        assert_eq!(inventory.streams.len(), 3);
        assert_eq!(inventory.streams[0].media_type, MediaType::Video);
        assert_eq!(inventory.streams[1].audio_stream_ordinal, Some(0));
        assert_eq!(inventory.streams[1].sample_rate, Some(48_000));
        assert_eq!(inventory.streams[1].language.as_deref(), Some("eng"));
        assert_eq!(inventory.streams[1].default_disposition, Some(true));
        assert_eq!(inventory.streams[2].audio_stream_ordinal, Some(1));
        assert_eq!(inventory.streams[2].sample_rate, Some(24_000));
        assert_eq!(inventory.streams[2].language.as_deref(), Some("deu"));
        assert_eq!(inventory.streams[2].default_disposition, Some(false));
    }

    #[test]
    #[cfg(feature = "ffmpeg-tests")]
    fn decodes_default_and_each_explicit_audio_stream() {
        use video_analysis_ingest::AudioFrameSource;

        if !(is_ffmpeg_available() && is_ffprobe_available()) {
            eprintln!("skipping FFmpeg test because ffmpeg/ffprobe is unavailable");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("two-audio-streams.mkv");
        write_two_audio_stream_test_media(&path).unwrap();

        for (selection, expected_sample_rate) in
            [(None, 48_000), (Some(0), 48_000), (Some(1), 24_000)]
        {
            let mut options = FfmpegAudioSourceOptions::recorded().samples_per_chunk(256);
            if let Some(selection) = selection {
                options = options.audio_stream_index(selection);
            }
            let mut source =
                FfmpegAudioSource::open_path_with_options_checked(&path, options).unwrap();
            assert_eq!(source.metadata().sample_rate, expected_sample_rate);
            let frame = source.next_audio_frame().unwrap().unwrap();
            assert_eq!(frame.sample_rate, expected_sample_rate);
            assert!(frame.samples_per_channel() > 0);
        }
    }

    #[test]
    #[cfg(feature = "ffmpeg-tests")]
    fn explicit_invalid_audio_ordinal_fails_with_inventory_before_decode() {
        if !(is_ffmpeg_available() && is_ffprobe_available()) {
            eprintln!("skipping FFmpeg test because ffmpeg/ffprobe is unavailable");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("two-audio-streams.mkv");
        write_two_audio_stream_test_media(&path).unwrap();

        let error = FfmpegAudioSource::open_path_with_options_checked(
            &path,
            FfmpegAudioSourceOptions::recorded().audio_stream_index(2),
        )
        .err()
        .expect("selection should fail before a source is spawned");
        let FfmpegError::InvalidAudioStreamSelection {
            selection,
            reason,
            available_streams,
        } = error
        else {
            panic!("expected typed selection error");
        };
        assert_eq!(selection, AudioStreamSelection::AudioOrdinal(2));
        assert_eq!(reason, AudioStreamSelectionErrorReason::OutOfRange);
        assert_eq!(available_streams.streams.len(), 3);
    }

    #[test]
    #[cfg(feature = "ffmpeg-tests")]
    fn explicit_audio_selection_reports_when_audio_is_absent() {
        if !(is_ffmpeg_available() && is_ffprobe_available()) {
            eprintln!("skipping FFmpeg test because ffmpeg/ffprobe is unavailable");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("video-only.mp4");
        write_two_scene_test_video(&path).unwrap();

        let error = FfmpegAudioSource::open_path_with_options_checked(
            &path,
            FfmpegAudioSourceOptions::recorded().audio_stream_index(0),
        )
        .err()
        .expect("selection should fail before a source is spawned");
        assert!(matches!(
            error,
            FfmpegError::InvalidAudioStreamSelection {
                reason: AudioStreamSelectionErrorReason::NoAudioStreams,
                ..
            }
        ));
    }

    #[test]
    #[cfg(feature = "ffmpeg-tests")]
    fn decodes_generated_audio() {
        use video_analysis_ingest::AudioFrameSource;

        if !(is_ffmpeg_available() && is_ffprobe_available()) {
            eprintln!("skipping FFmpeg test because ffmpeg/ffprobe is unavailable");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tone.wav");
        write_test_audio(&path).unwrap();
        let mut source = FfmpegAudioSource::open_path_with_options(
            &path,
            FfmpegAudioSourceOptions::recorded().samples_per_chunk(512),
        )
        .unwrap();
        assert_eq!(source.metadata().sample_rate, 48_000);
        assert_eq!(source.metadata().channels, 1);
        let mut count = 0;
        while let Some(frame) = source.next_audio_frame().unwrap() {
            assert_eq!(frame.sample_rate, 48_000);
            count += 1;
        }
        assert!(count > 0, "expected decoded audio frames");
    }
}
