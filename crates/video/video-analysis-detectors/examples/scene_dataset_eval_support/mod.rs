use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use video_analysis_core::{PixelFormat, SceneDetector};
use video_analysis_detectors::{
    AdaptiveDetector, ContentDetector, FlashFilterMode, HashDetector, HistogramDetector,
    ThresholdDetector,
};
use video_analysis_ffmpeg::FfmpegSourceOptions;

#[derive(Debug, Clone)]
pub struct Args {
    pub dataset: String,
    pub root: PathBuf,
    pub detector: String,
    pub preset: Option<String>,
    pub output: Option<PathBuf>,
    pub video_ids: Vec<String>,
    pub limit: Option<usize>,
    pub max_runtime: Option<Duration>,
    pub progress: bool,
    pub resume: bool,
    pub resize_width: Option<u32>,
    pub pixel_format: PixelFormat,
    pub config: DetectorConfig,
}

pub const BBC_CONTENT_TUNED_PRESET: &str = "bbc-content-tuned";
pub const BBC_CONTENT_TUNED_ADAPTIVE_THRESHOLD: f32 = 3.0;
pub const BBC_CONTENT_TUNED_ADAPTIVE_WINDOW_WIDTH: usize = 2;
pub const BBC_CONTENT_TUNED_ADAPTIVE_MIN_CONTENT_VAL: f32 = 15.0;
pub const BBC_CONTENT_TUNED_MIN_SCENE_LEN: u64 = 15;
pub const BBC_CONTENT_TUNED_POST_FILTER_WINDOW: u64 = 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterMode {
    Merge,
    Suppress,
}

impl FilterMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Merge => "merge",
            Self::Suppress => "suppress",
        }
    }

    fn into_flash_filter_mode(self) -> FlashFilterMode {
        match self {
            Self::Merge => FlashFilterMode::Merge,
            Self::Suppress => FlashFilterMode::Suppress,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DetectorConfig {
    pub content_threshold: f32,
    pub min_scene_len: u64,
    pub filter_mode: FilterMode,
    pub adaptive_threshold: f32,
    pub adaptive_window_width: usize,
    pub adaptive_min_content_val: f32,
    pub post_filter_window: u64,
    supplied: SuppliedDetectorFlags,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            content_threshold: 27.0,
            min_scene_len: 15,
            filter_mode: FilterMode::Merge,
            adaptive_threshold: 3.0,
            adaptive_window_width: 2,
            adaptive_min_content_val: 15.0,
            post_filter_window: 0,
            supplied: SuppliedDetectorFlags::default(),
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct SuppliedDetectorFlags {
    content_threshold: bool,
    filter_mode: bool,
    adaptive_threshold: bool,
    adaptive_window_width: bool,
    adaptive_min_content_val: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvalConfiguration {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset: Option<String>,
    pub resize_width: Option<u32>,
    pub content_threshold: Option<f32>,
    pub min_scene_len: u64,
    pub filter_mode: Option<String>,
    pub adaptive_threshold: Option<f32>,
    pub adaptive_window_width: Option<usize>,
    pub adaptive_min_content_val: Option<f32>,
    pub post_filter_window: u64,
}

impl Default for EvalConfiguration {
    fn default() -> Self {
        let args = Args {
            dataset: String::new(),
            root: PathBuf::new(),
            detector: "content".to_string(),
            preset: None,
            output: None,
            video_ids: Vec::new(),
            limit: None,
            max_runtime: None,
            progress: false,
            resume: false,
            resize_width: None,
            pixel_format: PixelFormat::Rgb24,
            config: DetectorConfig::default(),
        };
        eval_configuration(&args)
    }
}

pub fn parse_args() -> Result<Args, String> {
    parse_args_from(std::env::args().skip(1))
}

pub fn parse_args_from<I, S>(values: I) -> Result<Args, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut dataset = None;
    let mut root = None;
    let mut detector = None;
    let mut preset = None;
    let mut output = None;
    let mut video_ids = Vec::new();
    let mut limit = None;
    let mut max_runtime = None;
    let mut progress = false;
    let mut resume = false;
    let mut resize_width = None;
    let mut pixel_format = PixelFormat::Rgb24;
    let mut config = DetectorConfig::default();
    let mut args = values.into_iter().map(Into::into);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dataset" => dataset = Some(next_value(&mut args, "--dataset")?),
            "--root" => root = Some(PathBuf::from(next_value(&mut args, "--root")?)),
            "--detector" => detector = Some(next_value(&mut args, "--detector")?),
            "--preset" => {
                let value = next_value(&mut args, "--preset")?;
                apply_preset(&value, &mut detector, &mut config)?;
                preset = Some(value);
            }
            "--output" => output = Some(PathBuf::from(next_value(&mut args, "--output")?)),
            "--video-id" => push_unique(&mut video_ids, next_value(&mut args, "--video-id")?),
            "--limit" => {
                limit = Some(parse_usize(&mut args, "--limit")?);
            }
            "--max-runtime-seconds" => {
                let seconds = parse_u64(&mut args, "--max-runtime-seconds")?;
                max_runtime = Some(Duration::from_secs(seconds));
            }
            "--resize-width" => {
                let width = parse_u32(&mut args, "--resize-width")?;
                if width == 0 {
                    return Err("--resize-width must be greater than 0".to_string());
                }
                resize_width = Some(width);
            }
            "--pixel-format" => {
                pixel_format = parse_pixel_format(&mut args)?;
            }
            "--content-threshold" => {
                config.content_threshold = parse_positive_f32(&mut args, "--content-threshold")?;
                config.supplied.content_threshold = true;
            }
            "--min-scene-len" => {
                config.min_scene_len = parse_positive_u64(&mut args, "--min-scene-len")?;
            }
            "--filter-mode" => {
                config.filter_mode = parse_filter_mode(&mut args)?;
                config.supplied.filter_mode = true;
            }
            "--adaptive-threshold" => {
                config.adaptive_threshold = parse_positive_f32(&mut args, "--adaptive-threshold")?;
                config.supplied.adaptive_threshold = true;
            }
            "--adaptive-window-width" => {
                config.adaptive_window_width =
                    parse_positive_usize(&mut args, "--adaptive-window-width")?;
                config.supplied.adaptive_window_width = true;
            }
            "--adaptive-min-content-val" => {
                config.adaptive_min_content_val =
                    parse_positive_f32(&mut args, "--adaptive-min-content-val")?;
                config.supplied.adaptive_min_content_val = true;
            }
            "--post-filter-window" => {
                config.post_filter_window = parse_u64(&mut args, "--post-filter-window")?;
            }
            "--progress" => progress = true,
            "--resume" => resume = true,
            "-h" | "--help" => return Err(usage()),
            _ => return Err(format!("unknown argument `{arg}`")),
        }
    }
    let detector = detector.unwrap_or_else(|| "content".to_string());
    validate_detector_flags(&detector, &config)?;
    Ok(Args {
        dataset: dataset.ok_or_else(|| "missing --dataset".to_string())?,
        root: root.ok_or_else(|| "missing --root".to_string())?,
        detector,
        preset,
        output,
        video_ids,
        limit,
        max_runtime,
        progress,
        resume,
        resize_width,
        pixel_format,
        config,
    })
}

pub fn usage() -> String {
    "usage: scene_dataset_eval --dataset NAME --root PATH [--preset bbc-content-tuned] [--detector content|adaptive|threshold|histogram|hash] [--video-id ID ...] [--limit N] [--resize-width PIXELS] [--pixel-format rgb24|bgr24] [--content-threshold FLOAT] [--min-scene-len FRAMES] [--filter-mode merge|suppress] [--adaptive-threshold FLOAT] [--adaptive-window-width FRAMES] [--adaptive-min-content-val FLOAT] [--post-filter-window FRAMES] [--progress] [--resume] [--max-runtime-seconds SECONDS] [--output PATH]".to_string()
}

pub fn detector_from_args(args: &Args) -> video_analysis_core::Result<Box<dyn SceneDetector>> {
    let config = &args.config;
    match normalized_detector_name(&args.detector) {
        Some("content") => Ok(Box::new(
            ContentDetector::new(config.content_threshold, config.min_scene_len).filter_mode(
                config.filter_mode.into_flash_filter_mode(),
                config.min_scene_len,
            ),
        )),
        Some("adaptive") => Ok(Box::new(AdaptiveDetector::new(
            config.adaptive_threshold,
            config.min_scene_len,
            config.adaptive_window_width,
            config.adaptive_min_content_val,
        ))),
        Some("threshold") => Ok(Box::new(ThresholdDetector::new(12.0, config.min_scene_len))),
        Some("histogram") => Ok(Box::new(HistogramDetector::new(
            0.05,
            256,
            config.min_scene_len,
        ))),
        Some("hash") => Ok(Box::new(HashDetector::new(
            0.395,
            16,
            2,
            config.min_scene_len,
        ))),
        _ => Err(video_analysis_core::DetectError::InvalidArgument(format!(
            "unsupported detector `{}`",
            args.detector
        ))),
    }
}

pub fn eval_configuration(args: &Args) -> EvalConfiguration {
    let detector = normalized_detector_name(&args.detector);
    EvalConfiguration {
        preset: args.preset.clone(),
        resize_width: args.resize_width,
        content_threshold: (detector == Some("content")).then_some(args.config.content_threshold),
        min_scene_len: args.config.min_scene_len,
        filter_mode: (detector == Some("content"))
            .then(|| args.config.filter_mode.as_str().to_string()),
        adaptive_threshold: (detector == Some("adaptive"))
            .then_some(args.config.adaptive_threshold),
        adaptive_window_width: (detector == Some("adaptive"))
            .then_some(args.config.adaptive_window_width),
        adaptive_min_content_val: (detector == Some("adaptive"))
            .then_some(args.config.adaptive_min_content_val),
        post_filter_window: args.config.post_filter_window,
    }
}

fn apply_preset(
    value: &str,
    detector: &mut Option<String>,
    config: &mut DetectorConfig,
) -> Result<(), String> {
    match value {
        BBC_CONTENT_TUNED_PRESET => {
            *detector = Some("adaptive".to_string());
            config.adaptive_threshold = BBC_CONTENT_TUNED_ADAPTIVE_THRESHOLD;
            config.adaptive_window_width = BBC_CONTENT_TUNED_ADAPTIVE_WINDOW_WIDTH;
            config.adaptive_min_content_val = BBC_CONTENT_TUNED_ADAPTIVE_MIN_CONTENT_VAL;
            config.min_scene_len = BBC_CONTENT_TUNED_MIN_SCENE_LEN;
            config.post_filter_window = BBC_CONTENT_TUNED_POST_FILTER_WINDOW;
            Ok(())
        }
        _ => Err(format!(
            "unknown --preset `{value}`; expected `{BBC_CONTENT_TUNED_PRESET}`"
        )),
    }
}

pub fn ffmpeg_options_from_args(args: &Args) -> FfmpegSourceOptions {
    let mut options = FfmpegSourceOptions::recorded().pixel_format(args.pixel_format);
    if let Some(width) = args.resize_width {
        options = options.resize_width(width);
    }
    options
}

pub fn suppress_nearby_cuts(cuts: Vec<u64>, window: u64) -> Vec<u64> {
    if window == 0 {
        return cuts;
    }
    let mut kept = Vec::new();
    for cut in cuts {
        if kept
            .last()
            .is_none_or(|last| cut.saturating_sub(*last) >= window)
        {
            kept.push(cut);
        }
    }
    kept
}

fn validate_detector_flags(detector: &str, config: &DetectorConfig) -> Result<(), String> {
    match normalized_detector_name(detector) {
        Some("content") => {
            if config.supplied.adaptive_threshold
                || config.supplied.adaptive_window_width
                || config.supplied.adaptive_min_content_val
            {
                return Err("adaptive tuning flags require --detector adaptive".to_string());
            }
        }
        Some("adaptive") => {
            if config.supplied.content_threshold || config.supplied.filter_mode {
                return Err("content tuning flags require --detector content".to_string());
            }
        }
        Some("threshold" | "histogram" | "hash") => {
            if config.supplied.content_threshold
                || config.supplied.filter_mode
                || config.supplied.adaptive_threshold
                || config.supplied.adaptive_window_width
                || config.supplied.adaptive_min_content_val
            {
                return Err(format!(
                    "detector-specific tuning flags are only supported for content and adaptive detectors, not `{detector}`"
                ));
            }
        }
        Some(_) => {}
        None => {
            return Err(format!("unsupported detector `{detector}`"));
        }
    }
    Ok(())
}

fn normalized_detector_name(value: &str) -> Option<&'static str> {
    match value {
        "content" | "detect-content" => Some("content"),
        "adaptive" | "detect-adaptive" => Some("adaptive"),
        "threshold" | "detect-threshold" => Some("threshold"),
        "histogram" | "detect-hist" | "detect-histogram" => Some("histogram"),
        "hash" | "detect-hash" => Some("hash"),
        _ => None,
    }
}

fn next_value(args: &mut impl Iterator<Item = String>, name: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value for {name}"))
}

fn parse_filter_mode(args: &mut impl Iterator<Item = String>) -> Result<FilterMode, String> {
    let value = next_value(args, "--filter-mode")?;
    match value.as_str() {
        "merge" => Ok(FilterMode::Merge),
        "suppress" => Ok(FilterMode::Suppress),
        _ => Err(format!(
            "invalid --filter-mode `{value}`; expected `merge` or `suppress`"
        )),
    }
}

fn parse_pixel_format(args: &mut impl Iterator<Item = String>) -> Result<PixelFormat, String> {
    let value = next_value(args, "--pixel-format")?;
    match value.as_str() {
        "rgb24" => Ok(PixelFormat::Rgb24),
        "bgr24" => Ok(PixelFormat::Bgr24),
        _ => Err(format!(
            "invalid --pixel-format `{value}`; expected `rgb24` or `bgr24`"
        )),
    }
}

fn parse_positive_f32(args: &mut impl Iterator<Item = String>, name: &str) -> Result<f32, String> {
    let value = next_value(args, name)?;
    let parsed = value
        .parse::<f32>()
        .map_err(|error| format!("invalid {name}: {error}"))?;
    if !parsed.is_finite() || parsed <= 0.0 {
        return Err(format!("{name} must be finite and greater than 0"));
    }
    Ok(parsed)
}

fn parse_positive_u64(args: &mut impl Iterator<Item = String>, name: &str) -> Result<u64, String> {
    let parsed = parse_u64(args, name)?;
    if parsed == 0 {
        return Err(format!("{name} must be greater than 0"));
    }
    Ok(parsed)
}

fn parse_positive_usize(
    args: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<usize, String> {
    let parsed = parse_usize(args, name)?;
    if parsed == 0 {
        return Err(format!("{name} must be greater than 0"));
    }
    Ok(parsed)
}

fn parse_u64(args: &mut impl Iterator<Item = String>, name: &str) -> Result<u64, String> {
    next_value(args, name)?
        .parse::<u64>()
        .map_err(|error| format!("invalid {name}: {error}"))
}

fn parse_u32(args: &mut impl Iterator<Item = String>, name: &str) -> Result<u32, String> {
    next_value(args, name)?
        .parse::<u32>()
        .map_err(|error| format!("invalid {name}: {error}"))
}

fn parse_usize(args: &mut impl Iterator<Item = String>, name: &str) -> Result<usize, String> {
    next_value(args, name)?
        .parse::<usize>()
        .map_err(|error| format!("invalid {name}: {error}"))
}

fn push_unique(values: &mut Vec<String>, value: String) {
    if !values.iter().any(|existing| existing == &value) {
        values.push(value);
    }
}
