use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Serialize;
use video_analysis_core::OwnedVideoFrame;
use video_analysis_ffmpeg::FfmpegVideoSource;
use video_analysis_ingest::VideoFrameSource;

#[allow(dead_code)]
mod scene_dataset_eval_support;
use scene_dataset_eval_support::{
    detector_from_args, ffmpeg_options_from_args, Args as EvalArgs, DetectorConfig,
};

#[derive(Debug)]
struct Args {
    input: PathBuf,
    output: Option<PathBuf>,
    detector: String,
    resize_width: Option<u32>,
    iterations: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BenchmarkReport {
    input: String,
    detector: String,
    resize_width: Option<u32>,
    frame_count: usize,
    decode_elapsed_ms: f64,
    iterations: Vec<IterationReport>,
    avg_detector_elapsed_ms: f64,
    frames_per_second: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct IterationReport {
    iteration: usize,
    detector_elapsed_ms: f64,
    cuts: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let eval_args = eval_args(&args);
    let decode_started = Instant::now();
    let frames = decode_frames(&args.input, &eval_args)?;
    let decode_elapsed_ms = decode_started.elapsed().as_secs_f64() * 1000.0;

    let mut iterations = Vec::new();
    for iteration in 1..=args.iterations {
        let mut detector = detector_from_args(&eval_args)?;
        let started = Instant::now();
        let mut cuts = 0usize;
        for frame in &frames {
            cuts += detector.process_frame(&frame.as_frame(), None)?.len();
        }
        if let Some(last) = frames.last() {
            cuts += detector.finish(last.position, None)?.len();
        }
        iterations.push(IterationReport {
            iteration,
            detector_elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
            cuts,
        });
    }

    let avg_detector_elapsed_ms = if iterations.is_empty() {
        0.0
    } else {
        iterations
            .iter()
            .map(|iteration| iteration.detector_elapsed_ms)
            .sum::<f64>()
            / iterations.len() as f64
    };
    let report = BenchmarkReport {
        input: args.input.to_string_lossy().into_owned(),
        detector: args.detector,
        resize_width: args.resize_width,
        frame_count: frames.len(),
        decode_elapsed_ms,
        frames_per_second: if avg_detector_elapsed_ms > 0.0 {
            frames.len() as f64 / (avg_detector_elapsed_ms / 1000.0)
        } else {
            0.0
        },
        avg_detector_elapsed_ms,
        iterations,
    };
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(output) = args.output {
        atomic_write(&output, &format!("{json}\n"))?;
    } else {
        println!("{json}");
    }
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let mut input = None;
    let mut output = None;
    let mut detector = "content".to_string();
    let mut resize_width = None;
    let mut iterations = 5usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => input = Some(PathBuf::from(next_value(&mut args, "--input")?)),
            "--output" => output = Some(PathBuf::from(next_value(&mut args, "--output")?)),
            "--detector" => detector = next_value(&mut args, "--detector")?,
            "--resize-width" => {
                let width = next_value(&mut args, "--resize-width")?
                    .parse::<u32>()
                    .map_err(|error| format!("invalid --resize-width: {error}"))?;
                if width == 0 {
                    return Err("--resize-width must be greater than 0".to_string());
                }
                resize_width = Some(width);
            }
            "--iterations" => {
                iterations = next_value(&mut args, "--iterations")?
                    .parse::<usize>()
                    .map_err(|error| format!("invalid --iterations: {error}"))?;
                if iterations == 0 {
                    return Err("--iterations must be greater than 0".to_string());
                }
            }
            "-h" | "--help" => return Err(usage()),
            _ => return Err(format!("unknown argument `{arg}`")),
        }
    }
    Ok(Args {
        input: input.ok_or_else(|| "missing --input".to_string())?,
        output,
        detector,
        resize_width,
        iterations,
    })
}

fn usage() -> String {
    "usage: scene_detector_real_frame_benchmark --input PATH [--resize-width PIXELS] [--detector content|adaptive|threshold|histogram|hash] [--iterations N] [--output PATH]".to_string()
}

fn next_value(args: &mut impl Iterator<Item = String>, name: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value for {name}"))
}

fn eval_args(args: &Args) -> EvalArgs {
    EvalArgs {
        dataset: "real-frame-benchmark".to_string(),
        root: PathBuf::new(),
        detector: args.detector.clone(),
        output: None,
        video_ids: Vec::new(),
        limit: None,
        max_runtime: None,
        progress: false,
        preset: None,
        resume: false,
        resize_width: args.resize_width,
        pixel_format: video_analysis_core::PixelFormat::Rgb24,
        config: DetectorConfig::default(),
    }
}

fn decode_frames(
    path: &Path,
    args: &EvalArgs,
) -> Result<Vec<OwnedVideoFrame>, Box<dyn std::error::Error>> {
    let mut source =
        FfmpegVideoSource::open_path_with_options(path, ffmpeg_options_from_args(args))?;
    let mut frames = Vec::new();
    while let Some(frame) = source.next_video_frame()? {
        frames.push(frame);
    }
    Ok(frames)
}

fn atomic_write(path: &Path, content: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file_name = path
        .file_name()
        .and_then(|file_name| file_name.to_str())
        .unwrap_or("scene-detector-real-frame-benchmark.json");
    let tmp = path.with_file_name(format!("{file_name}.tmp"));
    fs::write(&tmp, content)?;
    fs::rename(tmp, path)
}
