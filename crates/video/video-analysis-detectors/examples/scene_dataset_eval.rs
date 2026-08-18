use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use video_analysis_ffmpeg::FfmpegVideoSource;
use video_analysis_ingest::VideoFrameSource;

mod scene_dataset_eval_support;
use scene_dataset_eval_support::{
    detector_from_args, eval_configuration, ffmpeg_options_from_args, parse_args,
    suppress_nearby_cuts, Args, EvalConfiguration,
};

pub(crate) const PYSCENEDETECT_BASELINE: &str = "0.6.7.1";

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct EvalReport {
    dataset: String,
    detector: String,
    #[serde(default = "default_implementation")]
    pub(crate) implementation: String,
    #[serde(default = "default_pyscenedetect_baseline")]
    pub(crate) pyscenedetect_baseline: String,
    #[serde(default)]
    configuration: EvalConfiguration,
    videos: Vec<VideoReport>,
    summary: EvalSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    mode: Option<EvalMode>,
}

pub(crate) fn default_implementation() -> String {
    "rust".to_string()
}

pub(crate) fn default_pyscenedetect_baseline() -> String {
    PYSCENEDETECT_BASELINE.to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EvalMode {
    video_ids: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    resize_width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_runtime_seconds: Option<u64>,
    complete: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VideoReport {
    id: String,
    path: String,
    annotation_path: Option<String>,
    predicted_cuts: Vec<u64>,
    ground_truth_cuts: Vec<u64>,
    elapsed_ms: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    decode_resize_elapsed_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    detector_elapsed_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    frame_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    effective_fps: Option<f64>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EvalSummary {
    recall: f64,
    precision: f64,
    f1: f64,
    avg_elapsed_ms: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let started = Instant::now();
    let args = parse_args()?;
    if args.resume && args.output.is_none() {
        return Err("--resume requires --output".into());
    }

    let annotations = annotation_map(&args.root)?;
    let videos = select_videos(video_files(&args.root)?, &args.video_ids, args.limit)?;
    if videos.is_empty() {
        return Err(format!("no video files selected under `{}`", args.root.display()).into());
    }
    let selected_ids = videos
        .iter()
        .map(|video| video_id(video))
        .collect::<Vec<_>>();

    let mut reports = resumed_reports(&args, &selected_ids)?;
    let completed = reports
        .iter()
        .map(|report| report.id.clone())
        .collect::<BTreeSet<_>>();

    for video in videos
        .into_iter()
        .filter(|video| !completed.contains(&video_id(video)))
    {
        if runtime_exceeded(started, args.max_runtime) {
            write_report(&args, &selected_ids, &reports, false)?;
            return Err(runtime_error(args.max_runtime));
        }

        let id = video_id(&video);
        if args.progress {
            eprintln!("scene_dataset_eval: start {id}");
        }
        let annotation_path = annotations.get(&id).cloned();
        let ground_truth_cuts = annotation_path
            .as_ref()
            .map(|path| load_annotation(path))
            .transpose()?
            .unwrap_or_default();
        let detection = match detect_video(&video, &args, started, args.max_runtime) {
            Ok(detection) => detection,
            Err(error) => {
                write_report(&args, &selected_ids, &reports, false)?;
                return Err(error);
            }
        };
        reports.push(VideoReport {
            id: id.clone(),
            path: video.to_string_lossy().into_owned(),
            annotation_path: annotation_path.map(|path| path.to_string_lossy().into_owned()),
            predicted_cuts: detection.predicted_cuts,
            ground_truth_cuts,
            elapsed_ms: detection.elapsed_ms,
            decode_resize_elapsed_ms: Some(detection.decode_resize_elapsed_ms),
            detector_elapsed_ms: Some(detection.detector_elapsed_ms),
            frame_count: Some(detection.frame_count),
            effective_fps: Some(detection.effective_fps),
        });
        reports.sort_by(|left, right| left.id.cmp(&right.id));

        let complete =
            reports.len() == selected_ids.len() && !runtime_exceeded(started, args.max_runtime);
        write_report(&args, &selected_ids, &reports, complete)?;
        if args.progress {
            eprintln!("scene_dataset_eval: finish {id}");
        }

        if runtime_exceeded(started, args.max_runtime) {
            write_report(&args, &selected_ids, &reports, false)?;
            return Err(runtime_error(args.max_runtime));
        }
    }

    let complete = reports.len() == selected_ids.len();
    write_report(&args, &selected_ids, &reports, complete)?;
    if !complete {
        return Err("scene dataset evaluation did not complete all selected videos".into());
    }
    Ok(())
}

fn select_videos(
    videos: Vec<PathBuf>,
    requested_ids: &[String],
    limit: Option<usize>,
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    if videos.is_empty() {
        return Ok(Vec::new());
    }
    let by_id = videos
        .into_iter()
        .map(|video| (video_id(&video), video))
        .collect::<BTreeMap<_, _>>();
    let mut selected = if requested_ids.is_empty() {
        by_id.values().cloned().collect::<Vec<_>>()
    } else {
        let mut selected = Vec::new();
        let mut missing = Vec::new();
        for id in requested_ids {
            if let Some(video) = by_id.get(id) {
                selected.push(video.clone());
            } else {
                missing.push(id.clone());
            }
        }
        if !missing.is_empty() {
            return Err(
                format!("requested video IDs were not found: {}", missing.join(", ")).into(),
            );
        }
        selected
    };
    selected.sort();
    if let Some(limit) = limit {
        selected.truncate(limit);
    }
    Ok(selected)
}

fn resumed_reports(
    args: &Args,
    selected_ids: &[String],
) -> Result<Vec<VideoReport>, Box<dyn std::error::Error>> {
    let Some(output) = args.output.as_ref().filter(|_| args.resume) else {
        return Ok(Vec::new());
    };
    if !output.exists() {
        return Ok(Vec::new());
    }
    let selected = selected_ids
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let report = serde_json::from_str::<EvalReport>(&fs::read_to_string(output)?)?;
    let mut reports = report
        .videos
        .into_iter()
        .filter(|report| selected.contains(report.id.as_str()))
        .collect::<Vec<_>>();
    reports.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(reports)
}

#[derive(Debug)]
struct DetectionReport {
    predicted_cuts: Vec<u64>,
    elapsed_ms: f64,
    decode_resize_elapsed_ms: f64,
    detector_elapsed_ms: f64,
    frame_count: u64,
    effective_fps: f64,
}

fn detect_video(
    path: &Path,
    args: &Args,
    started: Instant,
    max_runtime: Option<Duration>,
) -> Result<DetectionReport, Box<dyn std::error::Error>> {
    let video_started = Instant::now();
    let options = ffmpeg_options_from_args(args);
    let mut source = FfmpegVideoSource::open_path_with_options(path, options)?;
    let mut detector = detector_from_args(args)?;
    let mut cuts = Vec::new();
    let mut last_position = None;
    let mut decode_resize_elapsed = Duration::ZERO;
    let mut detector_elapsed = Duration::ZERO;
    let mut frame_count = 0_u64;
    loop {
        let decode_started = Instant::now();
        let frame = source.next_video_frame()?;
        decode_resize_elapsed += decode_started.elapsed();
        let Some(frame) = frame else {
            break;
        };
        if runtime_exceeded(started, max_runtime) {
            return Err(runtime_error(max_runtime));
        }
        frame_count += 1;
        last_position = Some(frame.position);
        let detector_started = Instant::now();
        cuts.extend(
            detector
                .process_frame(&frame.as_frame(), None)?
                .into_iter()
                .map(|cut| cut.position.frame_index),
        );
        detector_elapsed += detector_started.elapsed();
    }
    if let Some(last_position) = last_position {
        let detector_started = Instant::now();
        cuts.extend(
            detector
                .finish(last_position, None)?
                .into_iter()
                .map(|cut| cut.position.frame_index),
        );
        detector_elapsed += detector_started.elapsed();
    }
    cuts.sort_unstable();
    cuts.dedup();
    let cuts = suppress_nearby_cuts(cuts, args.config.post_filter_window);
    let elapsed_ms = video_started.elapsed().as_secs_f64() * 1000.0;
    Ok(DetectionReport {
        predicted_cuts: cuts,
        elapsed_ms,
        decode_resize_elapsed_ms: decode_resize_elapsed.as_secs_f64() * 1000.0,
        detector_elapsed_ms: detector_elapsed.as_secs_f64() * 1000.0,
        frame_count,
        effective_fps: if elapsed_ms > 0.0 {
            frame_count as f64 / (elapsed_ms / 1000.0)
        } else {
            0.0
        },
    })
}

fn runtime_exceeded(started: Instant, max_runtime: Option<Duration>) -> bool {
    max_runtime
        .map(|max_runtime| started.elapsed() >= max_runtime)
        .unwrap_or(false)
}

fn runtime_error(max_runtime: Option<Duration>) -> Box<dyn std::error::Error> {
    match max_runtime {
        Some(duration) => format!(
            "scene dataset evaluation exceeded max runtime of {} seconds",
            duration.as_secs()
        )
        .into(),
        None => "scene dataset evaluation exceeded max runtime".into(),
    }
}

fn annotation_map(root: &Path) -> std::io::Result<BTreeMap<String, PathBuf>> {
    let mut map = BTreeMap::new();
    for path in collect_files(root)? {
        let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
            continue;
        };
        if !matches!(ext.to_ascii_lowercase().as_str(), "txt" | "tsv" | "csv") {
            continue;
        }
        if let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) {
            map.entry(stem.to_string()).or_insert_with(|| path.clone());
            if let Some(prefix) = stem.split('-').next() {
                map.entry(prefix.to_string())
                    .or_insert_with(|| path.clone());
                if prefix.chars().all(|ch| ch.is_ascii_digit()) {
                    map.entry(format!("bbc_{prefix}"))
                        .or_insert_with(|| path.clone());
                }
            }
        }
    }
    Ok(map)
}

fn video_files(root: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut files = collect_files(root)?
        .into_iter()
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| {
                    matches!(
                        ext.to_ascii_lowercase().as_str(),
                        "mp4" | "webm" | "mkv" | "mov" | "avi"
                    )
                })
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    files.sort();
    Ok(files)
}

fn collect_files(root: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    collect_files_into(root, &mut files)?;
    Ok(files)
}

fn collect_files_into(path: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if path.is_file() {
        files.push(path.to_path_buf());
        return Ok(());
    }
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_files_into(&path, files)?;
        } else {
            files.push(path);
        }
    }
    Ok(())
}

fn load_annotation(path: &Path) -> Result<Vec<u64>, Box<dyn std::error::Error>> {
    let mut cuts = Vec::new();
    for line in fs::read_to_string(path)?.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields = line
            .split(['\t', ',', ' '])
            .filter(|field| !field.is_empty())
            .collect::<Vec<_>>();
        let value = fields
            .get(1)
            .or_else(|| fields.first())
            .ok_or_else(|| format!("invalid annotation row `{line}`"))?;
        if let Ok(frame) = value.parse::<u64>() {
            cuts.push(frame + 1);
        }
    }
    Ok(cuts)
}

fn write_report(
    args: &Args,
    selected_ids: &[String],
    reports: &[VideoReport],
    complete: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let report = EvalReport {
        dataset: args.dataset.clone(),
        detector: args.detector.clone(),
        implementation: "rust".to_string(),
        pyscenedetect_baseline: PYSCENEDETECT_BASELINE.to_string(),
        configuration: eval_configuration(args),
        summary: summarize(reports),
        videos: reports.to_vec(),
        mode: Some(EvalMode {
            video_ids: selected_ids.to_vec(),
            resize_width: args.resize_width,
            max_runtime_seconds: args.max_runtime.map(|duration| duration.as_secs()),
            complete,
        }),
    };
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(output) = &args.output {
        atomic_write(output, &format!("{json}\n"))?;
    } else if complete {
        println!("{json}");
    }
    Ok(())
}

fn atomic_write(path: &Path, content: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file_name = path
        .file_name()
        .and_then(|file_name| file_name.to_str())
        .unwrap_or("scene-dataset-eval.json");
    let tmp = path.with_file_name(format!("{file_name}.tmp"));
    fs::write(&tmp, content)?;
    fs::rename(tmp, path)
}

fn video_id(path: &Path) -> String {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("video")
        .to_string()
}

fn summarize(reports: &[VideoReport]) -> EvalSummary {
    let mut correct = 0usize;
    let mut predicted = 0usize;
    let mut ground_truth = 0usize;
    let elapsed = reports.iter().map(|report| report.elapsed_ms).sum::<f64>();
    for report in reports {
        let gt = report
            .ground_truth_cuts
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        correct += report
            .predicted_cuts
            .iter()
            .filter(|frame| gt.contains(frame))
            .count();
        predicted += report.predicted_cuts.len();
        ground_truth += report.ground_truth_cuts.len();
    }
    let recall = if ground_truth == 0 {
        0.0
    } else {
        correct as f64 / ground_truth as f64
    };
    let precision = if predicted == 0 {
        0.0
    } else {
        correct as f64 / predicted as f64
    };
    let f1 = if recall + precision == 0.0 {
        0.0
    } else {
        2.0 * recall * precision / (recall + precision)
    };
    EvalSummary {
        recall,
        precision,
        f1,
        avg_elapsed_ms: if reports.is_empty() {
            0.0
        } else {
            elapsed / reports.len() as f64
        },
    }
}
