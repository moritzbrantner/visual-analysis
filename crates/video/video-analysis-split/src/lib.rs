#![doc = include_str!("../README.md")]

pub mod surface;
use std::path::{Path, PathBuf};
use std::process::Command;

use video_analysis_core::{DetectError, Result, Scene};

/// Constant for default template.
pub const DEFAULT_TEMPLATE: &str = "$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4";

#[derive(Debug, Clone)]
/// Data type for split options.
pub struct SplitOptions {
    /// The output dir value.
    pub output_dir: PathBuf,
    /// The template value.
    pub template: String,
    /// The video name value.
    pub video_name: Option<String>,
    /// The FFmpeg args value.
    pub ffmpeg_args: Vec<String>,
}

impl Default for SplitOptions {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("."),
            template: DEFAULT_TEMPLATE.to_string(),
            video_name: None,
            ffmpeg_args: vec![
                "-map".to_string(),
                "0:v:0".to_string(),
                "-map".to_string(),
                "0:a?".to_string(),
                "-map".to_string(),
                "0:s?".to_string(),
                "-c:v".to_string(),
                "libx264".to_string(),
                "-preset".to_string(),
                "veryfast".to_string(),
                "-crf".to_string(),
                "22".to_string(),
                "-c:a".to_string(),
                "aac".to_string(),
            ],
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for split plan.
pub struct SplitPlan {
    /// The input video path value.
    pub input_video_path: PathBuf,
    /// The jobs value.
    pub jobs: Vec<SplitJob>,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for split job.
pub struct SplitJob {
    /// The scene index value.
    pub scene_index: usize,
    /// The output path value.
    pub output_path: PathBuf,
    /// The start seconds value.
    pub start_seconds: f64,
    /// FFmpeg receives this value through `-t`.
    ///
    /// Core scene frame indices are inclusive, but media trimming uses the
    /// scene end timestamp as the exclusive end instant. This preserves the
    /// previous `end_seconds - start_seconds` behavior while documenting the
    /// policy at the split boundary.
    pub duration_seconds: f64,
    /// The FFmpeg args value.
    pub ffmpeg_args: Vec<String>,
}

/// Executes scene split jobs.
pub trait SplitExecutor {
    /// Executes a split plan and returns output paths.
    fn execute_split_plan(&mut self, plan: &SplitPlan) -> Result<Vec<PathBuf>>;
}

#[derive(Debug, Clone, Copy, Default)]
/// Command-backed FFmpeg split executor.
pub struct CommandFfmpegSplitExecutor;

impl SplitExecutor for CommandFfmpegSplitExecutor {
    fn execute_split_plan(&mut self, plan: &SplitPlan) -> Result<Vec<PathBuf>> {
        let mut outputs = Vec::new();
        for job in &plan.jobs {
            let status = Command::new("ffmpeg")
                .args(job.command_args(&plan.input_video_path))
                .status()?;
            if !status.success() {
                return Err(DetectError::Source(format!(
                    "ffmpeg failed while writing `{}`",
                    job.output_path.display()
                )));
            }
            outputs.push(job.output_path.clone());
        }
        Ok(outputs)
    }
}

#[cfg(feature = "ffmpeg-native")]
#[derive(Debug, Clone, Copy, Default)]
/// Native FFmpeg split executor.
pub struct NativeFfmpegSplitExecutor;

#[cfg(feature = "ffmpeg-native")]
impl SplitExecutor for NativeFfmpegSplitExecutor {
    fn execute_split_plan(&mut self, plan: &SplitPlan) -> Result<Vec<PathBuf>> {
        let mut command = CommandFfmpegSplitExecutor;
        command.execute_split_plan(plan)
    }
}

impl SplitJob {
    /// Returns command args.
    pub fn command_args(&self, input_video_path: &Path) -> Vec<String> {
        let mut args = vec![
            "-y".to_string(),
            "-v".to_string(),
            "error".to_string(),
            "-ss".to_string(),
            self.start_seconds.to_string(),
            "-i".to_string(),
            input_video_path.to_string_lossy().into_owned(),
            "-t".to_string(),
            self.duration_seconds.to_string(),
        ];
        args.extend(self.ffmpeg_args.clone());
        args.push(self.output_path.to_string_lossy().into_owned());
        args
    }
}

/// Builds split plan.
pub fn build_split_plan(
    input_video_path: impl AsRef<Path>,
    scenes: &[Scene],
    options: &SplitOptions,
) -> Result<SplitPlan> {
    let input_video_path = input_video_path.as_ref();
    let video_name = options.video_name.clone().unwrap_or_else(|| {
        input_video_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("video")
            .to_string()
    });
    let digits = scenes.len().to_string().len().max(3);
    let jobs = scenes
        .iter()
        .enumerate()
        .map(|(index, scene)| {
            let scene_number = format!("{:0digits$}", index + 1);
            let output_name = options
                .template
                .replace("$VIDEO_NAME", &video_name)
                .replace("$SCENE_NUMBER", &scene_number)
                .replace("$START_FRAME", &scene.start.frame_index.to_string())
                .replace("$END_FRAME", &scene.end.frame_index.to_string());
            SplitJob {
                scene_index: index,
                output_path: options.output_dir.join(output_name),
                start_seconds: scene.start.timestamp.seconds(),
                duration_seconds: (scene.end.timestamp.seconds() - scene.start.timestamp.seconds())
                    .max(0.0),
                ffmpeg_args: options.ffmpeg_args.clone(),
            }
        })
        .collect();
    Ok(SplitPlan {
        input_video_path: input_video_path.to_path_buf(),
        jobs,
    })
}

/// Returns split video FFmpeg.
pub fn split_video_ffmpeg(
    input_video_path: impl AsRef<Path>,
    scenes: &[Scene],
    options: &SplitOptions,
) -> Result<Vec<PathBuf>> {
    split_video_with_executor(
        input_video_path,
        scenes,
        options,
        &mut CommandFfmpegSplitExecutor,
    )
}

/// Splits a video using a caller-provided executor.
pub fn split_video_with_executor(
    input_video_path: impl AsRef<Path>,
    scenes: &[Scene],
    options: &SplitOptions,
    executor: &mut impl SplitExecutor,
) -> Result<Vec<PathBuf>> {
    let input_video_path = input_video_path.as_ref();
    let plan = build_split_plan(input_video_path, scenes, options)?;
    std::fs::create_dir_all(&options.output_dir)?;
    executor.execute_split_plan(&plan)
}

#[cfg(test)]
mod tests {
    use num_rational::Rational64;
    use video_analysis_core::FramePosition;

    use super::*;

    #[test]
    fn default_template_is_pyscenedetect_like() {
        assert_eq!(DEFAULT_TEMPLATE, "$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4");
    }

    #[test]
    fn split_plan_expands_template_and_keeps_default_args() {
        let scene = Scene {
            start: FramePosition::from_frame_index(10, Rational64::new(10, 1)),
            end: FramePosition::from_frame_index(25, Rational64::new(10, 1)),
        };
        let options = SplitOptions {
            output_dir: PathBuf::from("out"),
            template: "$VIDEO_NAME-$SCENE_NUMBER-$START_FRAME-$END_FRAME.mp4".to_string(),
            ..SplitOptions::default()
        };

        let plan = build_split_plan("clip.mov", &[scene], &options).unwrap();

        assert_eq!(plan.jobs.len(), 1);
        assert_eq!(
            plan.jobs[0].output_path,
            PathBuf::from("out/clip-001-10-25.mp4")
        );
        assert_eq!(plan.jobs[0].start_seconds, 1.0);
        assert_eq!(plan.jobs[0].duration_seconds, 1.5);
        assert!(plan.jobs[0]
            .command_args(&plan.input_video_path)
            .contains(&"libx264".to_string()));
    }

    #[test]
    fn split_plan_accepts_zero_scenes() {
        let plan = build_split_plan("clip.mov", &[], &SplitOptions::default()).unwrap();

        assert!(plan.jobs.is_empty());
    }
}
