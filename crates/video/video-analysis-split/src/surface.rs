//! Library-owned runtime surface for `video-analysis-split`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use runtime_core::{
    OperationId, PackageSurface, RuntimeCapabilities, SurfaceOperation, SurfaceRequest,
    SurfaceResponse,
};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use video_analysis_core::{FramePosition, Scene, Timebase, Timestamp};

use crate::{build_split_plan, SplitOptions, DEFAULT_TEMPLATE};

/// Returns the package surface exposed by every transport wrapper.
pub fn package_surface() -> PackageSurface {
    PackageSurface {
        library: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities: RuntimeCapabilities::pure_rust(),
        operations: vec![
            operation(
                "describe",
                "Describe package",
                "Scene and segment splitting contracts for video-analysis outputs.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "video.split.scenePlan",
                "Plan scene split",
                "Builds an in-memory split plan from scene intervals without executing FFmpeg.",
                serde_json::json!({
                    "inputVideoPath": "demo.mp4",
                    "scenes": [{"startFrame": 0, "endFrame": 30, "fpsNum": 30, "fpsDen": 1}],
                    "options": {"outputDir": "segments", "template": "$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4"}
                }),
            ),
            operation(
                "video.split.segmentManifest",
                "Build segment manifest",
                "Summarizes segment names, time ranges, frame ranges, and output paths.",
                serde_json::json!({
                    "inputVideoPath": "demo.mp4",
                    "scenes": [{"startFrame": 0, "endFrame": 30}]
                }),
            ),
            operation(
                "video.split.namingPlan",
                "Plan segment names",
                "Builds deterministic file names for split outputs and reports collisions.",
                serde_json::json!({
                    "inputVideoPath": "demo.mp4",
                    "scenes": [{"startFrame": 0, "endFrame": 30}, {"startFrame": 31, "endFrame": 60}],
                    "options": {"template": "$VIDEO_NAME-$SCENE_NUMBER.mp4", "videoName": "demo"}
                }),
            ),
        ],
    }
}

fn operation(
    id: &str,
    name: &str,
    description: &str,
    example_request: serde_json::Value,
) -> SurfaceOperation {
    SurfaceOperation {
        id: OperationId::new(id),
        name: name.to_string(),
        description: Some(description.to_string()),
        curation: runtime_core::SurfaceOperationCuration::from_operation_id(id),
        input_schema: serde_json::json!({"type": "object", "additionalProperties": true, "xOperationCategory": runtime_core::operation_category(id)}),
        output_schema: serde_json::json!({"type": "object", "xOperationCategory": runtime_core::operation_category(id)}),
        example_request,
        wasm_supported: true,
        server_supported: true,
    }
}

/// Runs one library-owned operation.
pub fn run_surface_operation(request: SurfaceRequest) -> Result<SurfaceResponse, String> {
    let operation = request.operation.clone();
    let value = match request.operation.as_str() {
        "describe" => describe_value(request.input),
        "video.split.scenePlan" => scene_plan_value(parse_input(request.input)?)?,
        "video.split.segmentManifest" => segment_manifest_value(parse_input(request.input)?)?,
        "video.split.namingPlan" => naming_plan_value(parse_input(request.input)?)?,
        operation => {
            return Err(format!(
                "unsupported operation `{operation}` for {}",
                env!("CARGO_PKG_NAME")
            ))
        }
    };
    Ok(surface_response(operation, value))
}

fn describe_value(input: serde_json::Value) -> serde_json::Value {
    let surface = package_surface();
    serde_json::json!({
        "library": surface.library,
        "version": surface.version,
        "operationCount": surface.operations.len(),
        "operations": surface.operations.iter().map(|operation| operation.id.as_str()).collect::<Vec<_>>(),
        "input": input
    })
}

fn surface_response(operation: OperationId, value: serde_json::Value) -> SurfaceResponse {
    SurfaceResponse {
        operation,
        value,
        diagnostics: Vec::new(),
        artifacts: Vec::new(),
    }
}

fn parse_input<T: DeserializeOwned>(input: serde_json::Value) -> Result<T, String> {
    serde_json::from_value(input).map_err(|error| format!("invalid request: {error}"))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SplitPlanRequest {
    input_video_path: String,
    #[serde(default)]
    scenes: Vec<SceneRequest>,
    #[serde(default)]
    options: SplitOptionsRequest,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SplitOptionsRequest {
    output_dir: Option<String>,
    template: Option<String>,
    video_name: Option<String>,
    #[serde(default)]
    ffmpeg_args: Vec<String>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct SceneRequest {
    start_frame: u64,
    end_frame: u64,
    #[serde(default = "default_fps_num")]
    fps_num: i32,
    #[serde(default = "default_fps_den")]
    fps_den: i32,
}

fn default_fps_num() -> i32 {
    30
}

fn default_fps_den() -> i32 {
    1
}

fn scene_plan_value(request: SplitPlanRequest) -> Result<serde_json::Value, String> {
    let scenes = scenes(&request.scenes)?;
    let options = split_options(&request.options);
    let plan = build_split_plan(&request.input_video_path, &scenes, &options)
        .map_err(|error| error.to_string())?;
    let jobs = plan
        .jobs
        .iter()
        .map(|job| {
            serde_json::json!({
                "sceneIndex": job.scene_index,
                "outputPath": job.output_path,
                "startSeconds": job.start_seconds,
                "durationSeconds": job.duration_seconds,
                "ffmpegArgs": job.ffmpeg_args,
                "commandArgs": job.command_args(&plan.input_video_path)
            })
        })
        .collect::<Vec<_>>();
    Ok(serde_json::json!({
        "inputVideoPath": plan.input_video_path,
        "jobCount": jobs.len(),
        "jobs": jobs,
        "externalToolsRequired": false
    }))
}

fn segment_manifest_value(request: SplitPlanRequest) -> Result<serde_json::Value, String> {
    let scenes = scenes(&request.scenes)?;
    let options = split_options(&request.options);
    let plan = build_split_plan(&request.input_video_path, &scenes, &options)
        .map_err(|error| error.to_string())?;
    let segments = plan
        .jobs
        .iter()
        .zip(scenes.iter())
        .map(|(job, scene)| {
            serde_json::json!({
                "sceneNumber": job.scene_index + 1,
                "startFrame": scene.start.frame_index,
                "endFrame": scene.end.frame_index,
                "startSeconds": job.start_seconds,
                "endSeconds": job.start_seconds + job.duration_seconds,
                "durationSeconds": job.duration_seconds,
                "outputPath": job.output_path
            })
        })
        .collect::<Vec<_>>();
    Ok(serde_json::json!({
        "inputVideoPath": plan.input_video_path,
        "segmentCount": segments.len(),
        "segments": segments
    }))
}

fn naming_plan_value(request: SplitPlanRequest) -> Result<serde_json::Value, String> {
    let scenes = scenes(&request.scenes)?;
    let options = split_options(&request.options);
    let plan = build_split_plan(&request.input_video_path, &scenes, &options)
        .map_err(|error| error.to_string())?;
    let mut seen = BTreeSet::new();
    let mut counts = BTreeMap::<String, usize>::new();
    let names = plan
        .jobs
        .iter()
        .map(|job| {
            let output = job.output_path.to_string_lossy().to_string();
            *counts.entry(output.clone()).or_default() += 1;
            let unique = seen.insert(output.clone());
            serde_json::json!({
                "sceneIndex": job.scene_index,
                "outputPath": output,
                "unique": unique
            })
        })
        .collect::<Vec<_>>();
    let collisions = counts
        .into_iter()
        .filter(|(_, count)| *count > 1)
        .map(|(output_path, count)| serde_json::json!({"outputPath": output_path, "count": count}))
        .collect::<Vec<_>>();
    Ok(serde_json::json!({
        "template": options.template,
        "videoName": options.video_name,
        "nameCount": names.len(),
        "collisionCount": collisions.len(),
        "names": names,
        "collisions": collisions
    }))
}

fn split_options(request: &SplitOptionsRequest) -> SplitOptions {
    let mut options = SplitOptions::default();
    if let Some(output_dir) = &request.output_dir {
        options.output_dir = PathBuf::from(output_dir);
    }
    if let Some(template) = &request.template {
        options.template = template.clone();
    } else {
        options.template = DEFAULT_TEMPLATE.to_string();
    }
    options.video_name = request.video_name.clone();
    if !request.ffmpeg_args.is_empty() {
        options.ffmpeg_args = request.ffmpeg_args.clone();
    }
    options
}

fn scenes(requests: &[SceneRequest]) -> Result<Vec<Scene>, String> {
    requests
        .iter()
        .map(|request| {
            if request.end_frame < request.start_frame {
                return Err(
                    "scene endFrame must be greater than or equal to startFrame".to_string()
                );
            }
            Ok(Scene {
                start: frame_position(request.start_frame, request.fps_num, request.fps_den)?,
                end: frame_position(request.end_frame, request.fps_num, request.fps_den)?,
            })
        })
        .collect()
}

fn frame_position(frame_index: u64, fps_num: i32, fps_den: i32) -> Result<FramePosition, String> {
    if fps_num <= 0 || fps_den <= 0 {
        return Err("fps numerator and denominator must be positive".to_string());
    }
    Ok(FramePosition {
        frame_index,
        timestamp: Timestamp::new(frame_index as i64, Timebase::new(fps_den, fps_num)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_plan_returns_jobs_and_command_args() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.split.scenePlan"),
            input: serde_json::json!({
                "inputVideoPath": "demo.mp4",
                "scenes": [{"startFrame": 0, "endFrame": 30}]
            }),
        })
        .expect("scene plan");
        assert_eq!(response.value["jobCount"], 1);
        assert!(response.value["jobs"][0]["commandArgs"].is_array());
        assert!(response.value.get("plan").is_none());
    }

    #[test]
    fn naming_plan_reports_collisions() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.split.namingPlan"),
            input: serde_json::json!({
                "inputVideoPath": "demo.mp4",
                "scenes": [{"startFrame": 0, "endFrame": 30}, {"startFrame": 31, "endFrame": 60}],
                "options": {"template": "same.mp4"}
            }),
        })
        .expect("naming plan");
        assert_eq!(response.value["collisionCount"], 1);
    }

    #[test]
    fn segment_manifest_rejects_invalid_scene() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.split.segmentManifest"),
            input: serde_json::json!({
                "inputVideoPath": "demo.mp4",
                "scenes": [{"startFrame": 30, "endFrame": 0}]
            }),
        })
        .expect_err("invalid scene");
        assert!(error.contains("endFrame"));
    }
}
