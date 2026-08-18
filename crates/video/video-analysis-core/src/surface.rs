//! Library-owned runtime surface for `video-analysis-core`.

use num_rational::Rational64;
use num_traits::ToPrimitive;
use serde::Deserialize;

use crate::runtime::{
    describe_surface_response, parse_surface_input, set_surface_operation_curation,
    structured_operation_response, surface_operation, surface_operation_with_execution_plan,
    OperationId, PackageSurface, RuntimeCapabilities, SurfaceError, SurfaceExecutionMode,
    SurfaceExecutionPlan, SurfaceOperation, SurfaceOperationCuration, SurfaceRequest,
    SurfaceResponse, SurfaceSideEffect,
};
use crate::FrameTimecode;

/// Returns the package surface exposed by every transport wrapper.
pub fn package_surface() -> PackageSurface {
    PackageSurface {
        library: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities: RuntimeCapabilities::pure_rust(),
        operations: vec![
            curated(
                surface_operation(
                "describe",
                "Describe package",
                "Core media, timing, detection, and analyzer contracts for video-analysis.",
                serde_json::json!({"includeOperations": true}),
            ),
                SurfaceOperationCuration::debug(900),
            ),
            timecode_operation(),
            curated(
                surface_operation_with_execution_plan(
                "video.core.frameSummary",
                "Summarize frames",
                "Summarizes frame timing, dimensions, pixel formats, byte counts, and validation diagnostics.",
                serde_json::json!({"frames": [{"frameIndex": 0, "timestampSeconds": 0.0, "width": 1920, "height": 1080, "pixelFormat": "rgb24", "stride": 5760, "bytes": 6220800}]}),
                in_memory_plan("video.core.frameSummary"),
            ),
                SurfaceOperationCuration::workflow(10).primary(),
            ),
            curated(
                surface_operation_with_execution_plan(
                "video.core.sceneSummary",
                "Summarize scenes",
                "Summarizes scene and cut intervals using core video timing contracts.",
                serde_json::json!({"fps": {"numerator": 24, "denominator": 1}, "scenes": [{"startFrame": 0, "endFrame": 48}, {"startFrame": 48, "endFrame": 96}]}),
                in_memory_plan("video.core.sceneSummary"),
            ),
                SurfaceOperationCuration::debug(910),
            ),
        ],
    }
}

fn timecode_operation() -> SurfaceOperation {
    let mut operation = surface_operation_with_execution_plan(
        "video.core.timecode",
        "Convert timecode",
        "Converts between frame indexes, seconds, timestamps, and display timecode values.",
        serde_json::json!({"fps": {"numerator": 30000, "denominator": 1001}, "frames": [0, 30], "seconds": [1.5], "timecodes": ["00:00:02.000"]}),
        in_memory_plan("video.core.timecode"),
    );
    set_surface_operation_curation(&mut operation, SurfaceOperationCuration::workflow(20));
    crate::runtime::attach_landscape_contract(
        &mut operation,
        crate::runtime::landscape::LandscapeOperationContract::new(
            crate::runtime::landscape::LandscapeFunction::new(
                "video.core.parseTimecode",
                env!("CARGO_PKG_NAME"),
            )
            .input(crate::runtime::landscape::LandscapePort::new(
                "timecode",
                crate::runtime::landscape::well_known::video_timecode(),
            ))
            .output(crate::runtime::landscape::LandscapePort::new(
                "timecode",
                crate::runtime::landscape::well_known::video_timecode(),
            )),
        ),
    );
    operation
}

fn curated(
    mut operation: SurfaceOperation,
    curation: SurfaceOperationCuration,
) -> SurfaceOperation {
    set_surface_operation_curation(&mut operation, curation);
    operation
}

fn in_memory_plan(operation: &str) -> SurfaceExecutionPlan {
    SurfaceExecutionPlan {
        operation: OperationId::new(operation),
        mode: SurfaceExecutionMode::InMemory,
        side_effects: vec![SurfaceSideEffect::None],
        cancellable: false,
        progress_unit: None,
        expected_artifacts: Vec::new(),
        requirements: Vec::new(),
        max_recommended_input_bytes: Some(1_048_576),
    }
}

/// Runs one library-owned operation.
pub fn run_surface_operation(request: SurfaceRequest) -> Result<SurfaceResponse, String> {
    let surface = package_surface();
    let operation = request.operation.clone();
    let value = match request.operation.as_str() {
        "describe" => return Ok(describe_surface_response(&surface, request)),
        "video.core.timecode" => timecode_value(
            operation.as_str(),
            parse_surface_input(Some(operation.as_str()), request.input)?,
        )?,
        "video.core.frameSummary" => frame_summary_value(
            operation.as_str(),
            parse_surface_input(Some(operation.as_str()), request.input)?,
        )?,
        "video.core.sceneSummary" => scene_summary_value(
            operation.as_str(),
            parse_surface_input(Some(operation.as_str()), request.input)?,
        )?,
        operation => {
            return Err(
                SurfaceError::unsupported_operation(operation, env!("CARGO_PKG_NAME"))
                    .to_error_string(),
            );
        }
    };
    Ok(structured_operation_response(&surface, operation, value))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FpsRequest {
    numerator: i64,
    denominator: i64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TimecodeRequest {
    fps: FpsRequest,
    #[serde(default)]
    frames: Vec<u64>,
    #[serde(default)]
    seconds: Vec<f64>,
    #[serde(default)]
    timecodes: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FrameSummaryRequest {
    frames: Vec<FrameInput>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FrameInput {
    frame_index: u64,
    timestamp_seconds: f64,
    width: u32,
    height: u32,
    pixel_format: String,
    stride: usize,
    bytes: usize,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SceneSummaryRequest {
    fps: FpsRequest,
    scenes: Vec<SceneInput>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SceneInput {
    start_frame: u64,
    end_frame: u64,
}

fn timecode_value(operation: &str, request: TimecodeRequest) -> Result<serde_json::Value, String> {
    let fps = parse_fps(operation, request.fps)?;
    let fps_float = fps
        .to_f64()
        .ok_or_else(|| invalid_request(operation, "fps cannot be represented as f64"))?;
    let frame_entries = request
        .frames
        .into_iter()
        .map(|frame| timecode_entry_from_frame(frame, fps))
        .collect::<Vec<_>>();
    let second_entries = request
        .seconds
        .into_iter()
        .map(|seconds| {
            if seconds < 0.0 || !seconds.is_finite() {
                return Err(invalid_request(
                    operation,
                    "seconds must be finite and greater than or equal to zero",
                ));
            }
            let timecode = FrameTimecode::from_seconds(seconds, fps)
                .map_err(|error| invalid_request(operation, error.to_string()))?;
            Ok(serde_json::json!({
                "seconds": seconds,
                "frameIndex": timecode.frame_index,
                "display": timecode.timecode(3),
                "timestamp": timestamp_json(timecode)
            }))
        })
        .collect::<Result<Vec<serde_json::Value>, String>>()?;
    let parsed_timecodes = request
        .timecodes
        .into_iter()
        .map(|value| {
            let timecode = FrameTimecode::parse(&value, fps)
                .map_err(|error| invalid_request(operation, error.to_string()))?;
            Ok(serde_json::json!({
                "input": value,
                "frameIndex": timecode.frame_index,
                "seconds": timecode.seconds(),
                "display": timecode.timecode(3),
                "timestamp": timestamp_json(timecode)
            }))
        })
        .collect::<Result<Vec<serde_json::Value>, String>>()?;
    let conversion_count = frame_entries.len() + second_entries.len() + parsed_timecodes.len();

    Ok(serde_json::json!({
        "fps": {
            "numerator": fps.numer(),
            "denominator": fps.denom(),
            "value": fps_float
        },
        "frames": frame_entries,
        "seconds": second_entries,
        "timecodes": parsed_timecodes,
        "conversionCount": conversion_count
    }))
}

fn frame_summary_value(
    operation: &str,
    request: FrameSummaryRequest,
) -> Result<serde_json::Value, String> {
    if request.frames.is_empty() {
        return Err(invalid_request(operation, "frames must not be empty"));
    }
    let mut diagnostics = Vec::new();
    let mut total_bytes: u64 = 0;
    let mut min_timestamp = f64::INFINITY;
    let mut max_timestamp = f64::NEG_INFINITY;
    let mut dimensions = std::collections::BTreeMap::<String, usize>::new();
    let mut pixel_formats = std::collections::BTreeMap::<String, usize>::new();
    let mut previous_frame = None;

    for frame in &request.frames {
        validate_frame(operation, frame)?;
        total_bytes += frame.bytes as u64;
        min_timestamp = min_timestamp.min(frame.timestamp_seconds);
        max_timestamp = max_timestamp.max(frame.timestamp_seconds);
        *dimensions
            .entry(format!("{}x{}", frame.width, frame.height))
            .or_default() += 1;
        *pixel_formats.entry(frame.pixel_format.clone()).or_default() += 1;
        if let Some(previous) = previous_frame {
            if frame.frame_index <= previous {
                diagnostics.push(serde_json::json!({
                    "code": "non_monotonic_frame_index",
                    "message": format!("frame index {} is not greater than previous frame index {previous}", frame.frame_index)
                }));
            }
        }
        previous_frame = Some(frame.frame_index);
    }

    Ok(serde_json::json!({
        "frameCount": request.frames.len(),
        "dimensions": dimensions,
        "pixelFormats": pixel_formats,
        "durationBounds": {
            "startSeconds": min_timestamp,
            "endSeconds": max_timestamp,
            "spanSeconds": (max_timestamp - min_timestamp).max(0.0)
        },
        "totalBytes": total_bytes,
        "averageBytesPerFrame": total_bytes as f64 / request.frames.len() as f64,
        "diagnostics": diagnostics,
        "valid": diagnostics.is_empty()
    }))
}

fn scene_summary_value(
    operation: &str,
    request: SceneSummaryRequest,
) -> Result<serde_json::Value, String> {
    let fps = parse_fps(operation, request.fps)?;
    if request.scenes.is_empty() {
        return Err(invalid_request(operation, "scenes must not be empty"));
    }
    let mut timeline_start = u64::MAX;
    let mut timeline_end = 0_u64;
    let mut total_frames = 0_u64;
    let mut previous_end = None;
    let scenes = request
        .scenes
        .iter()
        .enumerate()
        .map(|(index, scene)| {
            if scene.end_frame <= scene.start_frame {
                return Err(invalid_request(
                    operation,
                    format!("scene {index} endFrame must be greater than startFrame"),
                ));
            }
            if let Some(previous) = previous_end {
                if scene.start_frame < previous {
                    return Err(invalid_request(
                        operation,
                        format!("scene {index} overlaps the previous scene"),
                    ));
                }
            }
            previous_end = Some(scene.end_frame);
            timeline_start = timeline_start.min(scene.start_frame);
            timeline_end = timeline_end.max(scene.end_frame);
            let frames = scene.end_frame - scene.start_frame;
            total_frames += frames;
            let start = FrameTimecode::from_frames(scene.start_frame, fps);
            let end = FrameTimecode::from_frames(scene.end_frame, fps);
            Ok(serde_json::json!({
                "index": index,
                "startFrame": scene.start_frame,
                "endFrame": scene.end_frame,
                "frameCount": frames,
                "startSeconds": start.seconds(),
                "endSeconds": end.seconds(),
                "durationSeconds": frames as f64 / fps.to_f64().unwrap_or(1.0),
                "startTimecode": start.timecode(3),
                "endTimecode": end.timecode(3)
            }))
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(serde_json::json!({
        "sceneCount": scenes.len(),
        "cutCount": scenes.len().saturating_sub(1),
        "totalFrames": total_frames,
        "totalDurationSeconds": total_frames as f64 / fps.to_f64().unwrap_or(1.0),
        "timelineBounds": {
            "startFrame": timeline_start,
            "endFrame": timeline_end,
            "startSeconds": FrameTimecode::from_frames(timeline_start, fps).seconds(),
            "endSeconds": FrameTimecode::from_frames(timeline_end, fps).seconds()
        },
        "scenes": scenes
    }))
}

fn parse_fps(operation: &str, fps: FpsRequest) -> Result<Rational64, String> {
    if fps.numerator <= 0 || fps.denominator <= 0 {
        return Err(invalid_request(
            operation,
            "fps numerator and denominator must be positive",
        ));
    }
    Ok(Rational64::new(fps.numerator, fps.denominator))
}

fn timecode_entry_from_frame(frame: u64, fps: Rational64) -> serde_json::Value {
    let timecode = FrameTimecode::from_frames(frame, fps);
    serde_json::json!({
        "frameIndex": frame,
        "seconds": timecode.seconds(),
        "display": timecode.timecode(3),
        "timestamp": timestamp_json(timecode)
    })
}

fn timestamp_json(timecode: FrameTimecode) -> serde_json::Value {
    let position = timecode.position();
    serde_json::json!({
        "pts": position.timestamp.pts,
        "timebase": {
            "numerator": position.timestamp.timebase.num,
            "denominator": position.timestamp.timebase.den
        },
        "seconds": position.timestamp.seconds()
    })
}

fn validate_frame(operation: &str, frame: &FrameInput) -> Result<(), String> {
    if frame.width == 0 || frame.height == 0 {
        return Err(invalid_request(
            operation,
            format!("invalid dimensions: {}x{}", frame.width, frame.height),
        ));
    }
    if !frame.timestamp_seconds.is_finite() || frame.timestamp_seconds < 0.0 {
        return Err(invalid_request(
            operation,
            "timestampSeconds must be finite and greater than or equal to zero",
        ));
    }
    let bytes_per_pixel = match frame.pixel_format.as_str() {
        "rgb24" | "bgr24" => 3,
        other => {
            return Err(SurfaceError::unsupported_value(
                Some(OperationId::new(operation)),
                "pixelFormat",
                other,
                &["rgb24", "bgr24"],
            )
            .to_error_string())
        }
    };
    let minimum_stride = frame.width as usize * bytes_per_pixel;
    if frame.stride < minimum_stride {
        return Err(invalid_request(
            operation,
            format!(
                "stride {} is smaller than minimum packed stride {minimum_stride}",
                frame.stride
            ),
        ));
    }
    let expected_bytes = frame.stride * frame.height as usize;
    if frame.bytes < expected_bytes {
        return Err(SurfaceError::artifact_error(
            OperationId::new(operation),
            format!(
                "frame buffer is too small: expected at least {expected_bytes} bytes, got {}",
                frame.bytes
            ),
            serde_json::json!({
                "expected": expected_bytes,
                "actual": frame.bytes
            }),
        )
        .to_error_string());
    }
    Ok(())
}

fn invalid_request(operation: &str, message: impl Into<String>) -> String {
    SurfaceError::invalid_request(Some(OperationId::new(operation)), message).to_error_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timecode_operation_converts_inputs() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.core.timecode"),
            input: serde_json::json!({
                "fps": {"numerator": 24, "denominator": 1},
                "frames": [48],
                "seconds": [1.5],
                "timecodes": ["00:00:02.000"]
            }),
        })
        .expect("timecode");

        assert_eq!(response.value["operation"], "video.core.timecode");
        assert_eq!(response.value["frames"][0]["display"], "00:00:02.000");
        assert_eq!(response.value["seconds"][0]["frameIndex"], 36);
        assert_eq!(response.value["timecodes"][0]["frameIndex"], 48);
    }

    #[test]
    fn frame_summary_reports_counts_and_diagnostics() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.core.frameSummary"),
            input: serde_json::json!({
                "frames": [
                    {"frameIndex": 0, "timestampSeconds": 0.0, "width": 2, "height": 2, "pixelFormat": "rgb24", "stride": 6, "bytes": 12},
                    {"frameIndex": 1, "timestampSeconds": 0.5, "width": 2, "height": 2, "pixelFormat": "rgb24", "stride": 6, "bytes": 12}
                ]
            }),
        })
        .expect("frame summary");

        assert_eq!(response.value["frameCount"], 2);
        assert_eq!(response.value["totalBytes"], 24);
        assert_eq!(response.value["valid"], true);
    }

    #[test]
    fn scene_summary_reports_scene_durations() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.core.sceneSummary"),
            input: serde_json::json!({
                "fps": {"numerator": 24, "denominator": 1},
                "scenes": [{"startFrame": 0, "endFrame": 24}, {"startFrame": 24, "endFrame": 48}]
            }),
        })
        .expect("scene summary");

        assert_eq!(response.value["sceneCount"], 2);
        assert_eq!(response.value["cutCount"], 1);
        assert_eq!(response.value["scenes"][0]["durationSeconds"], 1.0);
    }

    #[test]
    fn unknown_operation_returns_typed_error() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.core.nope"),
            input: serde_json::json!({}),
        })
        .expect_err("unknown operation");
        let parsed = crate::runtime::parse_surface_error(&error).expect("typed error");
        assert_eq!(parsed.code, "unsupported_operation");
    }

    #[test]
    fn unsupported_pixel_format_returns_typed_error() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.core.frameSummary"),
            input: serde_json::json!({
                "frames": [{"frameIndex": 0, "timestampSeconds": 0.0, "width": 2, "height": 2, "pixelFormat": "yuv420p", "stride": 2, "bytes": 8}]
            }),
        })
        .expect_err("unsupported format");
        let parsed = crate::runtime::parse_surface_error(&error).expect("typed error");
        assert_eq!(parsed.code, "unsupported_value");
        assert_eq!(parsed.details["field"], "pixelFormat");
    }
}
