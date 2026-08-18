use video_analysis_core::{FramePosition, PixelFormat, Timebase, Timestamp, VideoFrame};

use crate::contracts::{
    ConcatPlanRequest, CutPlanRequest, FrameApplyRequest, FrameEditRequest, FramePayload,
    SubtitlePlanRequest, TimelineClipRequest, TransitionRequest,
};
use crate::{
    build_concat_plan, build_cut_plan, build_subtitle_plan, FrameEdit, FrameEditor,
    SubtitleOverlay, TimeSpan, TimelineClip, TransitionKind, TransitionSpec,
};

pub fn cut_plan_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let request: CutPlanRequest =
        serde_json::from_value(input).map_err(|error| format!("invalid request: {error}"))?;
    let intervals = request
        .intervals
        .unwrap_or_default()
        .into_iter()
        .map(|span| TimeSpan::new(span.start_seconds, span.end_seconds))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    let edl = build_cut_plan(
        request.source_id.unwrap_or_else(|| "source".to_string()),
        &intervals,
    )
    .map_err(|error| error.to_string())?;
    Ok(edl_value(&edl))
}

pub fn concat_plan_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let request: ConcatPlanRequest =
        serde_json::from_value(input).map_err(|error| format!("invalid request: {error}"))?;
    let clips = parse_clips(request.clips.unwrap_or_default())?;
    let transitions = request
        .transitions
        .unwrap_or_default()
        .into_iter()
        .map(parse_transition)
        .collect::<Result<Vec<_>, _>>()?;
    let edl = build_concat_plan(clips, transitions).map_err(|error| error.to_string())?;
    Ok(edl_value(&edl))
}

pub fn subtitle_plan_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let request: SubtitlePlanRequest =
        serde_json::from_value(input).map_err(|error| format!("invalid request: {error}"))?;
    let clips = parse_clips(request.clips.unwrap_or_default())?;
    let subtitles = request
        .subtitles
        .unwrap_or_default()
        .into_iter()
        .map(|subtitle| {
            SubtitleOverlay::new(subtitle.start_seconds, subtitle.end_seconds, subtitle.text)
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let edl = build_subtitle_plan(clips, subtitles).map_err(|error| error.to_string())?;
    Ok(edl_value(&edl))
}

pub fn frame_apply_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let request: FrameApplyRequest =
        serde_json::from_value(input).map_err(|error| format!("invalid request: {error}"))?;
    let frame = request.frame.frame()?;
    let mut editor = FrameEditor::new();
    for edit in request.edits {
        editor = editor.edit(edit.edit()?);
    }
    let output = editor.apply(&frame).map_err(|error| error.to_string())?;
    let preview_limit = request.preview_limit.unwrap_or(32).min(512);
    Ok(serde_json::json!({
        "deterministic": true,
        "externalToolsRequired": false,
        "width": output.width,
        "height": output.height,
        "pixelFormat": pixel_format_name(output.pixel_format),
        "stride": output.stride,
        "dataLength": output.data.len(),
        "dataPreview": output.data.iter().take(preview_limit).copied().collect::<Vec<_>>(),
        "truncated": output.data.len() > preview_limit
    }))
}

pub fn deterministic_operation_value(
    surface: &runtime_core::PackageSurface,
    operation: &runtime_core::SurfaceOperation,
    input: serde_json::Value,
) -> serde_json::Value {
    let operation_id = operation.id.as_str();
    let result = serde_json::json!({
        "library": &surface.library,
        "version": &surface.version,
        "operation": operation_id,
        "name": &operation.name,
        "description": &operation.description,
        "deterministic": true,
        "externalToolsRequired": false,
        "request": input,
        "output": {
            "operationFamily": operation_family(operation_id),
            "sideEffects": false,
            "artifactMode": "inline-json"
        }
    });
    runtime_core::structured_surface_value(
        &operation.id,
        operation.name.clone(),
        operation
            .description
            .clone()
            .unwrap_or_else(|| format!("Ran package-surface operation `{operation_id}`.")),
        serde_json::json!({
            "status": "ok",
            "operationFamily": operation_family(operation_id),
            "externalToolsRequired": false
        }),
        result,
    )
}

fn parse_clips(requests: Vec<TimelineClipRequest>) -> Result<Vec<TimelineClip>, String> {
    requests
        .into_iter()
        .map(|clip| {
            TimelineClip::new(
                clip.source_id,
                clip.source_start_seconds,
                clip.source_end_seconds,
                clip.timeline_start_seconds,
            )
            .map_err(|error| error.to_string())
        })
        .collect()
}

fn parse_transition(request: TransitionRequest) -> Result<TransitionSpec, String> {
    TransitionSpec::new(
        request.from_clip,
        request.to_clip,
        request.duration_seconds,
        transition_kind(&request.kind)?,
    )
    .map_err(|error| error.to_string())
}

fn transition_kind(value: &str) -> Result<TransitionKind, String> {
    match value {
        "cut" => Ok(TransitionKind::Cut),
        "crossFade" | "cross_fade" => Ok(TransitionKind::CrossFade),
        other => Err(format!("unsupported transition kind `{other}`")),
    }
}

fn edl_value(edl: &crate::EditDecisionList) -> serde_json::Value {
    serde_json::json!({
        "deterministic": true,
        "externalToolsRequired": false,
        "clipCount": edl.clips.len(),
        "transitionCount": edl.transitions.len(),
        "subtitleCount": edl.subtitles.len(),
        "durationSeconds": edl.duration_seconds(),
        "clips": edl.clips.iter().map(|clip| serde_json::json!({
            "sourceId": clip.source_id,
            "sourceStartSeconds": clip.source_start_seconds,
            "sourceEndSeconds": clip.source_end_seconds,
            "timelineStartSeconds": clip.timeline_start_seconds,
            "timelineEndSeconds": clip.timeline_end_seconds()
        })).collect::<Vec<_>>(),
        "transitions": edl.transitions.iter().map(|transition| serde_json::json!({
            "fromClip": transition.from_clip,
            "toClip": transition.to_clip,
            "durationSeconds": transition.duration_seconds,
            "kind": match transition.kind {
                TransitionKind::Cut => "cut",
                TransitionKind::CrossFade => "crossFade",
            }
        })).collect::<Vec<_>>(),
        "subtitles": edl.subtitles.iter().map(|subtitle| serde_json::json!({
            "startSeconds": subtitle.start_seconds,
            "endSeconds": subtitle.end_seconds,
            "text": subtitle.text
        })).collect::<Vec<_>>()
    })
}

impl FramePayload {
    fn frame(&self) -> Result<VideoFrame<'_>, String> {
        let pixel_format = parse_pixel_format(&self.pixel_format)?;
        let stride = self.stride.unwrap_or(self.width as usize * 3);
        if stride < self.width as usize * 3 {
            return Err("frame stride must be at least width * 3".to_string());
        }
        VideoFrame::packed(
            FramePosition {
                frame_index: 0,
                timestamp: Timestamp::new(0, Timebase::new(1, 1)),
            },
            self.width,
            self.height,
            pixel_format,
            &self.data,
            stride,
        )
        .map_err(|error| error.to_string())
    }
}

impl FrameEditRequest {
    fn edit(self) -> Result<FrameEdit, String> {
        match self.kind.as_str() {
            "crop" => Ok(FrameEdit::Crop(
                video_analysis_core::BoundingBox::new(
                    self.x.ok_or("crop requires x")?,
                    self.y.ok_or("crop requires y")?,
                    self.width.ok_or("crop requires width")?,
                    self.height.ok_or("crop requires height")?,
                )
                .map_err(|error| error.to_string())?,
            )),
            "resizeNearest" => Ok(FrameEdit::ResizeNearest {
                width: self.width.ok_or("resizeNearest requires width")?,
                height: self.height.ok_or("resizeNearest requires height")?,
            }),
            "boxBlur" => Ok(FrameEdit::BoxBlur {
                radius: self.radius.unwrap_or(1),
            }),
            "grayscale" => Ok(FrameEdit::Grayscale),
            "invert" => Ok(FrameEdit::Invert),
            "flipHorizontal" => Ok(FrameEdit::FlipHorizontal),
            "flipVertical" => Ok(FrameEdit::FlipVertical),
            "rotate90" => Ok(FrameEdit::Rotate90 {
                clockwise_turns: self.clockwise_turns.unwrap_or(1),
            }),
            "fadeToColor" => Ok(FrameEdit::FadeToColor {
                color: self.color.unwrap_or([0, 0, 0]),
                amount: self.amount.unwrap_or(1.0),
            }),
            "brightnessContrast" => Ok(FrameEdit::BrightnessContrast {
                brightness: self.brightness.unwrap_or(0),
                contrast: self.contrast.unwrap_or(1.0),
            }),
            other => Err(format!("unsupported frame edit `{other}`")),
        }
    }
}

fn parse_pixel_format(value: &str) -> Result<PixelFormat, String> {
    match value {
        "rgb24" => Ok(PixelFormat::Rgb24),
        "bgr24" => Ok(PixelFormat::Bgr24),
        other => Err(format!("unsupported pixel format `{other}`")),
    }
}

fn pixel_format_name(format: PixelFormat) -> &'static str {
    match format {
        PixelFormat::Rgb24 => "rgb24",
        PixelFormat::Bgr24 => "bgr24",
    }
}

pub fn sample_frame_json() -> serde_json::Value {
    serde_json::json!({
        "width": 2,
        "height": 2,
        "pixelFormat": "rgb24",
        "stride": null,
        "data": [255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]
    })
}

fn operation_family(operation_id: &str) -> &str {
    operation_id
        .split_once('.')
        .map(|(family, _)| family)
        .unwrap_or(operation_id)
}
