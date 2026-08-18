use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CutPlanRequest {
    pub source_id: Option<String>,
    pub intervals: Option<Vec<TimeSpanRequest>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ConcatPlanRequest {
    pub clips: Option<Vec<TimelineClipRequest>>,
    pub transitions: Option<Vec<TransitionRequest>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SubtitlePlanRequest {
    pub clips: Option<Vec<TimelineClipRequest>>,
    pub subtitles: Option<Vec<SubtitleRequest>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TimeSpanRequest {
    pub start_seconds: f64,
    pub end_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TimelineClipRequest {
    pub source_id: String,
    pub source_start_seconds: f64,
    pub source_end_seconds: f64,
    pub timeline_start_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TransitionRequest {
    pub from_clip: usize,
    pub to_clip: usize,
    pub duration_seconds: f64,
    pub kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SubtitleRequest {
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FrameApplyRequest {
    pub frame: FramePayload,
    pub edits: Vec<FrameEditRequest>,
    pub preview_limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FramePayload {
    pub width: u32,
    pub height: u32,
    pub pixel_format: String,
    pub stride: Option<usize>,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FrameEditRequest {
    #[serde(rename = "type")]
    pub kind: String,
    pub x: Option<u32>,
    pub y: Option<u32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub radius: Option<u32>,
    pub brightness: Option<i16>,
    pub contrast: Option<f32>,
    pub clockwise_turns: Option<u8>,
    pub color: Option<[u8; 3]>,
    pub amount: Option<f32>,
}
