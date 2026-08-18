//! Library-owned runtime surface for `video-analysis-output`.

use std::collections::BTreeMap;

use runtime_core::{
    OperationId, PackageSurface, RuntimeCapabilities, SurfaceOperation, SurfaceRequest,
    SurfaceResponse,
};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use video_analysis_core::{
    Cut, DetectionResult, FramePosition, MetricsSink, MetricsStore, Scene, Timebase, Timestamp,
};

use crate::{
    write_detection_result_json, write_scene_list_csv, write_scene_list_edl, write_scene_list_fcp7,
    write_scene_list_fcpx, write_scene_list_html, write_scene_list_otio, write_scene_list_qp,
    write_stats_csv,
};

const DEFAULT_PREVIEW_LIMIT: usize = 20;

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
                "Report, export, and presentation contracts for video-analysis results.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "video.output.reportSummary",
                "Summarize report",
                "Summarizes detection result scenes, cuts, metrics, and JSON report size.",
                serde_json::json!({
                    "result": {
                        "framesProcessed": 30,
                        "scenes": [{"startFrame": 0, "endFrame": 30}],
                        "cuts": [{"frameIndex": 30, "detector": "content", "score": 0.9}],
                        "metrics": [{"frameIndex": 0, "values": {"content": 0.1}}]
                    }
                }),
            ),
            operation(
                "video.output.csvPlan",
                "Plan CSV export",
                "Renders scene and optional stats CSV previews in memory without writing files.",
                serde_json::json!({
                    "scenes": [{"startFrame": 0, "endFrame": 30}],
                    "metrics": [{"frameIndex": 0, "values": {"content": 0.1}}],
                    "previewLimit": 20
                }),
            ),
            operation(
                "video.output.htmlPlan",
                "Plan HTML export",
                "Renders an HTML scene list preview in memory without writing files.",
                serde_json::json!({"scenes": [{"startFrame": 0, "endFrame": 30}], "previewLimit": 20}),
            ),
            operation(
                "video.output.editListPlan",
                "Preview edit-list export",
                "Renders EDL, FCP7 XML, FCPXML, OTIO JSON, or qpfile scene-list previews in memory.",
                serde_json::json!({
                    "format": "edl",
                    "scenes": [{"startFrame": 0, "endFrame": 30}],
                    "previewLimit": 20
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
    let mut operation = SurfaceOperation {
        id: OperationId::new(id),
        name: name.to_string(),
        description: Some(description.to_string()),
        curation: runtime_core::SurfaceOperationCuration::from_operation_id(id),
        input_schema: serde_json::json!({"type": "object", "additionalProperties": true, "xOperationCategory": runtime_core::operation_category(id)}),
        output_schema: serde_json::json!({"type": "object", "xOperationCategory": runtime_core::operation_category(id)}),
        example_request,
        wasm_supported: true,
        server_supported: true,
    };
    if let Some(contract) = landscape_contract(id) {
        runtime_core::attach_landscape_contract(&mut operation, contract);
    }
    operation
}

fn landscape_contract(id: &str) -> Option<runtime_core::landscape::LandscapeOperationContract> {
    match id {
        "video.output.csvPlan" => Some(runtime_core::landscape::LandscapeOperationContract::new(
            runtime_core::landscape::LandscapeFunction::new(
                "video.output.planSceneListCsv",
                env!("CARGO_PKG_NAME"),
            )
            .input(
                runtime_core::landscape::LandscapePort::new(
                    "scenes",
                    runtime_core::landscape::well_known::video_scene(),
                )
                .many(),
            )
            .output(runtime_core::landscape::LandscapePort::new(
                "sceneList",
                runtime_core::landscape::well_known::video_scene_list(),
            )),
        )),
        _ => None,
    }
}

/// Runs one library-owned operation.
pub fn run_surface_operation(request: SurfaceRequest) -> Result<SurfaceResponse, String> {
    let operation = request.operation.clone();
    let value = match request.operation.as_str() {
        "describe" => describe_value(request.input),
        "video.output.reportSummary" => report_summary_value(parse_input(request.input)?)?,
        "video.output.csvPlan" => csv_plan_value(parse_input(request.input)?)?,
        "video.output.htmlPlan" => html_plan_value(parse_input(request.input)?)?,
        "video.output.editListPlan" => edit_list_plan_value(parse_input(request.input)?)?,
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
struct ReportSummaryRequest {
    result: DetectionResultRequest,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CsvPlanRequest {
    #[serde(default)]
    scenes: Vec<SceneRequest>,
    #[serde(default)]
    metrics: Vec<MetricRowRequest>,
    preview_limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct HtmlPlanRequest {
    #[serde(default)]
    scenes: Vec<SceneRequest>,
    preview_limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EditListPlanRequest {
    #[serde(default = "default_edit_list_format")]
    format: String,
    #[serde(default)]
    scenes: Vec<SceneRequest>,
    preview_limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DetectionResultRequest {
    #[serde(default)]
    scenes: Vec<SceneRequest>,
    #[serde(default)]
    cuts: Vec<CutRequest>,
    #[serde(default)]
    metrics: Vec<MetricRowRequest>,
    #[serde(default)]
    frames_processed: u64,
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

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CutRequest {
    frame_index: u64,
    #[serde(default = "default_fps_num")]
    fps_num: i32,
    #[serde(default = "default_fps_den")]
    fps_den: i32,
    #[serde(default = "default_detector")]
    detector: String,
    score: Option<f32>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct MetricRowRequest {
    frame_index: u64,
    #[serde(default)]
    values: BTreeMap<String, f64>,
}

fn default_fps_num() -> i32 {
    30
}

fn default_fps_den() -> i32 {
    1
}

fn default_detector() -> String {
    "surface".to_string()
}

fn default_edit_list_format() -> String {
    "edl".to_string()
}

fn report_summary_value(request: ReportSummaryRequest) -> Result<serde_json::Value, String> {
    let result = detection_result(request.result)?;
    let mut json = Vec::new();
    write_detection_result_json(&mut json, &result).map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "sceneCount": result.scenes.len(),
        "cutCount": result.cuts.len(),
        "metricKeys": result.metrics.keys().collect::<Vec<_>>(),
        "metricRowCount": result.metrics.rows().len(),
        "framesProcessed": result.frames_processed,
        "jsonBytes": json.len()
    }))
}

fn csv_plan_value(request: CsvPlanRequest) -> Result<serde_json::Value, String> {
    let scenes = scenes(&request.scenes)?;
    let mut scene_csv = Vec::new();
    write_scene_list_csv(&mut scene_csv, &scenes).map_err(|error| error.to_string())?;
    let scene_csv = String::from_utf8(scene_csv).map_err(|error| error.to_string())?;

    let stats = if request.metrics.is_empty() {
        None
    } else {
        let metrics = metrics_store(&request.metrics)?;
        let mut stats_csv = Vec::new();
        write_stats_csv(&mut stats_csv, &metrics).map_err(|error| error.to_string())?;
        Some(String::from_utf8(stats_csv).map_err(|error| error.to_string())?)
    };
    let limit = preview_limit(request.preview_limit);

    Ok(serde_json::json!({
        "sceneCount": scenes.len(),
        "sceneCsvLineCount": scene_csv.lines().count(),
        "sceneCsvPreview": preview_lines(&scene_csv, limit),
        "statsCsvLineCount": stats.as_ref().map(|value| value.lines().count()).unwrap_or(0),
        "statsCsvPreview": stats.as_ref().map(|value| preview_lines(value, limit)).unwrap_or_default(),
        "previewLimit": limit,
        "externalToolsRequired": false
    }))
}

fn html_plan_value(request: HtmlPlanRequest) -> Result<serde_json::Value, String> {
    let scenes = scenes(&request.scenes)?;
    let mut html = Vec::new();
    write_scene_list_html(&mut html, &scenes).map_err(|error| error.to_string())?;
    let html = String::from_utf8(html).map_err(|error| error.to_string())?;
    let limit = preview_limit(request.preview_limit);
    Ok(serde_json::json!({
        "sceneCount": scenes.len(),
        "byteLength": html.len(),
        "lineCount": html.lines().count(),
        "htmlPreview": preview_lines(&html, limit),
        "previewLimit": limit
    }))
}

fn edit_list_plan_value(request: EditListPlanRequest) -> Result<serde_json::Value, String> {
    let scenes = scenes(&request.scenes)?;
    let mut out = Vec::new();
    let format = request.format.to_ascii_lowercase();
    match format.as_str() {
        "edl" => write_scene_list_edl(&mut out, &scenes),
        "fcp7" | "fcp" | "xml" => write_scene_list_fcp7(&mut out, &scenes),
        "fcpx" | "fcpxml" => write_scene_list_fcpx(&mut out, &scenes),
        "otio" | "otiojson" => write_scene_list_otio(&mut out, &scenes),
        "qp" | "qpfile" => write_scene_list_qp(&mut out, &scenes),
        _ => return Err(format!("unsupported edit-list format `{}`", request.format)),
    }
    .map_err(|error| error.to_string())?;
    let rendered = String::from_utf8(out).map_err(|error| error.to_string())?;
    let limit = preview_limit(request.preview_limit);
    Ok(serde_json::json!({
        "title": "Edit-list export preview",
        "message": "Rendered an in-memory scene-list export preview without writing files or invoking external tools.",
        "summary": {
            "format": format,
            "sceneCount": scenes.len(),
            "lineCount": rendered.lines().count(),
            "byteLength": rendered.len(),
            "externalToolsRequired": false
        },
        "result": {
            "preview": preview_lines(&rendered, limit),
            "previewLimit": limit
        }
    }))
}

fn detection_result(request: DetectionResultRequest) -> Result<DetectionResult, String> {
    Ok(DetectionResult {
        scenes: scenes(&request.scenes)?,
        cuts: request
            .cuts
            .into_iter()
            .map(cut)
            .collect::<Result<Vec<_>, _>>()?,
        metrics: metrics_store(&request.metrics)?,
        frames_processed: request.frames_processed,
    })
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

fn cut(request: CutRequest) -> Result<Cut, String> {
    if let Some(score) = request.score {
        if !score.is_finite() {
            return Err("cut score must be finite".to_string());
        }
    }
    Ok(Cut {
        position: frame_position(request.frame_index, request.fps_num, request.fps_den)?,
        detector: Box::leak(request.detector.into_boxed_str()),
        score: request.score,
    })
}

fn metrics_store(requests: &[MetricRowRequest]) -> Result<MetricsStore, String> {
    let mut metrics = MetricsStore::default();
    for row in requests {
        for (key, value) in &row.values {
            if !value.is_finite() {
                return Err(format!("metric `{key}` must be finite"));
            }
            metrics.set_metric(row.frame_index, key, *value);
        }
    }
    Ok(metrics)
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

fn preview_limit(value: Option<usize>) -> usize {
    value
        .filter(|limit| *limit > 0)
        .unwrap_or(DEFAULT_PREVIEW_LIMIT)
        .min(DEFAULT_PREVIEW_LIMIT)
}

fn preview_lines(value: &str, limit: usize) -> Vec<&str> {
    value.lines().take(limit).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_summary_uses_detection_result_json() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.output.reportSummary"),
            input: serde_json::json!({
                "result": {
                    "framesProcessed": 30,
                    "scenes": [{"startFrame": 0, "endFrame": 30}],
                    "cuts": [{"frameIndex": 30, "detector": "content", "score": 0.9}],
                    "metrics": [{"frameIndex": 0, "values": {"content": 0.1}}]
                }
            }),
        })
        .expect("report summary");
        assert_eq!(response.value["sceneCount"], 1);
        assert_eq!(response.value["cutCount"], 1);
        assert!(response.value.get("plan").is_none());
    }

    #[test]
    fn csv_plan_returns_in_memory_preview() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.output.csvPlan"),
            input: serde_json::json!({
                "scenes": [{"startFrame": 0, "endFrame": 30}],
                "metrics": [{"frameIndex": 0, "values": {"content": 0.1}}],
                "previewLimit": 1
            }),
        })
        .expect("csv plan");
        assert_eq!(
            response.value["sceneCsvPreview"].as_array().unwrap().len(),
            1
        );
        assert_eq!(
            response.value["statsCsvPreview"].as_array().unwrap().len(),
            1
        );
    }

    #[test]
    fn html_plan_rejects_invalid_scene() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.output.htmlPlan"),
            input: serde_json::json!({"scenes": [{"startFrame": 2, "endFrame": 1}]}),
        })
        .expect_err("invalid scene");
        assert!(error.contains("endFrame"));
    }

    #[test]
    fn edit_list_plan_returns_structured_preview() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.output.editListPlan"),
            input: serde_json::json!({
                "format": "otio",
                "scenes": [{"startFrame": 0, "endFrame": 30}],
                "previewLimit": 3
            }),
        })
        .expect("edit-list plan");

        assert_eq!(response.value["title"], "Edit-list export preview");
        assert_eq!(response.value["summary"]["format"], "otio");
        assert!(
            response.value["result"]["preview"]
                .as_array()
                .unwrap()
                .len()
                <= 3
        );
    }
}
