//! Library-owned runtime surface for `video-analysis-editing`.

use runtime_core::{
    describe_surface_response, structured_operation_response, surface_operation, PackageSurface,
    RuntimeCapabilities, SurfaceRequest, SurfaceResponse,
};

use crate::operations::{
    concat_plan_value, cut_plan_value, deterministic_operation_value, frame_apply_value,
    sample_frame_json, subtitle_plan_value,
};

/// Returns the package surface exposed by every transport wrapper.
pub fn package_surface() -> PackageSurface {
    PackageSurface {
        library: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities: RuntimeCapabilities::pure_rust(),
        operations: vec![
            surface_operation(
                "describe",
                "Describe package",
                "Deterministic edit decision contracts for video-analysis outputs.",
                serde_json::json!({"includeOperations": true}),
            ),
            surface_operation(
                "video.editing.cutPlan",
                "Plan cuts",
                "Builds a deterministic edit decision list from scene and cut intervals.",
                serde_json::json!({"input": {}, "mode": "deterministic"}),
            ),
            surface_operation(
                "video.editing.concatPlan",
                "Plan concat",
                "Describes segment ordering, transitions, and output stream compatibility for concatenation.",
                serde_json::json!({"input": {}, "mode": "deterministic"}),
            ),
            surface_operation(
                "video.editing.subtitlePlan",
                "Plan subtitles",
                "Builds a subtitle overlay plan from transcript and scene timing records.",
                serde_json::json!({"input": {}, "mode": "deterministic"}),
            ),
            surface_operation(
                "video.editing.frameApply",
                "Apply frame edit",
                "Applies an in-memory deterministic frame edit pipeline and returns a capped output preview.",
                serde_json::json!({
                    "frame": sample_frame_json(),
                    "edits": [{"type": "flipHorizontal"}, {"type": "fadeToColor", "color": [0, 0, 0], "amount": 0.5}]
                }),
            ),
        ],
    }
}

/// Runs one library-owned operation.
pub fn run_surface_operation(request: SurfaceRequest) -> Result<SurfaceResponse, String> {
    let surface = package_surface();
    let operation = request.operation.clone();
    let Some(surface_operation) = surface
        .operations
        .iter()
        .find(|candidate| candidate.id.as_str() == operation.as_str())
    else {
        return Err(format!(
            "unsupported operation `{}` for {}",
            operation.as_str(),
            env!("CARGO_PKG_NAME")
        ));
    };

    let value = match operation.as_str() {
        "describe" => return Ok(describe_surface_response(&surface, request)),
        "video.editing.cutPlan" => cut_plan_value(request.input)?,
        "video.editing.concatPlan" => concat_plan_value(request.input)?,
        "video.editing.subtitlePlan" => subtitle_plan_value(request.input)?,
        "video.editing.frameApply" => frame_apply_value(request.input)?,
        _ => deterministic_operation_value(&surface, surface_operation, request.input),
    };

    Ok(structured_operation_response(&surface, operation, value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_core::OperationId;

    #[test]
    fn package_surface_has_multiple_operations() {
        let surface = package_surface();
        assert_eq!(surface.library, env!("CARGO_PKG_NAME"));
        assert!(surface
            .operations
            .iter()
            .any(|operation| operation.id.as_str() == "describe"));
        assert!(surface.operations.len() >= 4);
    }

    #[test]
    fn describe_operation_returns_surface_summary() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("describe"),
            input: serde_json::json!({"includeOperations": true}),
        })
        .expect("describe operation");

        assert_eq!(response.operation.as_str(), "describe");
        assert_eq!(response.value["library"], env!("CARGO_PKG_NAME"));
        assert!(response.value["operationCount"].as_u64().unwrap() >= 4);
    }

    #[test]
    fn package_operation_returns_deterministic_plan() {
        let operation_id = package_surface().operations[1].id.clone();
        let response = run_surface_operation(SurfaceRequest {
            operation: operation_id,
            input: serde_json::json!({"sample": true}),
        })
        .expect("package operation");

        assert_eq!(response.value["deterministic"], true);
        assert_eq!(response.value["externalToolsRequired"], false);
    }

    #[test]
    fn edit_plan_operations_return_real_edl_summaries() {
        let cut = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.editing.cutPlan"),
            input: serde_json::json!({
                "sourceId": "clip-a",
                "intervals": [
                    {"startSeconds": 0.0, "endSeconds": 1.0},
                    {"startSeconds": 2.0, "endSeconds": 3.0}
                ]
            }),
        })
        .expect("cut plan");
        assert_eq!(cut.value["clipCount"], 2);
        assert_eq!(cut.value["durationSeconds"], 2.0);

        let frame = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.editing.frameApply"),
            input: serde_json::json!({
                "frame": sample_frame_json(),
                "edits": [{"type": "flipHorizontal"}, {"type": "fadeToColor", "color": [0, 0, 0], "amount": 0.5}],
                "previewLimit": 3
            }),
        })
        .expect("frame apply");
        assert_eq!(frame.value["width"], 2);
    }
}
