//! Library-owned runtime surface for `image-analysis-segmentation`.

use runtime_core::{
    describe_surface_response, structured_operation_response, OperationId, PackageSurface,
    RuntimeCapabilities, SurfaceOperation, SurfaceRequest, SurfaceResponse,
};
use serde::Deserialize;
use video_analysis_core::BoundingBox;

use crate::{
    default_sam_model_spec, BinaryMask, ImageSegmentationPrompt, ImageSegmentationRequest,
    SegmentationPoint,
};

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
                "Image segmentation model metadata, prompt validation, and imported mask summaries.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "image.segmentation.model",
                "SAM model",
                "Returns the default SAM model spec without downloading or running SAM.",
                serde_json::json!({}),
            ),
            operation(
                "image.segmentation.promptSummary",
                "Prompt summary",
                "Validates segmentation prompt/request-shaped JSON and summarizes prompt controls.",
                serde_json::json!({"points": [{"x": 1, "y": 2, "label": "foreground"}], "boxes": [{"x": 0, "y": 0, "width": 8, "height": 8}], "multimaskOutput": true, "minMaskPixels": 4}),
            ),
            operation(
                "image.segmentation.maskSummary",
                "Mask summary",
                "Builds a binary mask from raw bytes or a rectangle and returns active pixels and bounds.",
                serde_json::json!({"width": 4, "height": 3, "rect": {"x": 1, "y": 1, "width": 2, "height": 1}}),
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
    let surface = package_surface();
    let operation = request.operation.clone();
    let value = match request.operation.as_str() {
        "describe" => return Ok(describe_surface_response(&surface, request)),
        "image.segmentation.model" => model_value(),
        "image.segmentation.promptSummary" => prompt_summary_value(parse_input(request.input)?)?,
        "image.segmentation.maskSummary" => mask_summary_value(parse_input(request.input)?)?,
        operation => {
            return Err(format!(
                "unsupported operation `{operation}` for {}",
                env!("CARGO_PKG_NAME")
            ));
        }
    };
    Ok(structured_operation_response(&surface, operation, value))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PromptSummaryInput {
    #[serde(default)]
    prompt: Option<PromptInput>,
    #[serde(default)]
    points: Vec<PointInput>,
    #[serde(default)]
    boxes: Vec<BoxInput>,
    #[serde(default)]
    automatic_mask_generation: bool,
    #[serde(default)]
    multimask_output: bool,
    #[serde(default = "default_min_mask_pixels")]
    min_mask_pixels: usize,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PromptInput {
    #[serde(default)]
    points: Vec<PointInput>,
    #[serde(default)]
    boxes: Vec<BoxInput>,
    #[serde(default)]
    automatic_mask_generation: bool,
    #[serde(default)]
    multimask_output: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PointInput {
    x: u32,
    y: u32,
    #[serde(default = "default_point_label")]
    label: String,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BoxInput {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MaskInput {
    width: u32,
    height: u32,
    #[serde(default)]
    data: Option<Vec<u8>>,
    #[serde(default)]
    rect: Option<BoxInput>,
}

fn model_value() -> serde_json::Value {
    let spec = default_sam_model_spec();
    serde_json::json!({
        "model": {
            "name": spec.name,
            "repoId": spec.repo_id_value(),
            "revision": spec.revision_value(),
            "task": spec.task.as_protocol_str(),
            "files": spec.files
        }
    })
}

fn prompt_summary_value(input: PromptSummaryInput) -> Result<serde_json::Value, String> {
    let merged_prompt = input.prompt.unwrap_or(PromptInput {
        points: input.points,
        boxes: input.boxes,
        automatic_mask_generation: input.automatic_mask_generation,
        multimask_output: input.multimask_output,
    });
    let mut prompt = ImageSegmentationPrompt::new();
    prompt.automatic_mask_generation = merged_prompt.automatic_mask_generation;
    prompt.multimask_output = merged_prompt.multimask_output;
    for point in merged_prompt.points {
        prompt = prompt.point(match point.label.as_str() {
            "foreground" | "fg" | "positive" => SegmentationPoint::foreground(point.x, point.y),
            "background" | "bg" | "negative" => SegmentationPoint::background(point.x, point.y),
            other => return Err(format!("unsupported point label `{other}`")),
        });
    }
    for region in merged_prompt.boxes {
        prompt = prompt.bounding_box(region.bounding_box()?);
    }
    let request = ImageSegmentationRequest::new(prompt).min_mask_pixels(input.min_mask_pixels);
    Ok(serde_json::json!({
        "pointCount": request.prompt.points.len(),
        "boxCount": request.prompt.boxes.len(),
        "hasBox": !request.prompt.boxes.is_empty(),
        "automaticMaskGeneration": request.prompt.automatic_mask_generation,
        "multimaskOutput": request.prompt.multimask_output,
        "minMaskPixels": request.min_mask_pixels
    }))
}

fn mask_summary_value(input: MaskInput) -> Result<serde_json::Value, String> {
    let mask = if let Some(rect) = input.rect {
        BinaryMask::filled_rect(input.width, input.height, rect.bounding_box()?)
    } else {
        BinaryMask::new(
            input.width,
            input.height,
            input
                .data
                .ok_or_else(|| "mask data or rect is required".to_string())?,
        )
    }
    .map_err(|error| error.to_string())?;
    let bounds = mask.bounding_box();
    Ok(serde_json::json!({
        "width": mask.width,
        "height": mask.height,
        "activePixels": mask.active_pixels(),
        "boundingBox": bounds.map(|region| serde_json::json!({
            "x": region.x,
            "y": region.y,
            "width": region.width,
            "height": region.height
        }))
    }))
}

impl BoxInput {
    fn bounding_box(self) -> Result<BoundingBox, String> {
        BoundingBox::new(self.x, self.y, self.width, self.height).map_err(|error| error.to_string())
    }
}

fn parse_input<T: for<'de> Deserialize<'de>>(input: serde_json::Value) -> Result<T, String> {
    serde_json::from_value(input).map_err(|error| format!("invalid request: {error}"))
}

fn default_min_mask_pixels() -> usize {
    1
}

fn default_point_label() -> String {
    "foreground".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_surface_lists_segmentation_operations() {
        let ids = package_surface()
            .operations
            .into_iter()
            .map(|operation| operation.id.0)
            .collect::<Vec<_>>();
        assert!(ids.contains(&"image.segmentation.model".to_string()));
        assert!(ids.contains(&"image.segmentation.maskSummary".to_string()));
    }

    #[test]
    fn mask_summary_builds_rect_mask() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.segmentation.maskSummary"),
            input: serde_json::json!({"width": 4, "height": 3, "rect": {"x": 1, "y": 1, "width": 2, "height": 1}}),
        })
        .expect("mask summary");
        assert_eq!(response.value["activePixels"], 2);
        assert_eq!(response.value["boundingBox"]["x"], 1);
    }

    #[test]
    fn prompt_summary_rejects_unknown_point_label() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.segmentation.promptSummary"),
            input: serde_json::json!({"points": [{"x": 1, "y": 2, "label": "maybe"}]}),
        })
        .expect_err("bad point label");
        assert!(error.contains("unsupported point label"));
    }
}
