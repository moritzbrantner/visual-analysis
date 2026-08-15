//! Library-owned runtime surface for `image-analysis-synthesis`.

use runtime_core::{
    describe_surface_response, structured_operation_response, OperationId, PackageSurface,
    RuntimeCapabilities, SurfaceOperation, SurfaceRequest, SurfaceResponse,
};
use serde::Deserialize;

use image_analysis_core::{ImagePixelFormat, OwnedImage};

use crate::{
    image_from_luma_histogram, solid_image, vertical_gradient, ImageSynthesisConfig, RgbColor,
};

/// Returns the package surface exposed by every transport wrapper.
pub fn package_surface() -> PackageSurface {
    PackageSurface {
        library: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities: RuntimeCapabilities::pure_rust().with_max_recommended_input_bytes(64 * 1024),
        operations: vec![
            operation(
                "describe",
                "Describe package",
                "Deterministic image synthesis helpers returning summaries instead of encoded image bytes.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "image.synthesis.solid",
                "Solid image",
                "Creates a deterministic solid image and returns dimensions, color statistics, and trace summary.",
                serde_json::json!({"width": 4, "height": 2, "color": {"red": 10, "green": 20, "blue": 30}}),
            ),
            operation(
                "image.synthesis.gradient",
                "Vertical gradient",
                "Creates a deterministic vertical gradient and returns dimensions, color statistics, and trace summary.",
                serde_json::json!({"width": 4, "height": 2, "top": {"red": 0, "green": 0, "blue": 0}, "bottom": {"red": 255, "green": 255, "blue": 255}}),
            ),
            operation(
                "image.synthesis.histogram",
                "Histogram image",
                "Creates a deterministic luma histogram image and returns preview statistics and trace summary.",
                serde_json::json!({"width": 4, "height": 1, "pixelFormat": "gray8", "histogram": [1, 3]}),
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
        "image.synthesis.solid" => solid_value(parse_input(request.input)?)?,
        "image.synthesis.gradient" => gradient_value(parse_input(request.input)?)?,
        "image.synthesis.histogram" => histogram_value(parse_input(request.input)?)?,
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
struct SolidRequest {
    #[serde(default = "default_width")]
    width: u32,
    #[serde(default = "default_height")]
    height: u32,
    #[serde(default = "default_pixel_format")]
    pixel_format: String,
    color: ColorRequest,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GradientRequest {
    #[serde(default = "default_width")]
    width: u32,
    #[serde(default = "default_height")]
    height: u32,
    #[serde(default = "default_pixel_format")]
    pixel_format: String,
    top: ColorRequest,
    bottom: ColorRequest,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct HistogramRequest {
    #[serde(default = "default_width")]
    width: u32,
    #[serde(default = "default_height")]
    height: u32,
    #[serde(default = "default_gray_pixel_format")]
    pixel_format: String,
    histogram: Vec<u64>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ColorRequest {
    red: u8,
    green: u8,
    blue: u8,
}

fn solid_value(request: SolidRequest) -> Result<serde_json::Value, String> {
    let color = request.color.rgb();
    let generated = solid_image(
        color,
        config(request.width, request.height, &request.pixel_format)?,
    )
    .map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "dimensions": dimensions_json(&generated.value),
        "pixelFormat": pixel_format_name(generated.value.pixel_format),
        "color": color_json(color),
        "meanColor": mean_color_json(&generated.value),
        "luma": color.luma(),
        "trace": trace_json(&generated.trace)
    }))
}

fn gradient_value(request: GradientRequest) -> Result<serde_json::Value, String> {
    let top = request.top.rgb();
    let bottom = request.bottom.rgb();
    let generated = vertical_gradient(
        top,
        bottom,
        config(request.width, request.height, &request.pixel_format)?,
    )
    .map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "dimensions": dimensions_json(&generated.value),
        "pixelFormat": pixel_format_name(generated.value.pixel_format),
        "topColor": color_json(top),
        "bottomColor": color_json(bottom),
        "meanColor": mean_color_json(&generated.value),
        "trace": trace_json(&generated.trace)
    }))
}

fn histogram_value(request: HistogramRequest) -> Result<serde_json::Value, String> {
    let generated = image_from_luma_histogram(
        &request.histogram,
        config(request.width, request.height, &request.pixel_format)?,
    )
    .map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "dimensions": dimensions_json(&generated.value),
        "pixelFormat": pixel_format_name(generated.value.pixel_format),
        "histogramBins": request.histogram.len(),
        "preview": preview_stats_json(&generated.value),
        "meanColor": mean_color_json(&generated.value),
        "trace": trace_json(&generated.trace)
    }))
}

fn config(width: u32, height: u32, pixel_format: &str) -> Result<ImageSynthesisConfig, String> {
    ImageSynthesisConfig::new(width, height, parse_pixel_format(pixel_format)?)
        .map_err(|error| error.to_string())
}

fn parse_pixel_format(value: &str) -> Result<ImagePixelFormat, String> {
    match value {
        "rgb24" => Ok(ImagePixelFormat::Rgb24),
        "bgr24" => Ok(ImagePixelFormat::Bgr24),
        "gray8" => Ok(ImagePixelFormat::Gray8),
        other => Err(format!("unsupported image pixel format `{other}`")),
    }
}

fn dimensions_json(image: &OwnedImage) -> serde_json::Value {
    serde_json::json!({"width": image.width, "height": image.height})
}

fn color_json(color: RgbColor) -> serde_json::Value {
    serde_json::json!({"red": color.red, "green": color.green, "blue": color.blue})
}

fn trace_json(trace: &data_inversion_core::InversionTrace) -> serde_json::Value {
    serde_json::json!({
        "sourceType": trace.source_type,
        "targetType": trace.target_type,
        "fidelity": format!("{:?}", trace.fidelity),
        "confidence": trace.confidence,
        "assumptions": trace.assumptions,
        "notes": trace.notes.iter().map(|note| serde_json::json!({
            "field": note.field,
            "method": format!("{:?}", note.method),
            "message": note.message
        })).collect::<Vec<_>>()
    })
}

fn mean_color_json(image: &OwnedImage) -> serde_json::Value {
    let mut red = 0_u64;
    let mut green = 0_u64;
    let mut blue = 0_u64;
    let pixels = (image.width as u64 * image.height as u64).max(1);
    match image.pixel_format {
        ImagePixelFormat::Rgb24 => {
            for chunk in image.data.chunks_exact(3) {
                red += chunk[0] as u64;
                green += chunk[1] as u64;
                blue += chunk[2] as u64;
            }
        }
        ImagePixelFormat::Bgr24 => {
            for chunk in image.data.chunks_exact(3) {
                blue += chunk[0] as u64;
                green += chunk[1] as u64;
                red += chunk[2] as u64;
            }
        }
        ImagePixelFormat::Gray8 => {
            for value in &image.data {
                red += *value as u64;
                green += *value as u64;
                blue += *value as u64;
            }
        }
    }
    serde_json::json!({
        "red": red as f64 / pixels as f64,
        "green": green as f64 / pixels as f64,
        "blue": blue as f64 / pixels as f64
    })
}

fn preview_stats_json(image: &OwnedImage) -> serde_json::Value {
    let preview = image.data.iter().take(16).copied().collect::<Vec<_>>();
    let min = image.data.iter().copied().min();
    let max = image.data.iter().copied().max();
    serde_json::json!({
        "byteLength": image.data.len(),
        "previewBytes": preview,
        "minByte": min,
        "maxByte": max
    })
}

fn pixel_format_name(format: ImagePixelFormat) -> &'static str {
    match format {
        ImagePixelFormat::Rgb24 => "rgb24",
        ImagePixelFormat::Bgr24 => "bgr24",
        ImagePixelFormat::Gray8 => "gray8",
    }
}

impl ColorRequest {
    fn rgb(self) -> RgbColor {
        RgbColor::new(self.red, self.green, self.blue)
    }
}

fn parse_input<T: for<'de> Deserialize<'de>>(input: serde_json::Value) -> Result<T, String> {
    serde_json::from_value(input).map_err(|error| format!("invalid request: {error}"))
}

fn default_width() -> u32 {
    512
}

fn default_height() -> u32 {
    512
}

fn default_pixel_format() -> String {
    "rgb24".to_string()
}

fn default_gray_pixel_format() -> String {
    "gray8".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_surface_lists_synthesis_operations() {
        let ids = package_surface()
            .operations
            .into_iter()
            .map(|operation| operation.id.0)
            .collect::<Vec<_>>();
        assert!(ids.contains(&"image.synthesis.solid".to_string()));
        assert!(ids.contains(&"image.synthesis.histogram".to_string()));
    }

    #[test]
    fn solid_operation_returns_summary_not_bytes() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.synthesis.solid"),
            input: serde_json::json!({"width": 2, "height": 1, "color": {"red": 10, "green": 20, "blue": 30}}),
        })
        .expect("solid");
        assert_eq!(response.value["dimensions"]["width"], 2);
        assert_eq!(response.value["meanColor"]["green"], 20.0);
        assert!(response.value.get("data").is_none());
    }

    #[test]
    fn histogram_rejects_empty_histogram() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("image.synthesis.histogram"),
            input: serde_json::json!({"width": 2, "height": 1, "histogram": []}),
        })
        .expect_err("empty histogram");
        assert!(error.contains("histogram bins"));
    }
}
