//! ONNX object detection: runners, the detector, and output decoding.

use std::collections::BTreeMap;

use image_analysis_core::ImageView;
use image_analysis_processing::{
    preprocess_image_for_model, validate_image_model_preprocessing, ImageModelPreprocessing,
    ImageModelTensor,
};
use video_analysis_core::{BoundingBox, DetectError, Result};

use crate::onnx_bundle::validate_threshold;
#[cfg(feature = "onnx")]
use crate::onnx_bundle::runtime_onnx_error;
#[cfg(feature = "onnx")]
use crate::validate_onnx_vision_bundle;
use crate::{object_detection_options_from_bundle, ImageDetection};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Variants describing box format.
pub enum BoxFormat {
    /// The xyxy absolute variant.
    XyxyAbsolute,
    /// The xyxy normalized variant.
    XyxyNormalized,
    /// The cxcywh absolute variant.
    CxcywhAbsolute,
    /// The cxcywh normalized variant.
    CxcywhNormalized,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for ONNX object detection options.
pub struct OnnxObjectDetectionOptions {
    /// The preprocessing value.
    pub preprocessing: ImageModelPreprocessing,
    /// The score threshold value.
    pub score_threshold: f32,
    /// The labels value.
    pub labels: BTreeMap<i64, String>,
    /// The box format value.
    pub box_format: BoxFormat,
}

impl Default for OnnxObjectDetectionOptions {
    fn default() -> Self {
        Self {
            preprocessing: ImageModelPreprocessing::default(),
            score_threshold: 0.0,
            labels: BTreeMap::new(),
            box_format: BoxFormat::CxcywhNormalized,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for ONNX object detection output.
pub struct OnnxObjectDetectionOutput {
    /// The boxes value.
    pub boxes: Vec<[f32; 4]>,
    /// The class identifiers value.
    pub class_ids: Vec<i64>,
    /// The scores value.
    pub scores: Vec<f32>,
    /// The box format value.
    pub box_format: BoxFormat,
}

/// Trait for ONNX object detection runner implementations.
pub trait OnnxObjectDetectionRunner {
    /// Runs object detection.
    fn run_object_detection(
        &mut self,
        input: &ImageModelTensor,
    ) -> Result<OnnxObjectDetectionOutput>;
}

#[derive(Debug, Clone, Default)]
/// Data type for unavailable ONNX runner.
pub struct UnavailableOnnxRunner;

impl OnnxObjectDetectionRunner for UnavailableOnnxRunner {
    fn run_object_detection(
        &mut self,
        _input: &ImageModelTensor,
    ) -> Result<OnnxObjectDetectionOutput> {
        Err(DetectError::Source(
            "native ONNX image detection execution is unavailable; construct with a runner or enable `onnx`"
                .to_string(),
        ))
    }
}

#[cfg(feature = "onnx")]
impl OnnxObjectDetectionRunner for runtime_onnx::OnnxSession {
    fn run_object_detection(
        &mut self,
        input: &ImageModelTensor,
    ) -> Result<OnnxObjectDetectionOutput> {
        use runtime_onnx::{OnnxRunner, OnnxTensor, OnnxTensorValue};

        let metadata = self.metadata().map_err(runtime_onnx_error)?;
        let input_name = runtime_onnx::input_name(&metadata, 0)
            .map_err(runtime_onnx_error)?
            .to_string();
        let mut inputs = vec![runtime_onnx::OnnxNamedTensor {
            name: input_name,
            tensor: OnnxTensorValue::F32(
                OnnxTensor::new(
                    vec![
                        1,
                        input.channels,
                        input.height as usize,
                        input.width as usize,
                    ],
                    input.values.clone(),
                )
                .map_err(runtime_onnx_error)?,
            ),
        }];
        if let Some(mask_input) = metadata
            .inputs
            .iter()
            .find(|input| input.name == "pixel_mask")
        {
            inputs.push(pixel_mask_input(mask_input, input)?);
        }
        let outputs = self.run(inputs).map_err(runtime_onnx_error)?;
        if outputs.len() < 2 {
            return Err(DetectError::InvalidArgument(
                "ONNX object detection model must return logits and boxes".to_string(),
            ));
        }
        let first = runtime_onnx::first_f32_output(&outputs).map_err(runtime_onnx_error)?;
        let second = runtime_onnx::f32_output_by_name_or_index(&outputs, "", 1)
            .map_err(runtime_onnx_error)?;
        decode_detr_like_outputs(first, second)
    }
}

#[cfg(feature = "onnx")]
fn pixel_mask_input(
    input_info: &runtime_onnx::OnnxIoInfo,
    input: &ImageModelTensor,
) -> Result<runtime_onnx::OnnxNamedTensor> {
    let height = fixed_or_image_dimension(input_info.dimensions.get(1), input.height as usize);
    let width = fixed_or_image_dimension(input_info.dimensions.get(2), input.width as usize);
    let values = vec![1_i64; height * width];
    Ok(runtime_onnx::OnnxNamedTensor {
        name: input_info.name.clone(),
        tensor: runtime_onnx::OnnxTensorValue::I64(
            runtime_onnx::OnnxTensor::new(vec![1, height, width], values)
                .map_err(runtime_onnx_error)?,
        ),
    })
}

#[cfg(feature = "onnx")]
fn fixed_or_image_dimension(
    dimension: Option<&runtime_onnx::OnnxDimension>,
    image_dimension: usize,
) -> usize {
    match dimension {
        Some(runtime_onnx::OnnxDimension::Fixed(value)) => *value,
        _ => image_dimension,
    }
}

#[derive(Debug)]
/// Data type for ONNX object detector.
pub struct OnnxObjectDetector<R = UnavailableOnnxRunner> {
    preprocessing: ImageModelPreprocessing,
    pub(crate) score_threshold: f32,
    labels: BTreeMap<i64, String>,
    box_format: BoxFormat,
    runner: R,
}

#[cfg(not(feature = "onnx"))]
impl OnnxObjectDetector<UnavailableOnnxRunner> {
    /// Builds this value from bundle.
    pub fn from_bundle(bundle: model_runtime::ModelBundle) -> Result<Self> {
        Self::from_runner(bundle, UnavailableOnnxRunner)
    }
}

#[cfg(feature = "onnx")]
impl OnnxObjectDetector<runtime_onnx::OnnxSession> {
    /// Builds this value from bundle.
    pub fn from_bundle(bundle: model_runtime::ModelBundle) -> Result<Self> {
        let info = validate_onnx_vision_bundle(&bundle)?;
        let options = object_detection_options_from_bundle(&bundle)?;
        Ok(Self {
            preprocessing: options.preprocessing,
            score_threshold: options.score_threshold,
            labels: options.labels,
            box_format: options.box_format,
            runner: runtime_onnx::OnnxSession::from_file(info.model_path)
                .map_err(runtime_onnx_error)?,
        })
    }
}

impl<R: OnnxObjectDetectionRunner> OnnxObjectDetector<R> {
    /// Builds this value from runner.
    pub fn from_runner(bundle: model_runtime::ModelBundle, runner: R) -> Result<Self> {
        let options = object_detection_options_from_bundle(&bundle)?;
        Ok(Self {
            preprocessing: options.preprocessing,
            score_threshold: options.score_threshold,
            labels: options.labels,
            box_format: options.box_format,
            runner,
        })
    }

    /// Returns this value with options.
    pub fn with_options(options: OnnxObjectDetectionOptions, runner: R) -> Result<Self> {
        validate_image_model_preprocessing(&options.preprocessing)?;
        validate_threshold(options.score_threshold)?;
        Ok(Self {
            preprocessing: options.preprocessing,
            score_threshold: options.score_threshold,
            labels: options.labels,
            box_format: options.box_format,
            runner,
        })
    }

    /// Returns preprocessing.
    pub fn preprocessing(&self) -> &ImageModelPreprocessing {
        &self.preprocessing
    }

    /// Returns labels.
    pub fn labels(&self) -> &BTreeMap<i64, String> {
        &self.labels
    }

    /// Returns detect image.
    pub fn detect_image(&mut self, image: &ImageView<'_>) -> Result<Vec<ImageDetection>> {
        let input = preprocess_image_for_model(image, &self.preprocessing)?;
        let mut output = self.runner.run_object_detection(&input)?;
        if output.box_format != self.box_format {
            output.box_format = self.box_format;
        }
        Ok(decode_object_detections(
            &output,
            &self.labels,
            self.score_threshold,
            (image.width, image.height),
        ))
    }
}

/// Returns decode object detections.
pub fn decode_object_detections(
    output: &OnnxObjectDetectionOutput,
    labels: &BTreeMap<i64, String>,
    score_threshold: f32,
    original_size: (u32, u32),
) -> Vec<ImageDetection> {
    output
        .boxes
        .iter()
        .zip(output.class_ids.iter())
        .zip(output.scores.iter())
        .filter_map(|((bbox, class_id), score)| {
            if !score.is_finite() || *score < score_threshold {
                return None;
            }
            let region = decode_bounding_box(*bbox, output.box_format, original_size)?;
            Some(ImageDetection {
                label: labels
                    .get(class_id)
                    .cloned()
                    .unwrap_or_else(|| format!("LABEL_{class_id}")),
                score: Some(*score),
                region,
                attributes: BTreeMap::new(),
            })
        })
        .collect()
}

#[cfg(feature = "onnx")]
fn decode_detr_like_outputs(
    first: &runtime_onnx::OnnxF32Tensor,
    second: &runtime_onnx::OnnxF32Tensor,
) -> Result<OnnxObjectDetectionOutput> {
    let (boxes, logits) = match (is_box_output(first), is_box_output(second)) {
        (true, false) => (first, second),
        (false, true) => (second, first),
        _ => {
            return Err(DetectError::InvalidArgument(
                "ONNX object detection outputs must include one box tensor ending in 4 values and one logits tensor".to_string(),
            ))
        }
    };
    let query_count = output_query_count(&boxes.shape)?;
    let class_count = output_last_dim(&logits.shape)?;
    if output_query_count(&logits.shape)? != query_count {
        return Err(DetectError::InvalidArgument(
            "ONNX logits and boxes output query counts differ".to_string(),
        ));
    }
    if boxes.values.len() != query_count * 4 || logits.values.len() != query_count * class_count {
        return Err(DetectError::InvalidArgument(
            "ONNX output tensor shapes do not match their values".to_string(),
        ));
    }
    if class_count < 2 {
        return Err(DetectError::InvalidArgument(
            "ONNX object detection logits must include at least one class plus background"
                .to_string(),
        ));
    }

    let mut decoded_boxes = Vec::with_capacity(query_count);
    let mut class_ids = Vec::with_capacity(query_count);
    let mut scores = Vec::with_capacity(query_count);
    for query in 0..query_count {
        decoded_boxes.push([
            boxes.values[query * 4],
            boxes.values[query * 4 + 1],
            boxes.values[query * 4 + 2],
            boxes.values[query * 4 + 3],
        ]);
        let start = query * class_count;
        let probabilities = softmax(&logits.values[start..start + class_count]);
        let (class_index, score) = probabilities[..class_count - 1]
            .iter()
            .copied()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .ok_or_else(|| {
                DetectError::InvalidArgument(
                    "ONNX logits contain no foreground classes".to_string(),
                )
            })?;
        class_ids.push(class_index as i64);
        scores.push(score);
    }

    Ok(OnnxObjectDetectionOutput {
        boxes: decoded_boxes,
        class_ids,
        scores,
        box_format: BoxFormat::CxcywhNormalized,
    })
}

fn decode_bounding_box(
    bbox: [f32; 4],
    format: BoxFormat,
    original_size: (u32, u32),
) -> Option<BoundingBox> {
    if bbox.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let (width, height) = (original_size.0 as f32, original_size.1 as f32);
    let (mut xmin, mut ymin, mut xmax, mut ymax) = match format {
        BoxFormat::XyxyAbsolute => (bbox[0], bbox[1], bbox[2], bbox[3]),
        BoxFormat::XyxyNormalized => (
            bbox[0] * width,
            bbox[1] * height,
            bbox[2] * width,
            bbox[3] * height,
        ),
        BoxFormat::CxcywhAbsolute => {
            let half_width = bbox[2] / 2.0;
            let half_height = bbox[3] / 2.0;
            (
                bbox[0] - half_width,
                bbox[1] - half_height,
                bbox[0] + half_width,
                bbox[1] + half_height,
            )
        }
        BoxFormat::CxcywhNormalized => {
            let half_width = bbox[2] * width / 2.0;
            let half_height = bbox[3] * height / 2.0;
            (
                bbox[0] * width - half_width,
                bbox[1] * height - half_height,
                bbox[0] * width + half_width,
                bbox[1] * height + half_height,
            )
        }
    };
    xmin = xmin.clamp(0.0, width);
    xmax = xmax.clamp(0.0, width);
    ymin = ymin.clamp(0.0, height);
    ymax = ymax.clamp(0.0, height);
    if xmax <= xmin || ymax <= ymin {
        return None;
    }
    let epsilon = 0.0001;
    let x = (xmin + epsilon).floor().max(0.0) as u32;
    let y = (ymin + epsilon).floor().max(0.0) as u32;
    let box_width = ((xmax - epsilon).ceil() as u32).saturating_sub(x);
    let box_height = ((ymax - epsilon).ceil() as u32).saturating_sub(y);
    BoundingBox::new(x, y, box_width, box_height).ok()
}

#[cfg(feature = "onnx")]
fn is_box_output(output: &runtime_onnx::OnnxF32Tensor) -> bool {
    matches!(output_last_dim(&output.shape), Ok(4))
}

#[cfg(feature = "onnx")]
fn output_query_count(shape: &[usize]) -> Result<usize> {
    match shape {
        [queries, _] => Ok(*queries),
        [1, queries, _] => Ok(*queries),
        _ => Err(DetectError::InvalidArgument(format!(
            "unsupported ONNX output shape `{shape:?}`"
        ))),
    }
}

#[cfg(feature = "onnx")]
fn output_last_dim(shape: &[usize]) -> Result<usize> {
    match shape {
        [_, last] | [1, _, last] => Ok(*last),
        _ => Err(DetectError::InvalidArgument(format!(
            "unsupported ONNX output shape `{shape:?}`"
        ))),
    }
}

#[cfg(feature = "onnx")]
fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities = logits
        .iter()
        .map(|value| (value - max).exp())
        .collect::<Vec<_>>();
    let total = probabilities.iter().sum::<f32>();
    if total <= f32::EPSILON || !total.is_finite() {
        return vec![0.0; logits.len()];
    }
    for probability in &mut probabilities {
        *probability /= total;
    }
    probabilities
}
