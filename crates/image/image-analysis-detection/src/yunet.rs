//! YuNet face detection output decoding for its tensor layouts.

use image_analysis_processing::ImageModelPreprocessing;
use video_analysis_core::{BoundingBox, DetectError, Result};

use crate::{
    FaceBox, FaceDetection, FaceLandmarks, OnnxFaceDetectionOptions, OnnxFaceDetectionOutput,
};

/// Returns decoded YuNet face detections from ONNX output tensors.
pub fn decode_yunet_face_detections(
    output: &OnnxFaceDetectionOutput,
    options: &OnnxFaceDetectionOptions,
    original_size: (u32, u32),
) -> Result<Vec<FaceDetection>> {
    let modern_layout = output.tensors.keys().any(|name| {
        ["cls_", "obj_", "bbox_", "kps_"]
            .iter()
            .any(|prefix| name.starts_with(*prefix))
    });
    let mut detections = if modern_layout {
        decode_yunet_2023_tensors(output, options, original_size)?
    } else if let Some(tensor) = output
        .tensors
        .values()
        .find(|tensor| output_last_dim_any(&tensor.shape) == Some(15))
    {
        decode_yunet_single_tensor(tensor, options, original_size)?
    } else {
        decode_yunet_split_tensors(output, options, original_size)?
    };
    detections.sort_by(|left, right| right.confidence.total_cmp(&left.confidence));
    if detections.len() > options.top_k {
        detections.truncate(options.top_k);
    }
    Ok(nms_face_detections(detections, options.nms_threshold))
}

fn decode_yunet_single_tensor(
    tensor: &runtime_onnx::OnnxF32Tensor,
    options: &OnnxFaceDetectionOptions,
    original_size: (u32, u32),
) -> Result<Vec<FaceDetection>> {
    let rows = yunet_row_count(&tensor.shape, 15)?;
    let mut detections = Vec::new();
    for row in 0..rows {
        let start = row * 15;
        let score = tensor.values[start + 4];
        if !score.is_finite() || score < options.score_threshold {
            continue;
        }
        let bbox = tensor.values[start..start + 4].try_into().unwrap();
        if let Some(detection) = face_detection_from_xyxy(
            bbox,
            score,
            &tensor.values[start + 5..start + 15],
            options,
            original_size,
        )? {
            detections.push(detection);
        }
    }
    Ok(detections)
}

// YuNet 2023 uses anchor-free grid offsets, not the legacy loc/conf/iou priors.
// Contract: OpenCV modules/objdetect/src/face_detect.cpp, FaceDetectorYNImpl::postProcess.
fn decode_yunet_2023_tensors(
    output: &OnnxFaceDetectionOutput,
    options: &OnnxFaceDetectionOptions,
    original_size: (u32, u32),
) -> Result<Vec<FaceDetection>> {
    let width = options.preprocessing.input_width;
    let height = options.preprocessing.input_height;
    if width == 0 || height == 0 || !width.is_multiple_of(32) || !height.is_multiple_of(32) {
        return Err(DetectError::InvalidArgument(
            "YuNet 2023 input dimensions must be positive multiples of 32".to_string(),
        ));
    }
    let mut detections = Vec::new();
    for stride in [8_u32, 16, 32] {
        let columns = (width / stride) as usize;
        let rows = columns
            .checked_mul((height / stride) as usize)
            .ok_or_else(|| {
                DetectError::InvalidArgument("YuNet output grid size overflows".to_string())
            })?;
        let cls = yunet_2023_tensor(output, "cls", stride, rows, 1)?;
        let obj = yunet_2023_tensor(output, "obj", stride, rows, 1)?;
        let bbox = yunet_2023_tensor(output, "bbox", stride, rows, 4)?;
        let kps = yunet_2023_tensor(output, "kps", stride, rows, 10)?;
        for index in 0..rows {
            let confidence = cls.values[index];
            let objectness = obj.values[index];
            if !confidence.is_finite() || !objectness.is_finite() {
                continue;
            }
            let score = (confidence.clamp(0.0, 1.0) * objectness.clamp(0.0, 1.0)).sqrt();
            if score < options.score_threshold {
                continue;
            }
            let x = (index % columns) as f32;
            let y = (index / columns) as f32;
            let stride = stride as f32;
            let offsets = &bbox.values[index * 4..index * 4 + 4];
            let cx = (x + offsets[0]) * stride;
            let cy = (y + offsets[1]) * stride;
            let box_width = offsets[2].exp() * stride;
            let box_height = offsets[3].exp() * stride;
            let bounds = [
                (cx - box_width / 2.0) / width as f32,
                (cy - box_height / 2.0) / height as f32,
                (cx + box_width / 2.0) / width as f32,
                (cy + box_height / 2.0) / height as f32,
            ];
            let mut landmarks = [0.0; 10];
            for (point, pair) in kps.values[index * 10..index * 10 + 10]
                .as_chunks::<2>()
                .0
                .iter()
                .enumerate()
            {
                landmarks[point * 2] = (x + pair[0]) * stride / width as f32;
                landmarks[point * 2 + 1] = (y + pair[1]) * stride / height as f32;
            }
            if bounds
                .iter()
                .chain(landmarks.iter())
                .any(|value| !value.is_finite())
            {
                continue;
            }
            // Explicitly normalize and clip before entering the legacy helper,
            // whose coordinate heuristic would misread near-origin pixel values.
            let bounds = bounds.map(|value| value.clamp(0.0, 1.0));
            let landmarks = landmarks.map(|value| value.clamp(0.0, 1.0));
            if let Some(detection) = face_detection_from_xyxy(
                bounds,
                score,
                &landmarks,
                options,
                original_size,
            )? {
                detections.push(detection);
            }
        }
    }
    Ok(detections)
}

fn yunet_2023_tensor<'a>(
    output: &'a OnnxFaceDetectionOutput,
    head: &str,
    stride: u32,
    rows: usize,
    columns: usize,
) -> Result<&'a runtime_onnx::OnnxF32Tensor> {
    let name = format!("{head}_{stride}");
    let tensor = output.tensors.get(&name).ok_or_else(|| {
        DetectError::InvalidArgument(format!("YuNet output is missing `{name}` tensor"))
    })?;
    if yunet_row_count(&tensor.shape, columns)? != rows
        || rows.checked_mul(columns) != Some(tensor.values.len())
    {
        return Err(DetectError::InvalidArgument(format!(
            "YuNet `{name}` tensor must contain {rows} rows of {columns} values"
        )));
    }
    Ok(tensor)
}

fn decode_yunet_split_tensors(
    output: &OnnxFaceDetectionOutput,
    options: &OnnxFaceDetectionOptions,
    original_size: (u32, u32),
) -> Result<Vec<FaceDetection>> {
    let loc_groups = yunet_tensor_groups(output, "loc")?;
    let conf_groups = yunet_tensor_groups(output, "conf")?;
    let iou_groups = yunet_tensor_groups(output, "iou")?;

    if loc_groups.len() == 1
        && conf_groups.len() == 1
        && iou_groups.len() == 1
        && loc_groups[0].stride.is_none()
        && conf_groups[0].stride.is_none()
        && iou_groups[0].stride.is_none()
    {
        return decode_yunet_split_group(
            loc_groups[0].tensor,
            conf_groups[0].tensor,
            iou_groups[0].tensor,
            &generate_yunet_priors(&options.preprocessing),
            options,
            original_size,
        );
    }

    if loc_groups.iter().all(|group| group.stride.is_none())
        && conf_groups.iter().all(|group| group.stride.is_none())
        && iou_groups.iter().all(|group| group.stride.is_none())
        && loc_groups.len() == conf_groups.len()
        && loc_groups.len() == iou_groups.len()
        && loc_groups.len() == 3
    {
        let mut detections = Vec::new();
        for (((loc, conf), iou), stride) in loc_groups
            .iter()
            .zip(conf_groups.iter())
            .zip(iou_groups.iter())
            .zip([8_u32, 16, 32])
        {
            detections.extend(decode_yunet_split_group(
                loc.tensor,
                conf.tensor,
                iou.tensor,
                &generate_yunet_priors_for_stride(&options.preprocessing, stride),
                options,
                original_size,
            )?);
        }
        return Ok(detections);
    }

    let mut detections = Vec::new();
    for stride in [8_u32, 16, 32] {
        let loc = yunet_group_for_stride(&loc_groups, "loc", stride)?;
        let conf = yunet_group_for_stride(&conf_groups, "conf", stride)?;
        let iou = yunet_group_for_stride(&iou_groups, "iou", stride)?;
        detections.extend(decode_yunet_split_group(
            loc.tensor,
            conf.tensor,
            iou.tensor,
            &generate_yunet_priors_for_stride(&options.preprocessing, stride),
            options,
            original_size,
        )?);
    }
    Ok(detections)
}

fn decode_yunet_split_group(
    loc: &runtime_onnx::OnnxF32Tensor,
    conf: &runtime_onnx::OnnxF32Tensor,
    iou: &runtime_onnx::OnnxF32Tensor,
    priors: &[YunetPrior],
    options: &OnnxFaceDetectionOptions,
    original_size: (u32, u32),
) -> Result<Vec<FaceDetection>> {
    let count = yunet_row_count(&loc.shape, 14)?;
    if count != priors.len() {
        return Err(DetectError::InvalidArgument(format!(
            "YuNet loc output count {count} does not match generated prior count {}",
            priors.len()
        )));
    }
    if yunet_row_count(&conf.shape, output_last_dim_any(&conf.shape).unwrap_or(0))? != count
        || yunet_row_count(&iou.shape, output_last_dim_any(&iou.shape).unwrap_or(0))? != count
    {
        return Err(DetectError::InvalidArgument(
            "YuNet split output tensor counts differ".to_string(),
        ));
    }
    let conf_stride = output_last_dim_any(&conf.shape).unwrap_or(0);
    let iou_stride = output_last_dim_any(&iou.shape).unwrap_or(0);
    if conf_stride == 0 || iou_stride == 0 {
        return Err(DetectError::InvalidArgument(
            "YuNet conf/iou outputs must have a non-zero last dimension".to_string(),
        ));
    }

    let mut detections = Vec::new();
    for (index, prior) in priors.iter().enumerate() {
        let confidence = if conf_stride >= 2 {
            conf.values[index * conf_stride + 1]
        } else {
            conf.values[index * conf_stride]
        };
        let iou_score = iou.values[index * iou_stride].clamp(0.0, 1.0);
        let score = (confidence.clamp(0.0, 1.0) * iou_score)
            .sqrt()
            .clamp(0.0, 1.0);
        if !score.is_finite() || score < options.score_threshold {
            continue;
        }
        let start = index * 14;
        let decoded = decode_yunet_prior(prior, &loc.values[start..start + 14]);
        if let Some(detection) = face_detection_from_xyxy(
            decoded.box_xyxy,
            score,
            &decoded.landmarks,
            options,
            original_size,
        )? {
            detections.push(detection);
        }
    }
    Ok(detections)
}

#[derive(Debug, Clone, Copy)]
struct YunetTensorGroup<'a> {
    stride: Option<u32>,
    tensor: &'a runtime_onnx::OnnxF32Tensor,
}

#[derive(Debug, Clone, Copy)]
struct YunetPrior {
    cx: f32,
    cy: f32,
    width: f32,
    height: f32,
}

#[derive(Debug, Clone)]
struct DecodedYunetPrior {
    box_xyxy: [f32; 4],
    landmarks: Vec<f32>,
}

fn generate_yunet_priors(options: &ImageModelPreprocessing) -> Vec<YunetPrior> {
    let mut priors = Vec::new();
    for stride in [8_u32, 16, 32] {
        priors.extend(generate_yunet_priors_for_stride(options, stride));
    }
    priors
}

fn generate_yunet_priors_for_stride(
    options: &ImageModelPreprocessing,
    stride: u32,
) -> Vec<YunetPrior> {
    let input_width = options.input_width as f32;
    let input_height = options.input_height as f32;
    let feature_width = options.input_width.div_ceil(stride);
    let feature_height = options.input_height.div_ceil(stride);
    let prior_width = stride as f32 / input_width;
    let prior_height = stride as f32 / input_height;
    let mut priors = Vec::with_capacity(feature_width as usize * feature_height as usize);
    for y in 0..feature_height {
        for x in 0..feature_width {
            priors.push(YunetPrior {
                cx: (x as f32 + 0.5) * stride as f32 / input_width,
                cy: (y as f32 + 0.5) * stride as f32 / input_height,
                width: prior_width,
                height: prior_height,
            });
        }
    }
    priors
}

fn decode_yunet_prior(prior: &YunetPrior, values: &[f32]) -> DecodedYunetPrior {
    let center_x = prior.cx + values[0] * 0.1 * prior.width;
    let center_y = prior.cy + values[1] * 0.1 * prior.height;
    let width = prior.width * (values[2] * 0.2).exp();
    let height = prior.height * (values[3] * 0.2).exp();
    let mut landmarks = Vec::with_capacity(10);
    for pair in values[4..14].as_chunks::<2>().0.iter() {
        landmarks.push(prior.cx + pair[0] * 0.1 * prior.width);
        landmarks.push(prior.cy + pair[1] * 0.1 * prior.height);
    }
    DecodedYunetPrior {
        box_xyxy: [
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        ],
        landmarks,
    }
}

fn yunet_tensor_groups<'a>(
    output: &'a OnnxFaceDetectionOutput,
    needle: &str,
) -> Result<Vec<YunetTensorGroup<'a>>> {
    let groups = output
        .tensors
        .iter()
        .filter(|(name, _)| name.to_ascii_lowercase().contains(needle))
        .map(|(name, tensor)| YunetTensorGroup {
            stride: yunet_stride_from_name(name),
            tensor,
        })
        .collect::<Vec<_>>();
    if groups.is_empty() {
        Err(DetectError::InvalidArgument(format!(
            "YuNet output is missing `{needle}` tensor"
        )))
    } else {
        Ok(groups)
    }
}

fn yunet_group_for_stride<'a>(
    groups: &'a [YunetTensorGroup<'a>],
    needle: &str,
    stride: u32,
) -> Result<&'a YunetTensorGroup<'a>> {
    groups
        .iter()
        .find(|group| group.stride == Some(stride))
        .ok_or_else(|| {
            DetectError::InvalidArgument(format!(
                "YuNet output is missing `{needle}` tensor for stride {stride}"
            ))
        })
}

fn yunet_stride_from_name(name: &str) -> Option<u32> {
    name.split(|character: char| !character.is_ascii_digit())
        .filter_map(|part| part.parse::<u32>().ok())
        .find(|stride| [8_u32, 16, 32].contains(stride))
}

fn yunet_row_count(shape: &[usize], expected_last_dim: usize) -> Result<usize> {
    if expected_last_dim == 0 {
        return Err(DetectError::InvalidArgument(format!(
            "unsupported YuNet output shape `{shape:?}`"
        )));
    }
    match shape {
        [rows, last] if *last == expected_last_dim => Ok(*rows),
        [1, rows, last] if *last == expected_last_dim => Ok(*rows),
        _ => Err(DetectError::InvalidArgument(format!(
            "unsupported YuNet output shape `{shape:?}`"
        ))),
    }
}

fn output_last_dim_any(shape: &[usize]) -> Option<usize> {
    shape.last().copied()
}

fn face_detection_from_xyxy(
    bbox: [f32; 4],
    score: f32,
    landmarks: &[f32],
    options: &OnnxFaceDetectionOptions,
    original_size: (u32, u32),
) -> Result<Option<FaceDetection>> {
    let Some(region) = decode_yunet_box(bbox, options, original_size) else {
        return Ok(None);
    };
    let face_box = FaceBox::new(
        region.x as f32 / original_size.0 as f32,
        region.y as f32 / original_size.1 as f32,
        region.width as f32 / original_size.0 as f32,
        region.height as f32 / original_size.1 as f32,
    )?;
    let mut detection = FaceDetection::new(face_box, score)?;
    if landmarks.len() >= 10 {
        let points = landmarks
            .as_chunks::<2>()
            .0
            .iter()
            .take(5)
            .map(|pair| map_yunet_point(pair[0], pair[1], options, original_size))
            .collect::<Vec<_>>();
        detection = detection.landmarks(FaceLandmarks::new(points)?);
    }
    Ok(Some(detection))
}

fn decode_yunet_box(
    bbox: [f32; 4],
    options: &OnnxFaceDetectionOptions,
    original_size: (u32, u32),
) -> Option<BoundingBox> {
    if bbox.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let normalized = bbox.iter().all(|value| (-0.5..=1.5).contains(value));
    let scale_x = if normalized {
        original_size.0 as f32
    } else {
        original_size.0 as f32 / options.preprocessing.input_width as f32
    };
    let scale_y = if normalized {
        original_size.1 as f32
    } else {
        original_size.1 as f32 / options.preprocessing.input_height as f32
    };
    let width = original_size.0 as f32;
    let height = original_size.1 as f32;
    let xmin = (bbox[0] * scale_x).clamp(0.0, width);
    let ymin = (bbox[1] * scale_y).clamp(0.0, height);
    let xmax = (bbox[2] * scale_x).clamp(0.0, width);
    let ymax = (bbox[3] * scale_y).clamp(0.0, height);
    if xmax <= xmin || ymax <= ymin {
        return None;
    }
    let x = xmin.floor() as u32;
    let y = ymin.floor() as u32;
    let box_width = xmax.ceil() as u32 - x;
    let box_height = ymax.ceil() as u32 - y;
    BoundingBox::new(x, y, box_width, box_height).ok()
}

fn map_yunet_point(
    x: f32,
    y: f32,
    options: &OnnxFaceDetectionOptions,
    _original_size: (u32, u32),
) -> [f32; 2] {
    let normalized = (-0.5..=1.5).contains(&x) && (-0.5..=1.5).contains(&y);
    if normalized {
        [x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)]
    } else {
        [
            (x / options.preprocessing.input_width as f32).clamp(0.0, 1.0),
            (y / options.preprocessing.input_height as f32).clamp(0.0, 1.0),
        ]
    }
}

fn nms_face_detections(detections: Vec<FaceDetection>, threshold: f32) -> Vec<FaceDetection> {
    let mut kept: Vec<FaceDetection> = Vec::new();
    for detection in detections {
        if kept
            .iter()
            .all(|existing| face_iou(existing, &detection) <= threshold)
        {
            kept.push(detection);
        }
    }
    kept
}

fn face_iou(left: &FaceDetection, right: &FaceDetection) -> f32 {
    let left = left.bbox;
    let right = right.bbox;
    let left_x1 = left.x + left.width;
    let left_y1 = left.y + left.height;
    let right_x1 = right.x + right.width;
    let right_y1 = right.y + right.height;
    let ix0 = left.x.max(right.x);
    let iy0 = left.y.max(right.y);
    let ix1 = left_x1.min(right_x1);
    let iy1 = left_y1.min(right_y1);
    if ix1 <= ix0 || iy1 <= iy0 {
        return 0.0;
    }
    let intersection = (ix1 - ix0) * (iy1 - iy0);
    let left_area = left.width * left.height;
    let right_area = right.width * right.height;
    intersection / (left_area + right_area - intersection)
}
