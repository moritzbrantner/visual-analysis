use std::collections::BTreeMap;

use super::*;
#[cfg(feature = "onnx")]
use crate::onnx_face::configure_face_model_input;
use image_analysis_core::{ImagePixelFormat, ImageView, OwnedImage};
use image_analysis_processing::{ImageModelPreprocessing, ImageModelTensor};
use image_analysis_segmentation::{
    BinaryMask, ImageSegment, ImageSegmentationBackend, ImageSegmentationRequest,
};
use video_analysis_core::{
    BoundingBox, DetectError, FramePosition, PixelFormat, Result, Timebase, Timestamp, VideoFrame,
};

struct StubSegmentationBackend {
    segments: Vec<ImageSegment>,
}

impl ImageSegmentationBackend for StubSegmentationBackend {
    fn segment_image(
        &mut self,
        _image: &ImageView<'_>,
        _request: &ImageSegmentationRequest,
    ) -> Result<Vec<ImageSegment>> {
        Ok(self.segments.clone())
    }
}

struct StubFaceRunner {
    output: OnnxFaceDetectionOutput,
}

impl OnnxFaceDetectionRunner for StubFaceRunner {
    fn run_face_detection(
        &mut self,
        _input: &ImageModelTensor,
    ) -> Result<OnnxFaceDetectionOutput> {
        Ok(self.output.clone())
    }
}

fn f32_tensor(shape: Vec<usize>, values: Vec<f32>) -> runtime_onnx::OnnxF32Tensor {
    runtime_onnx::OnnxF32Tensor::new(shape, values).unwrap()
}

fn face_output(
    entries: impl IntoIterator<Item = (&'static str, runtime_onnx::OnnxF32Tensor)>,
) -> OnnxFaceDetectionOutput {
    OnnxFaceDetectionOutput {
        tensors: entries
            .into_iter()
            .map(|(name, tensor)| (name.to_string(), tensor))
            .collect(),
    }
}

fn image() -> OwnedImage {
    OwnedImage::new(8, 8, ImagePixelFormat::Rgb24, vec![32; 8 * 8 * 3], 8 * 3).unwrap()
}

fn frame<'a>(bytes: &'a [u8]) -> VideoFrame<'a> {
    VideoFrame::packed(
        FramePosition {
            frame_index: 3,
            timestamp: Timestamp::new(3, Timebase::new(1, 1)),
        },
        8,
        8,
        PixelFormat::Rgb24,
        bytes,
        8 * 3,
    )
    .unwrap()
}

#[test]
fn detector_converts_segments_into_boxes() {
    let segment = ImageSegment::new(
        BinaryMask::filled_rect(8, 8, BoundingBox::new(1, 2, 3, 2).unwrap()).unwrap(),
    )
    .unwrap()
    .label("person")
    .score(0.9);
    let mut detector = MaskProposalDetector::new(StubSegmentationBackend {
        segments: vec![segment],
    });
    let detections = detector.detect_image(&image().as_view()).unwrap();
    assert_eq!(detections[0].label, "person");
    assert_eq!(detections[0].region, BoundingBox::new(1, 2, 3, 2).unwrap());
}

#[test]
fn detector_supports_video_frames() {
    let segment = ImageSegment::new(
        BinaryMask::filled_rect(8, 8, BoundingBox::new(2, 1, 2, 3).unwrap()).unwrap(),
    )
    .unwrap();
    let owned = image();
    let mut detector = MaskProposalDetector::new(StubSegmentationBackend {
        segments: vec![segment],
    });
    let detections = detector.detect_frame(&frame(&owned.data)).unwrap();
    assert_eq!(detections.position.frame_index, 3);
    assert_eq!(detections.detections.len(), 1);
}

#[test]
fn automatic_mask_proposals_are_explicit() {
    assert!(
        !ImageDetectionRequest::default()
            .segmentation
            .prompt
            .automatic_mask_generation
    );
    assert!(
        ImageDetectionRequest::automatic_mask_proposals()
            .segmentation
            .prompt
            .automatic_mask_generation
    );
}

#[test]
fn red_blob_detector_finds_red_rectangle() {
    let mut data = vec![0_u8; 16 * 16 * 3];
    for y in 4..10 {
        for x in 3..11 {
            let offset = (y * 16 + x) * 3;
            data[offset] = 220;
            data[offset + 1] = 20;
            data[offset + 2] = 20;
        }
    }
    let image = OwnedImage::new_rgb(16, 16, data).unwrap();
    let mut detector = ColorBlobDetector::red_car();
    let detections = detector.detect_image(&image.as_view()).unwrap();
    assert_eq!(detections.len(), 1);
    assert_eq!(detections[0].label, "car");
    assert_eq!(detections[0].attributes["color"], "red");
    assert_eq!(detections[0].region, BoundingBox::new(3, 4, 8, 6).unwrap());
}

#[test]
fn red_blob_detector_removes_small_noise() {
    let mut data = vec![0_u8; 16 * 16 * 3];
    let offset = (7 * 16 + 7) * 3;
    data[offset] = 255;
    let image = OwnedImage::new_rgb(16, 16, data).unwrap();
    let mut detector = ColorBlobDetector::red_car();
    let detections = detector.detect_image(&image.as_view()).unwrap();
    assert!(detections.is_empty());
}

#[test]
fn red_blob_detector_handles_bgr_and_gray() {
    let mut bgr = vec![0_u8; 12 * 12 * 3];
    for y in 2..8 {
        for x in 2..8 {
            let offset = (y * 12 + x) * 3;
            bgr[offset] = 10;
            bgr[offset + 1] = 10;
            bgr[offset + 2] = 220;
        }
    }
    let image = OwnedImage::new_bgr(12, 12, bgr).unwrap();
    let mut detector = ColorBlobDetector::red_car();
    assert_eq!(detector.detect_image(&image.as_view()).unwrap().len(), 1);

    let gray = OwnedImage::new_gray(12, 12, vec![220; 12 * 12]).unwrap();
    assert!(detector.detect_image(&gray.as_view()).unwrap().is_empty());
}

#[test]
fn color_blob_detector_validates_options() {
    let invalid_ratio = ColorBlobDetector::new(ColorBlobDetectionOptions {
        dominance_ratio: f32::NAN,
        ..ColorBlobDetectionOptions::default()
    })
    .unwrap_err();
    assert!(matches!(invalid_ratio, DetectError::InvalidArgument(_)));

    let empty_label = ColorBlobDetector::new(ColorBlobDetectionOptions {
        label: "  ".to_string(),
        ..ColorBlobDetectionOptions::default()
    })
    .unwrap_err();
    assert!(matches!(empty_label, DetectError::InvalidArgument(_)));
}

#[test]
fn color_blob_detector_supports_custom_options_and_video_frames() {
    let mut data = vec![0_u8; 8 * 8 * 3];
    for &(x, y) in &[(1_usize, 1_usize), (6, 2)] {
        let offset = (y * 8 + x) * 3;
        data[offset] = 180;
        data[offset + 1] = 50;
        data[offset + 2] = 50;
    }
    let mut detector = ColorBlobDetector::new(
        ColorBlobDetectionOptions::red_car()
            .min_area_pixels(1)
            .label("marker"),
    )
    .unwrap();
    detector.options.morph_open_3x3 = false;

    let detections = detector.detect_frame(&frame(&data)).unwrap();
    assert_eq!(detections.position.frame_index, 3);
    assert_eq!(detections.detections.len(), 2);
    assert!(detections
        .detections
        .iter()
        .all(|detection| detection.label == "marker"));
    assert_eq!(detections.detections[0].attributes["area"], "1");
}

#[test]
fn face_detection_contracts_validate_inputs() {
    let bbox = FaceBox::new(0.1, 0.2, 0.3, 0.4).unwrap();
    let detection = FaceDetection::new(bbox, 0.95)
        .unwrap()
        .landmarks(FaceLandmarks::new(vec![[0.2, 0.3], [0.4, 0.5]]).unwrap());
    assert_eq!(detection.landmarks.unwrap().points.len(), 2);
    assert!(FaceBox::new(0.0, 0.0, f32::NAN, 1.0).is_err());
    assert_eq!(
        FaceDetectionPreset::OpenCvYuNet
            .model_spec()
            .repo_id_value(),
        Some("opencv/face_detection_yunet")
    );
}

#[test]
fn yunet_preprocessing_sends_unscaled_bgr_channels_to_the_runner() {
    struct BgrRunner;
    impl OnnxFaceDetectionRunner for BgrRunner {
        fn run_face_detection(
            &mut self,
            input: &ImageModelTensor,
        ) -> Result<OnnxFaceDetectionOutput> {
            assert_eq!((input.width, input.height, input.channels), (32, 32, 3));
            assert_eq!(input.values[0], 30.0);
            assert_eq!(input.values[32 * 32], 20.0);
            assert_eq!(input.values[2 * 32 * 32], 10.0);
            Ok(yunet_2023_output())
        }
    }
    let image = OwnedImage::new_rgb(1, 1, vec![10, 20, 30]).unwrap();
    let mut detector = OnnxFaceDetector::with_options(yunet_2023_options(), BgrRunner).unwrap();
    assert!(detector.detect_faces(&image.as_view()).unwrap().is_empty());
}

#[cfg(feature = "onnx")]
#[test]
fn yunet_preprocessing_respects_fixed_model_input_dimensions() {
    use runtime_onnx::{OnnxDimension, OnnxIoInfo, OnnxTensorElementType};

    let mut preprocessing = OnnxFaceDetectionOptions::default().preprocessing;
    let mut input = OnnxIoInfo {
        name: "input".to_string(),
        element_type: Some(OnnxTensorElementType::F32),
        dimensions: [1, 3, 640, 640].map(OnnxDimension::Fixed).to_vec(),
    };
    configure_face_model_input(&mut preprocessing, &input).unwrap();
    assert_eq!(
        (preprocessing.input_width, preprocessing.input_height),
        (640, 640)
    );
    input.dimensions[2] = OnnxDimension::Fixed(0);
    assert!(configure_face_model_input(&mut preprocessing, &input).is_err());
}

#[test]
fn yunet_2023_runner_decodes_scores_boxes_landmarks_and_nms() {
    let mut output = yunet_2023_output();
    // Two overlapping predictions across strides; only the higher score survives.
    output.tensors.get_mut("cls_8").unwrap().values[5] = 0.81;
    output.tensors.get_mut("obj_8").unwrap().values[5] = 1.0;
    output.tensors.get_mut("bbox_8").unwrap().values[20..24]
        .copy_from_slice(&[1.0, 1.0, 2.0_f32.ln(), 2.0_f32.ln()]);
    output.tensors.get_mut("kps_8").unwrap().values[50..60]
        .copy_from_slice(&[0.125, 0.25, 1.0, 0.0, 0.5, 0.5, 0.0, 1.0, 1.0, 1.0]);
    output.tensors.get_mut("cls_16").unwrap().values[0] = 0.64;
    output.tensors.get_mut("obj_16").unwrap().values[0] = 1.0;
    output.tensors.get_mut("bbox_16").unwrap().values[..4]
        .copy_from_slice(&[1.0, 1.0, 0.0, 0.0]);
    // The coarsest head contributes a distinct, lower-confidence full-image box.
    output.tensors.get_mut("cls_32").unwrap().values[0] = 0.49;
    output.tensors.get_mut("obj_32").unwrap().values[0] = 1.0;
    output.tensors.get_mut("bbox_32").unwrap().values[..4]
        .copy_from_slice(&[0.5, 0.5, 0.0, 0.0]);

    let mut detector = OnnxFaceDetector::with_options(
        yunet_2023_options(),
        StubFaceRunner { output },
    )
    .unwrap();
    let pixels = vec![0_u8; 64 * 96 * 3];
    let image = ImageView::packed(64, 96, ImagePixelFormat::Rgb24, &pixels).unwrap();
    let detections = detector.detect_faces(&image).unwrap();
    assert_eq!(detections.len(), 2);
    assert!((detections[0].confidence - 0.9).abs() < 0.0001);
    assert!((detections[1].confidence - 0.7).abs() < 0.0001);
    assert_eq!(detections[0].bbox, FaceBox::new(0.25, 0.25, 0.5, 0.5).unwrap());
    assert_eq!(detections[1].bbox, FaceBox::new(0.0, 0.0, 1.0, 1.0).unwrap());
    assert_eq!(
        detections[0].landmarks.as_ref().unwrap().points[0],
        [0.28125, 0.3125]
    );
}

#[test]
fn yunet_2023_landmarks_near_origin_are_pixels_not_normalized_coordinates() {
    let mut output = yunet_2023_output();
    output.tensors.get_mut("cls_8").unwrap().values[0] = 1.0;
    output.tensors.get_mut("obj_8").unwrap().values[0] = 1.0;
    output.tensors.get_mut("bbox_8").unwrap().values[..4]
        .copy_from_slice(&[0.5, 0.5, 0.0, 0.0]);
    output.tensors.get_mut("kps_8").unwrap().values[..10].fill(0.125);
    let detections =
        decode_yunet_face_detections(&output, &yunet_2023_options(), (32, 32)).unwrap();
    assert_eq!(
        detections[0].landmarks.as_ref().unwrap().points[0],
        [1.0 / 32.0; 2]
    );
}

#[test]
fn yunet_2023_rejects_missing_heads_and_malformed_buffers() {
    let options = yunet_2023_options();
    let mut missing = yunet_2023_output();
    missing.tensors.remove("kps_16");
    assert!(decode_yunet_face_detections(&missing, &options, (32, 32))
        .unwrap_err()
        .to_string()
        .contains("kps_16"));

    let mut truncated = yunet_2023_output();
    truncated.tensors.get_mut("bbox_8").unwrap().values.pop();
    assert!(decode_yunet_face_detections(&truncated, &options, (32, 32)).is_err());

    let mut wrong_grid = yunet_2023_output();
    wrong_grid.tensors.insert(
        "cls_8".to_string(),
        f32_tensor(vec![1, 15, 1], vec![0.0; 15]),
    );
    assert!(decode_yunet_face_detections(&wrong_grid, &options, (32, 32)).is_err());
}

fn yunet_2023_options() -> OnnxFaceDetectionOptions {
    OnnxFaceDetectionOptions {
        preprocessing: ImageModelPreprocessing {
            input_width: 32,
            input_height: 32,
            ..OnnxFaceDetectionOptions::default().preprocessing
        },
        score_threshold: 0.5,
        ..OnnxFaceDetectionOptions::default()
    }
}

fn yunet_2023_output() -> OnnxFaceDetectionOutput {
    let mut tensors = BTreeMap::new();
    for stride in [8, 16, 32] {
        let rows = (32 / stride) * (32 / stride);
        for (name, columns) in [("cls", 1), ("obj", 1), ("bbox", 4), ("kps", 10)] {
            tensors.insert(
                format!("{name}_{stride}"),
                f32_tensor(vec![1, rows, columns], vec![0.0; rows * columns]),
            );
        }
    }
    OnnxFaceDetectionOutput { tensors }
}

#[test]
fn yunet_decode_supports_single_tensor_layout() {
    let options = OnnxFaceDetectionOptions {
        score_threshold: 0.5,
        nms_threshold: 0.3,
        preprocessing: ImageModelPreprocessing {
            input_width: 100,
            input_height: 100,
            ..OnnxFaceDetectionOptions::default().preprocessing
        },
        ..OnnxFaceDetectionOptions::default()
    };
    let output = face_output([(
        "faces",
        f32_tensor(
            vec![2, 15],
            vec![
                10.0, 20.0, 30.0, 40.0, 0.8, 12.0, 22.0, 28.0, 22.0, 20.0, 30.0, 14.0, 36.0,
                26.0, 36.0, 50.0, 60.0, 70.0, 80.0, 0.2, 52.0, 62.0, 68.0, 62.0, 60.0, 70.0,
                54.0, 76.0, 66.0, 76.0,
            ],
        ),
    )]);
    let detections = decode_yunet_face_detections(&output, &options, (100, 100)).unwrap();
    assert_eq!(detections.len(), 1);
    assert_eq!(detections[0].confidence, 0.8);
    assert_eq!(
        detections[0].bbox,
        FaceBox::new(0.1, 0.2, 0.2, 0.2).unwrap()
    );
    assert_eq!(detections[0].landmarks.as_ref().unwrap().points.len(), 5);
}

#[test]
fn yunet_decode_supports_batched_single_tensor_layout() {
    let options = OnnxFaceDetectionOptions {
        score_threshold: 0.5,
        nms_threshold: 0.3,
        preprocessing: ImageModelPreprocessing {
            input_width: 100,
            input_height: 100,
            ..OnnxFaceDetectionOptions::default().preprocessing
        },
        ..OnnxFaceDetectionOptions::default()
    };
    let output = face_output([(
        "faces",
        f32_tensor(
            vec![1, 1, 15],
            vec![
                -10.0, -10.0, 50.0, 50.0, 0.8, 0.0, 0.0, 60.0, 0.0, 25.0, 25.0, 0.0, 60.0,
                60.0, 60.0,
            ],
        ),
    )]);
    let detections = decode_yunet_face_detections(&output, &options, (100, 100)).unwrap();
    assert_eq!(detections.len(), 1);
    assert_eq!(
        detections[0].bbox,
        FaceBox::new(0.0, 0.0, 0.5, 0.5).unwrap()
    );
    assert!(detections[0]
        .landmarks
        .as_ref()
        .unwrap()
        .points
        .iter()
        .flatten()
        .all(|value| (0.0..=1.0).contains(value)));
}

#[test]
fn yunet_decode_supports_split_loc_conf_iou_outputs() {
    let options = OnnxFaceDetectionOptions {
        score_threshold: 0.5,
        nms_threshold: 0.3,
        preprocessing: ImageModelPreprocessing {
            input_width: 8,
            input_height: 8,
            ..OnnxFaceDetectionOptions::default().preprocessing
        },
        ..OnnxFaceDetectionOptions::default()
    };
    let output = face_output([
        ("loc", f32_tensor(vec![3, 14], vec![0.0; 3 * 14])),
        (
            "conf",
            f32_tensor(vec![3, 2], vec![0.0, 0.81, 0.0, 0.1, 0.0, 0.1]),
        ),
        ("iou", f32_tensor(vec![3, 1], vec![1.0, 1.0, 1.0])),
    ]);
    let detections = decode_yunet_face_detections(&output, &options, (8, 8)).unwrap();
    assert_eq!(detections.len(), 1);
    assert!((detections[0].confidence - 0.9).abs() < 0.0001);
    assert_eq!(
        detections[0].bbox,
        FaceBox::new(0.0, 0.0, 1.0, 1.0).unwrap()
    );
}

#[test]
fn yunet_decode_supports_per_stride_split_outputs() {
    let options = OnnxFaceDetectionOptions {
        score_threshold: 0.5,
        nms_threshold: 0.3,
        preprocessing: ImageModelPreprocessing {
            input_width: 8,
            input_height: 8,
            ..OnnxFaceDetectionOptions::default().preprocessing
        },
        ..OnnxFaceDetectionOptions::default()
    };
    let output = face_output([
        ("loc_8", f32_tensor(vec![1, 14], vec![0.0; 14])),
        ("conf_8", f32_tensor(vec![1, 2], vec![0.0, 0.81])),
        ("iou_8", f32_tensor(vec![1, 1], vec![1.0])),
        ("loc_16", f32_tensor(vec![1, 14], vec![0.0; 14])),
        ("conf_16", f32_tensor(vec![1, 2], vec![0.0, 0.1])),
        ("iou_16", f32_tensor(vec![1, 1], vec![1.0])),
        ("loc_32", f32_tensor(vec![1, 14], vec![0.0; 14])),
        ("conf_32", f32_tensor(vec![1, 2], vec![0.0, 0.1])),
        ("iou_32", f32_tensor(vec![1, 1], vec![1.0])),
    ]);
    let detections = decode_yunet_face_detections(&output, &options, (8, 8)).unwrap();
    assert_eq!(detections.len(), 1);
    assert!((detections[0].confidence - 0.9).abs() < 0.0001);
}

#[test]
fn yunet_decode_rejects_mismatched_prior_count() {
    let options = OnnxFaceDetectionOptions {
        score_threshold: 0.5,
        nms_threshold: 0.3,
        preprocessing: ImageModelPreprocessing {
            input_width: 8,
            input_height: 8,
            ..OnnxFaceDetectionOptions::default().preprocessing
        },
        ..OnnxFaceDetectionOptions::default()
    };
    let output = face_output([
        ("loc", f32_tensor(vec![2, 14], vec![0.0; 2 * 14])),
        ("conf", f32_tensor(vec![2, 2], vec![0.0, 0.9, 0.0, 0.8])),
        ("iou", f32_tensor(vec![2, 1], vec![1.0, 1.0])),
    ]);
    let err = decode_yunet_face_detections(&output, &options, (8, 8)).unwrap_err();
    assert!(err.to_string().contains("prior count"));
}

#[test]
fn yunet_decode_rejects_missing_stride_group() {
    let options = OnnxFaceDetectionOptions {
        score_threshold: 0.5,
        nms_threshold: 0.3,
        preprocessing: ImageModelPreprocessing {
            input_width: 8,
            input_height: 8,
            ..OnnxFaceDetectionOptions::default().preprocessing
        },
        ..OnnxFaceDetectionOptions::default()
    };
    let output = face_output([
        ("loc_8", f32_tensor(vec![1, 14], vec![0.0; 14])),
        ("conf_8", f32_tensor(vec![1, 2], vec![0.0, 0.81])),
        ("iou_8", f32_tensor(vec![1, 1], vec![1.0])),
    ]);
    let err = decode_yunet_face_detections(&output, &options, (8, 8)).unwrap_err();
    let message = err.to_string();
    assert!(message.contains("stride 16"));
    assert!(message.contains("loc"));
}

#[test]
fn face_detector_runner_decodes_and_applies_nms() {
    let options = OnnxFaceDetectionOptions {
        score_threshold: 0.5,
        nms_threshold: 0.4,
        preprocessing: ImageModelPreprocessing {
            input_width: 16,
            input_height: 16,
            ..OnnxFaceDetectionOptions::default().preprocessing
        },
        ..OnnxFaceDetectionOptions::default()
    };
    let output = face_output([(
        "faces",
        f32_tensor(
            vec![3, 15],
            vec![
                1.0, 1.0, 9.0, 9.0, 0.9, 2.0, 2.0, 8.0, 2.0, 5.0, 5.0, 3.0, 8.0, 7.0, 8.0, 2.0,
                2.0, 10.0, 10.0, 0.8, 3.0, 3.0, 9.0, 3.0, 6.0, 6.0, 4.0, 9.0, 8.0, 9.0, 12.0,
                12.0, 15.0, 15.0, 0.7, 12.0, 12.0, 15.0, 12.0, 14.0, 14.0, 13.0, 15.0, 15.0,
                15.0,
            ],
        ),
    )]);
    let mut detector =
        OnnxFaceDetector::with_options(options, StubFaceRunner { output }).unwrap();
    let detections = detector.detect_faces(&image().as_view()).unwrap();
    assert_eq!(detections.len(), 2);
    assert!(detections[0].confidence > detections[1].confidence);
}
