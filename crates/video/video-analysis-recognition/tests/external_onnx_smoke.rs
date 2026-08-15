#![cfg(feature = "onnx")]

use std::path::{Path, PathBuf};

use model_runtime::ModelBundle;
use num_rational::Rational64;
use video_analysis_core::{FramePosition, OwnedVideoFrame, PixelFormat};
use video_analysis_recognition::OnnxObjectDetectorBackend;

#[test]
#[ignore = "requires local ONNX object-detection bundle"]
fn video_recognition_onnx_detects_fixture_frame_when_configured() {
    let Some(bundle) = detection_bundle() else {
        eprintln!(
            "skipping video recognition ONNX smoke because RECOGNITION_ONNX_MODEL_BUNDLE is missing"
        );
        return;
    };
    let mut detector = OnnxObjectDetectorBackend::from_bundle(bundle).expect("load ONNX detector");
    let frame = fixture_frame();
    let predictions = detector
        .predict_frame_with_original_size(&frame.as_frame())
        .expect("run ONNX detector");
    for prediction in &predictions {
        if let Some(score) = prediction.score {
            assert!(score.is_finite());
        }
        if let Some(region) = &prediction.region {
            if let Some(width) = region.width {
                assert!(width.is_finite());
                assert!(width >= 0.0);
            }
            if let Some(height) = region.height {
                assert!(height.is_finite());
                assert!(height >= 0.0);
            }
        }
    }
}

fn detection_bundle() -> Option<ModelBundle> {
    let path = std::env::var_os("RECOGNITION_ONNX_MODEL_BUNDLE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".model-runtime/xenova-detr-resnet-50-onnx/main"));
    path.join("manifest.json")
        .is_file()
        .then(|| ModelBundle::load(Path::new(&path)).expect("load model bundle"))
}

fn fixture_frame() -> OwnedVideoFrame {
    let width = 16;
    let height = 16;
    let data = std::env::var_os("RECOGNITION_ONNX_IMAGE")
        .and_then(|path| std::fs::read(path).ok())
        .filter(|data| data.len() == width * height * 3)
        .unwrap_or_else(|| vec![96; width * height * 3]);
    OwnedVideoFrame {
        position: FramePosition::from_frame_index(0, Rational64::new(30, 1)),
        width: width as u32,
        height: height as u32,
        pixel_format: PixelFormat::Rgb24,
        data,
        stride: width * 3,
    }
}
