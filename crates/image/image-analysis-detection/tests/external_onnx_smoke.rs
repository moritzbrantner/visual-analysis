#![cfg(feature = "external-tests")]

use std::path::{Path, PathBuf};

use image_analysis_core::OwnedImage;
use image_analysis_detection::OnnxObjectDetector;
use model_runtime::ModelBundle;

fn bundle(name: &str) -> Option<ModelBundle> {
    let root = PathBuf::from(".model-runtime").join(name).join("main");
    if !root.join("manifest.json").is_file() {
        eprintln!(
            "skipping external ONNX smoke test; missing {}",
            root.display()
        );
        return None;
    }
    Some(ModelBundle::load(Path::new(&root)).expect("load model bundle"))
}

#[test]
#[ignore = "requires local DETR/YOLOS ONNX bundle in .model-runtime"]
fn detr_like_onnx_decodes_detection_shapes() {
    let Some(bundle) = bundle("xenova-detr-resnet-50-onnx") else {
        return;
    };
    let image = OwnedImage::new_rgb(16, 16, vec![64; 16 * 16 * 3]).unwrap();
    let mut detector = OnnxObjectDetector::from_bundle(bundle).unwrap();
    let detections = detector.detect_image(&image.as_view()).unwrap();
    assert!(detections.iter().all(|detection| {
        detection.score.is_some_and(f32::is_finite)
            && detection.region.width > 0
            && detection.region.height > 0
    }));
}
