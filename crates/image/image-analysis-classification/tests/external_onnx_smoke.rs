#![cfg(feature = "external-tests")]

use std::path::{Path, PathBuf};

use image_analysis_classification::{ImageClassifierBackend, OnnxImageClassifier};
use image_analysis_core::OwnedImage;
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
#[ignore = "requires local ViT ONNX bundle in .model-runtime"]
fn vit_onnx_classifies_tiny_fixture_image() {
    let Some(bundle) = bundle("vit-base-patch16-224") else {
        return;
    };
    let image = OwnedImage::new_rgb(8, 8, vec![128; 8 * 8 * 3]).unwrap();
    let mut classifier = OnnxImageClassifier::from_bundle(bundle).unwrap();
    let predictions = classifier.classify_image(&image.as_view()).unwrap();
    assert!(!predictions.is_empty());
    assert!(predictions
        .iter()
        .all(|prediction| prediction.score.is_some_and(f32::is_finite)));
}
