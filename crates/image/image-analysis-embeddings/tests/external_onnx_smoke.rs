#![cfg(feature = "external-tests")]

use std::path::{Path, PathBuf};

use image_analysis_core::OwnedImage;
use image_analysis_embeddings::{ImageEmbedderBackend, OnnxImageEmbedder};
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
#[ignore = "requires local CLIP image embedding ONNX bundle in .model-runtime"]
fn onnx_image_embedding_returns_finite_normalized_vector() {
    let Some(bundle) = bundle("xenova-clip-vit-base-patch32-onnx") else {
        return;
    };
    let image = OwnedImage::new_rgb(8, 8, vec![96; 8 * 8 * 3]).unwrap();
    let mut embedder = OnnxImageEmbedder::from_bundle(bundle).unwrap();
    let embedding = embedder.embed_image(&image.as_view()).unwrap();
    assert!(!embedding.vector.is_empty());
    assert!(embedding.vector.iter().all(|value| value.is_finite()));
    let norm = embedding
        .vector
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    assert!((norm - 1.0).abs() < 0.01);
}
