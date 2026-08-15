#![cfg(feature = "external-tests")]

use std::path::{Path, PathBuf};

use image_analysis_captioning::{ImageCaptionerBackend, OnnxImageCaptioner};
use image_analysis_core::OwnedImage;
use model_runtime::ModelBundle;

fn bundle(names: &[&str]) -> Option<ModelBundle> {
    let mut roots = Vec::new();
    for name in names {
        let root = PathBuf::from(".model-runtime").join(name).join("main");
        if root.join("manifest.json").is_file() {
            return Some(ModelBundle::load(Path::new(&root)).expect("load model bundle"));
        }
        roots.push(root);
    }
    eprintln!(
        "skipping external ONNX smoke test; missing any manifest at {}",
        roots
            .iter()
            .map(|root| root.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    None
}

#[test]
#[ignore = "requires local ViT-GPT2 captioning ONNX bundle in .model-runtime"]
fn vit_gpt2_onnx_returns_non_empty_caption() {
    let Some(bundle) = bundle(&[
        "vit-gpt2-image-captioning",
        "vit-gpt2-image-captioning-onnx",
    ]) else {
        return;
    };
    let image = OwnedImage::new_rgb(8, 8, vec![180; 8 * 8 * 3]).unwrap();
    let mut captioner = OnnxImageCaptioner::from_bundle(bundle).unwrap();
    let captions = captioner.caption_image(&image.as_view()).unwrap();
    assert_eq!(captions.len(), 1);
    assert!(!captions[0].text.trim().is_empty());
    assert_eq!(captions[0].score, None);
}
