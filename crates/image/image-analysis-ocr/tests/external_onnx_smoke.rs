#![cfg(feature = "external-tests")]

use std::path::{Path, PathBuf};

use image_analysis_core::OwnedImage;
use image_analysis_ocr::{OcrBackend, OcrRequest, OnnxTrOcrBackend};
use model_runtime::ModelBundle;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .find(|root| root.join("tests/fixtures").is_dir())
        .expect("workspace root with tests/fixtures")
        .to_path_buf()
}

fn model_runtime_roots() -> Vec<PathBuf> {
    vec![
        PathBuf::from(".model-runtime"),
        workspace_root().join(".model-runtime"),
    ]
}

fn fixture_image() -> OwnedImage {
    image_analysis_io::read_image(workspace_root().join("tests/fixtures/ocr/trocr-hello.png"))
        .expect("load OCR fixture")
}

fn normalize_ocr_text(text: &str) -> String {
    text.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_uppercase)
        .collect()
}

fn bundle(names: &[&str]) -> Option<ModelBundle> {
    let mut roots = Vec::new();
    for bundle_root in model_runtime_roots() {
        for name in names {
            let root = bundle_root.join(name).join("main");
            if root.join("manifest.json").is_file() {
                return Some(ModelBundle::load(&root).expect("load model bundle"));
            }
            roots.push(root);
        }
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
#[ignore = "requires local TrOCR ONNX bundle in .model-runtime"]
fn trocr_onnx_returns_non_empty_ocr_document() {
    let Some(bundle) = bundle(&["trocr-base-printed-onnx", "trocr-base-printed"]) else {
        return;
    };
    let image = fixture_image();
    let mut backend = OnnxTrOcrBackend::from_bundle(bundle).unwrap();
    let document = backend
        .recognize_image(
            &image.as_view(),
            &OcrRequest::new().languages(["en"]).include_tokens(false),
        )
        .unwrap();

    let normalized = normalize_ocr_text(&document.text);
    assert!(
        normalized.contains("HELLO"),
        "expected OCR text to contain HELLO, got {:?}",
        document.text
    );
    assert_eq!(
        (document.width, document.height),
        (image.width, image.height)
    );
}
