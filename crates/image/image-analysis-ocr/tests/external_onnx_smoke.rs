#![cfg(feature = "external-tests")]

use std::path::{Path, PathBuf};

use runtime_core::{OperationId, SurfaceRequest};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .find(|root| root.join("tests/fixtures/ocr").is_dir())
        .expect("workspace root with OCR fixtures")
        .to_path_buf()
}

fn normalize_ocr_text(text: &str) -> String {
    text.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_uppercase)
        .collect()
}

#[test]
#[ignore = "downloads/uses local TrOCR ONNX bundle and requires ONNX Runtime"]
fn standard_ocr_surface_recognizes_hello_fixture() {
    let root = workspace_root();
    let response = image_analysis_ocr::surface::run_surface_operation(SurfaceRequest {
        operation: OperationId::new("image.ocr.recognize"),
        input: serde_json::json!({
            "imagePath": root.join("tests/fixtures/ocr/trocr-hello.png"),
            "modelRoot": root.join(".model-runtime"),
            "model": "trocr-base-printed-onnx",
            "autoDownload": true,
            "languages": ["en"],
            "maxNewTokens": 64
        }),
    })
    .expect("run OCR surface");

    assert_eq!(response.value["executed"], true);
    let text = response.value["document"]["text"]
        .as_str()
        .expect("OCR document text");
    assert!(
        normalize_ocr_text(text).contains("HELLO"),
        "expected OCR text to contain HELLO, got {text:?}"
    );
    assert_eq!(response.value["document"]["width"], 240);
    assert_eq!(response.value["document"]["height"], 80);
}
