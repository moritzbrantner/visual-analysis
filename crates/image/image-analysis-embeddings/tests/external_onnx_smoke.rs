#![cfg(feature = "external-tests")]

use std::path::{Path, PathBuf};

use image_analysis_core::OwnedImage;
use image_analysis_embeddings::{ImageEmbedderBackend, OnnxImageEmbedder};
use model_runtime::ModelBundle;
use runtime_core::{OperationId, SurfaceRequest};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .find(|root| root.join("Cargo.toml").is_file() && root.join("crates/image").is_dir())
        .expect("visual-analysis workspace root")
        .to_path_buf()
}

fn face_fixture_path() -> PathBuf {
    workspace_root()
        .join(".external-test-tools/visual-fixtures")
        .join("opencv-lena.jpg")
}

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


#[test]
#[ignore = "downloads/uses local YuNet and SFace ONNX bundles and requires ONNX Runtime"]
fn standard_face_detection_and_sface_embedding_align() {
    let root = workspace_root();
    let fixture = face_fixture_path();
    assert!(
        fixture.is_file(),
        "missing face fixture {}; run scripts/setup_visual_acceptance_fixtures.sh",
        fixture.display()
    );

    let detection = image_analysis_detection::surface::run_surface_operation(SurfaceRequest {
        operation: OperationId::new("image.detection.detectFaces"),
        input: serde_json::json!({
            "imagePath": fixture,
            "modelRoot": root.join(".model-runtime"),
            "model": "opencv-yunet-onnx",
            "autoDownload": true,
            "limit": 1
        }),
    })
    .expect("run YuNet face detection")
    .value;

    assert_eq!(detection["executed"], true);
    let face = detection["detections"]
        .as_array()
        .and_then(|detections| detections.first())
        .expect("YuNet should detect the pinned face fixture");
    assert!(
        face["score"].as_f64().is_some_and(|score| score >= 0.9),
        "unexpected face confidence: {face:?}"
    );
    let landmarks = face["landmarks"]
        .as_array()
        .expect("YuNet face should expose landmarks");
    assert_eq!(landmarks.len(), 5, "SFace requires five YuNet landmarks");

    let embedding = image_analysis_embeddings::surface::run_surface_operation(SurfaceRequest {
        operation: OperationId::new("image.embeddings.faceEmbed"),
        input: serde_json::json!({
            "imagePath": fixture,
            "modelRoot": root.join(".model-runtime"),
            "model": "opencv-sface-onnx",
            "autoDownload": true,
            "region": face["region"].clone(),
            "landmarks": face["landmarks"].clone()
        }),
    })
    .expect("run SFace face embedding")
    .value;

    assert_eq!(embedding["executed"], true);
    assert_eq!(embedding["alignment"], "sface-five-point-alignment");
    assert_eq!(embedding["dimensions"], 128);
    let vector = embedding["vector"].as_array().expect("embedding vector");
    assert_eq!(vector.len(), 128);
    let norm = vector
        .iter()
        .map(|value| {
            let value = value.as_f64().expect("finite JSON embedding value");
            value * value
        })
        .sum::<f64>()
        .sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "expected normalized SFace vector, norm={norm}"
    );
}
