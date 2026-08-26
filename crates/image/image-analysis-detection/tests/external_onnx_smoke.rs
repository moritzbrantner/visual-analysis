#![cfg(feature = "external-tests")]

use std::path::{Path, PathBuf};

use runtime_core::{OperationId, SurfaceRequest};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .find(|root| root.join("Cargo.toml").is_file() && root.join("crates/image").is_dir())
        .expect("visual-analysis workspace root")
        .to_path_buf()
}

fn fixture_path() -> PathBuf {
    workspace_root()
        .join(".external-test-tools/visual-fixtures")
        .join("coco-000000039769.png")
}

fn run_detection(min_score: f32) -> serde_json::Value {
    let root = workspace_root();
    let fixture = fixture_path();
    assert!(
        fixture.is_file(),
        "missing detection fixture {}; run scripts/setup_visual_acceptance_fixtures.sh",
        fixture.display()
    );

    image_analysis_detection::surface::run_surface_operation(SurfaceRequest {
        operation: OperationId::new("image.detection.detect"),
        input: serde_json::json!({
            "imagePath": fixture,
            "modelRoot": root.join(".model-runtime"),
            "model": "xenova-detr-resnet-50-onnx",
            "autoDownload": true,
            "minScore": min_score,
            "limit": 20
        }),
    })
    .expect("run detection surface")
    .value
}

#[test]
#[ignore = "downloads/uses local DETR ONNX bundle and requires ONNX Runtime"]
fn standard_detection_surface_finds_cats_in_coco_fixture() {
    let value = run_detection(0.5);
    assert_eq!(value["executed"], true);
    let detections = value["detections"].as_array().expect("detections array");
    assert!(!detections.is_empty(), "expected non-empty DETR detections");

    let cats = detections
        .iter()
        .filter(|detection| {
            detection["label"]
                .as_str()
                .is_some_and(|label| label.eq_ignore_ascii_case("cat"))
        })
        .collect::<Vec<_>>();
    assert!(
        cats.len() >= 2,
        "expected at least two cat detections, got {detections:?}"
    );
    assert!(cats.iter().all(|cat| {
        cat["score"].as_f64().is_some_and(|score| score >= 0.5)
            && cat["region"]["width"].as_u64().is_some_and(|width| width > 80)
            && cat["region"]["height"].as_u64().is_some_and(|height| height > 100)
    }));
}

#[test]
#[ignore = "downloads/uses local DETR ONNX bundle and requires ONNX Runtime"]
fn standard_detection_surface_places_cats_on_both_sides_of_fixture() {
    let value = run_detection(0.5);
    let detections = value["detections"].as_array().expect("detections array");
    let cat_centers = detections
        .iter()
        .filter(|detection| {
            detection["label"]
                .as_str()
                .is_some_and(|label| label.eq_ignore_ascii_case("cat"))
        })
        .filter_map(|cat| {
            let x = cat["region"]["x"].as_u64()?;
            let width = cat["region"]["width"].as_u64()?;
            Some(x + width / 2)
        })
        .collect::<Vec<_>>();

    assert!(
        cat_centers.iter().any(|center| *center < 320),
        "expected a cat detection in the left half, got {cat_centers:?}"
    );
    assert!(
        cat_centers.iter().any(|center| *center >= 320),
        "expected a cat detection in the right half, got {cat_centers:?}"
    );
}
