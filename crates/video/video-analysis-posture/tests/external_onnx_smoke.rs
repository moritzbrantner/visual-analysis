#![cfg(feature = "onnx")]

use std::path::{Path, PathBuf};

use model_runtime::ModelBundle;
use num_rational::Rational64;
use video_analysis_core::{FramePosition, OwnedVideoFrame, PixelFormat};
use video_analysis_posture::OnnxPose2dEstimator;

#[test]
#[ignore = "requires local YOLOv8n pose ONNX bundle"]
fn yolov8n_pose_onnx_estimator_loads_and_runs_fixture_frame() {
    let strict = std::env::var_os("STRICT_EXTERNAL_RUNTIME_CHECKS").is_some();
    let Some(bundle) = posture_bundle(strict) else {
        eprintln!("skipping posture ONNX smoke because POSTURE_ONNX_MODEL_BUNDLE is missing");
        return;
    };
    let image_is_external = std::env::var_os("POSTURE_ONNX_IMAGE").is_some();
    let frame = fixture_frame().expect("load posture fixture frame");
    let mut estimator = OnnxPose2dEstimator::from_bundle(bundle).expect("load ONNX pose estimator");

    let poses = estimator
        .predict_frame(&frame.as_frame())
        .expect("run ONNX pose estimator");

    if image_is_external {
        assert!(
            !poses.is_empty(),
            "POSTURE_ONNX_IMAGE was provided, so the smoke expects at least one pose"
        );
    }
    for pose in &poses {
        if let Some(score) = pose.score {
            assert!(score.is_finite());
        }
        for keypoint in &pose.keypoints {
            assert!(keypoint.x.is_finite());
            assert!(keypoint.y.is_finite());
            if let Some(score) = keypoint.score {
                assert!(score.is_finite());
            }
        }
    }
}

fn posture_bundle(strict: bool) -> Option<ModelBundle> {
    let path = std::env::var_os("POSTURE_ONNX_MODEL_BUNDLE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".model-runtime/xenova-yolov8n-pose-onnx/main"));
    if path.join("manifest.json").is_file() {
        return Some(ModelBundle::load(Path::new(&path)).expect("load model bundle"));
    }
    assert!(
        !strict,
        "POSTURE_ONNX_MODEL_BUNDLE must point at a materialized model bundle"
    );
    None
}

fn fixture_frame() -> Result<OwnedVideoFrame, Box<dyn std::error::Error>> {
    if let Some(path) = std::env::var_os("POSTURE_ONNX_IMAGE") {
        let image = image::open(path)?.to_rgb8();
        let (width, height) = image.dimensions();
        return Ok(OwnedVideoFrame {
            position: FramePosition::from_frame_index(0, Rational64::new(30, 1)),
            width,
            height,
            pixel_format: PixelFormat::Rgb24,
            stride: width as usize * 3,
            data: image.into_raw(),
        });
    }
    Ok(stick_figure_frame())
}

fn stick_figure_frame() -> OwnedVideoFrame {
    let width = 640usize;
    let height = 640usize;
    let mut data = vec![245u8; width * height * 3];
    draw_line(&mut data, width, 320, 120, 320, 420);
    draw_line(&mut data, width, 320, 210, 220, 300);
    draw_line(&mut data, width, 320, 210, 420, 300);
    draw_line(&mut data, width, 320, 420, 250, 560);
    draw_line(&mut data, width, 320, 420, 390, 560);
    draw_circle(&mut data, width, 320, 80, 40);
    OwnedVideoFrame {
        position: FramePosition::from_frame_index(0, Rational64::new(30, 1)),
        width: width as u32,
        height: height as u32,
        pixel_format: PixelFormat::Rgb24,
        data,
        stride: width * 3,
    }
}

fn draw_line(data: &mut [u8], width: usize, x0: i32, y0: i32, x1: i32, y1: i32) {
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut error = dx + dy;
    let (mut x, mut y) = (x0, y0);
    loop {
        set_pixel(data, width, x, y);
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * error;
        if e2 >= dy {
            error += dy;
            x += sx;
        }
        if e2 <= dx {
            error += dx;
            y += sy;
        }
    }
}

fn draw_circle(data: &mut [u8], width: usize, cx: i32, cy: i32, radius: i32) {
    for y in (cy - radius)..=(cy + radius) {
        for x in (cx - radius)..=(cx + radius) {
            let dx = x - cx;
            let dy = y - cy;
            if dx * dx + dy * dy <= radius * radius {
                set_pixel(data, width, x, y);
            }
        }
    }
}

fn set_pixel(data: &mut [u8], width: usize, x: i32, y: i32) {
    if x < 0 || y < 0 {
        return;
    }
    let (x, y) = (x as usize, y as usize);
    let index = (y * width + x) * 3;
    if index + 2 < data.len() {
        data[index] = 20;
        data[index + 1] = 20;
        data[index + 2] = 20;
    }
}
