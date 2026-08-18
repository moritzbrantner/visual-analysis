use criterion::{black_box, criterion_group, criterion_main, Criterion};
use math_linear::Kernel2d;
use num_rational::Rational64;
use video_analysis_core::{FramePosition, OwnedVideoFrame, PixelFormat};
use video_analysis_editing::{box_blur_frame, edge_detect_frame, grayscale_frame, FrameEditor};

fn frame(width: u32, height: u32) -> OwnedVideoFrame {
    let mut data = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            data.push(((x * 19 + y * 3) % 253) as u8);
            data.push(((x * 7 + y * 23) % 251) as u8);
            data.push(((x * 5 + y * 13) % 247) as u8);
        }
    }
    OwnedVideoFrame {
        position: FramePosition::from_frame_index(0, Rational64::new(30, 1)),
        width,
        height,
        pixel_format: PixelFormat::Rgb24,
        stride: width as usize * 3,
        data,
    }
}

fn bench_editing(c: &mut Criterion) {
    let frame = frame(640, 360);
    let view = frame.as_frame();
    let editor = FrameEditor::new()
        .brightness_contrast(8, 1.15)
        .grayscale()
        .filter_3x3_kernel(Kernel2d::sharpen_3x3(), 1.0, 0.0)
        .unwrap();

    c.bench_function("frame_grayscale_640x360", |b| {
        b.iter(|| grayscale_frame(black_box(&view)).unwrap())
    });

    c.bench_function("frame_box_blur_radius_3_640x360", |b| {
        b.iter(|| box_blur_frame(black_box(&view), black_box(3)).unwrap())
    });

    c.bench_function("frame_edge_detect_640x360", |b| {
        b.iter(|| edge_detect_frame(black_box(&view)).unwrap())
    });

    c.bench_function("frame_editor_chain_640x360", |b| {
        b.iter(|| editor.apply(black_box(&view)).unwrap())
    });
}

criterion_group!(benches, bench_editing);
criterion_main!(benches);
