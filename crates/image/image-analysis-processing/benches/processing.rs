use criterion::{black_box, criterion_group, criterion_main, Criterion};
use image_analysis_core::{ImagePixelFormat, OwnedImage};
use image_analysis_processing::{
    convolve_3x3_kernel, grayscale_image, resize_nearest, ImageOperation, ImageProcessor,
};
use math_linear::Kernel2d;

fn image(width: u32, height: u32) -> OwnedImage {
    let mut data = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            data.push(((x * 13 + y * 7) % 251) as u8);
            data.push(((x * 5 + y * 17) % 241) as u8);
            data.push(((x * 3 + y * 11) % 239) as u8);
        }
    }
    OwnedImage::new(
        width,
        height,
        ImagePixelFormat::Rgb24,
        data,
        width as usize * 3,
    )
    .unwrap()
}

fn bench_processing(c: &mut Criterion) {
    let image = image(1_024, 1_024);
    let view = image.as_view();
    let sharpen = Kernel2d::sharpen_3x3();
    let processor = ImageProcessor::new()
        .operation(ImageOperation::Grayscale)
        .operation(ImageOperation::ResizeNearest {
            width: 512,
            height: 512,
        })
        .operation(ImageOperation::Convolve3x3 {
            kernel: Kernel2d::edge_3x3().as_array_3x3().unwrap(),
            divisor: 1.0,
            bias: 0.0,
        });

    c.bench_function("image_grayscale_1024", |b| {
        b.iter(|| grayscale_image(black_box(&view)).unwrap())
    });

    c.bench_function("image_resize_nearest_1024_to_512", |b| {
        b.iter(|| resize_nearest(black_box(&view), black_box(512), black_box(512)).unwrap())
    });

    c.bench_function("image_convolve_3x3_1024", |b| {
        b.iter(|| {
            convolve_3x3_kernel(
                black_box(&view),
                black_box(&sharpen),
                black_box(1.0),
                black_box(0.0),
            )
            .unwrap()
        })
    });

    c.bench_function("image_processing_chain_1024", |b| {
        b.iter(|| processor.process(black_box(&view)).unwrap())
    });
}

criterion_group!(benches, bench_processing);
criterion_main!(benches);
