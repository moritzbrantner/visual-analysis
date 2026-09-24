use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use image_analysis_core::OwnedImage;
use image_analysis_detection::FaceLandmarks;
use image_analysis_embeddings::align_face_for_sface;

const REFERENCE: [[f64; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

fn fixture(width: u32, height: u32) -> (OwnedImage, FaceLandmarks) {
    let image = OwnedImage::new_rgb(
        width,
        height,
        vec![127; width as usize * height as usize * 3],
    )
    .unwrap();
    let scale = (f64::from(width.min(height)) / 112.0) * 0.72;
    let tx = f64::from(width) * 0.16;
    let ty = f64::from(height) * 0.10;
    let landmarks = FaceLandmarks::new(
        REFERENCE
            .iter()
            .map(|point| {
                [
                    ((point[0] * scale + tx) / f64::from(width)) as f32,
                    ((point[1] * scale + ty) / f64::from(height)) as f32,
                ]
            })
            .collect(),
    )
    .unwrap();
    (image, landmarks)
}

fn benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("sface_alignment");
    group.sample_size(20);
    group.warm_up_time(std::time::Duration::from_millis(100));
    group.measurement_time(std::time::Duration::from_millis(500));

    for (width, height) in [(320_u32, 320_u32), (1920, 1080)] {
        let (image, landmarks) = fixture(width, height);
        let aligned = align_face_for_sface(&image.as_view(), &landmarks, 112, 112).unwrap();
        assert_eq!((aligned.width, aligned.height), (112, 112));
        assert_eq!(aligned.data.len(), 112 * 112 * 3);

        group.throughput(Throughput::Elements((112 * 112) as u64));
        group.bench_with_input(
            BenchmarkId::new("five_point_warp", format!("{width}x{height}")),
            &(image, landmarks),
            |b, (image, landmarks)| {
                b.iter(|| {
                    black_box(
                        align_face_for_sface(
                            &black_box(image.as_view()),
                            black_box(landmarks),
                            112,
                            112,
                        )
                        .unwrap(),
                    )
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
