use criterion::{black_box, criterion_group, criterion_main, Criterion};
use video_analysis_core::BoundingBox;
use video_analysis_tracking::{
    detect_detection_collisions_with_broad_phase, CollisionBroadPhaseOptions,
    CollisionBroadPhaseStrategy, CollisionCellSize, CollisionOptions, TrackedDetection,
};

fn sparse_detections(count: usize) -> Vec<TrackedDetection> {
    (0..count)
        .map(|index| {
            TrackedDetection::new(
                BoundingBox::new((index * 10) as u32, (index * 7) as u32, 2, 2).unwrap(),
            )
        })
        .collect()
}

fn clustered_detections(count: usize) -> Vec<TrackedDetection> {
    (0..count)
        .map(|index| {
            let x = ((index % 256) * 3) as u32;
            let y = (((index / 256) % 256) * 3) as u32;
            TrackedDetection::new(BoundingBox::new(x, y, 8, 8).unwrap())
        })
        .collect()
}

fn overlapping_detections(count: usize) -> Vec<TrackedDetection> {
    (0..count)
        .map(|index| {
            TrackedDetection::new(BoundingBox::new((index % 512) as u32, 0, 4, 16).unwrap())
        })
        .collect()
}

fn options(strategy: CollisionBroadPhaseStrategy) -> CollisionBroadPhaseOptions {
    CollisionBroadPhaseOptions {
        strategy,
        cell_size: CollisionCellSize::Auto,
        ..CollisionBroadPhaseOptions::default()
    }
}

fn bench_collision(c: &mut Criterion) {
    for (name, make_detections) in [
        (
            "sparse",
            sparse_detections as fn(usize) -> Vec<TrackedDetection>,
        ),
        ("clustered", clustered_detections),
        ("many_overlap", overlapping_detections),
    ] {
        for count in [1_000, 10_000, 50_000] {
            let detections = make_detections(count);
            c.bench_function(&format!("tracking_{name}_auto_{count}"), |b| {
                b.iter(|| {
                    detect_detection_collisions_with_broad_phase(
                        black_box(&detections),
                        black_box(CollisionOptions::default()),
                        black_box(options(CollisionBroadPhaseStrategy::Auto)),
                    )
                    .unwrap()
                })
            });
            c.bench_function(&format!("tracking_{name}_sweep_{count}"), |b| {
                b.iter(|| {
                    detect_detection_collisions_with_broad_phase(
                        black_box(&detections),
                        black_box(CollisionOptions::default()),
                        black_box(options(CollisionBroadPhaseStrategy::SweepAndPrune)),
                    )
                    .unwrap()
                })
            });
            c.bench_function(&format!("tracking_{name}_grid_{count}"), |b| {
                b.iter(|| {
                    detect_detection_collisions_with_broad_phase(
                        black_box(&detections),
                        black_box(CollisionOptions::default()),
                        black_box(options(CollisionBroadPhaseStrategy::SpatialHashGrid)),
                    )
                    .unwrap()
                })
            });
            if count == 1_000 {
                c.bench_function(&format!("tracking_{name}_brute_{count}"), |b| {
                    b.iter(|| {
                        detect_detection_collisions_with_broad_phase(
                            black_box(&detections),
                            black_box(CollisionOptions::default()),
                            black_box(options(CollisionBroadPhaseStrategy::BruteForce)),
                        )
                        .unwrap()
                    })
                });
            }
        }
    }
}

criterion_group!(benches, bench_collision);
criterion_main!(benches);
