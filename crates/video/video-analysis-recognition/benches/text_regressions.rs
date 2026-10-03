//! Bounded sampling and deterministic association benchmarks, with semantic
//! assertions outside the timed loop. No model loading or timing thresholds.
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use num_rational::Rational64;
use video_analysis_core::{BoundingBox, FramePosition, Observation, ObservationKind, Scene};
use video_analysis_recognition::{
    analyze_video_text_semantics, representative_scene_frames, VideoTextSemanticContext,
};
fn position(index: u64) -> FramePosition {
    FramePosition::from_frame_index(index, Rational64::new(30, 1))
}
fn benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_regressions");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_millis(100));
    group.measurement_time(std::time::Duration::from_millis(500));
    for count in [16_u64, 64, 128] {
        let scenes = [Scene {
            start: position(0),
            end: position(count * 30),
        }];
        let observations = (0..count)
            .flat_map(|index| {
                [800, 860].map(move |x| {
                    Observation::new("ocr", ObservationKind::Text)
                        .text("OK")
                        .at_frame(position(index * 30))
                        .region(BoundingBox::new(x, 900, 40, 30).unwrap())
                })
            })
            .collect::<Vec<_>>();
        let context = VideoTextSemanticContext {
            video_id: "benchmark",
            width: 1920,
            height: 1080,
            duration_seconds: Some(300.0),
            scenes: &scenes,
        };
        let result = analyze_video_text_semantics(context, &observations).unwrap();
        assert_eq!(result.tracks.len(), 2);
        assert!(result
            .tracks
            .iter()
            .all(|track| track.sample_count as u64 == count));
        group.throughput(Throughput::Elements(observations.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("neighboring_tracks", count),
            &observations,
            |b, input| {
                b.iter(|| {
                    black_box(analyze_video_text_semantics(context, black_box(input)).unwrap())
                })
            },
        );
        let targets = representative_scene_frames(&scenes).unwrap();
        assert!(targets.len() <= (count * 30 / 120 + 3) as usize);
        group.throughput(Throughput::Elements(targets.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("sampling_plan", count),
            &scenes,
            |b, input| b.iter(|| black_box(representative_scene_frames(black_box(input)).unwrap())),
        );
    }
    group.finish();
}
criterion_group!(benches, benchmark);
criterion_main!(benches);
