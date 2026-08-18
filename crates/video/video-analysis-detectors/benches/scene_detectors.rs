use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use num_rational::Rational64;
use video_analysis_core::{OwnedVideoFrame, PixelFormat, SceneDetector, ScenePipeline, VideoFrame};
use video_analysis_detectors::{
    AdaptiveDetector, ContentDetector, ContentScoreAlgorithm, HashDetector, HistogramDetector,
    HistogramScoreAlgorithm, ThresholdDetector, WeightedComponent, WeightedCompositeDetector,
};
use video_analysis_test_support::{synthetic_frames, ScenePattern, SyntheticVideoSpec};

fn bench_scene_detectors(c: &mut Criterion) {
    bench_group(c, 64, 64, 120, "smoke");
    bench_group(c, 640, 360, 600, "640x360");
    bench_content_hot_paths(c, 320, 180);
    bench_content_hot_paths(c, 640, 360);
}

fn bench_group(c: &mut Criterion, width: u32, height: u32, frame_count: u64, suffix: &str) {
    let mut group = c.benchmark_group(format!("scene_detectors_{suffix}"));
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(1));

    let hard_cut = frames(width, height, frame_count)
        .pattern(ScenePattern::SolidColor {
            start_frame: 0,
            end_frame: frame_count / 3,
            rgb: [16, 16, 16],
        })
        .pattern(ScenePattern::SolidColor {
            start_frame: frame_count / 3,
            end_frame: (frame_count * 2) / 3,
            rgb: [240, 48, 48],
        })
        .pattern(ScenePattern::SolidColor {
            start_frame: (frame_count * 2) / 3,
            end_frame: frame_count,
            rgb: [48, 240, 48],
        });
    let hard_cut = synthetic_frames(&hard_cut);

    let fade = synthetic_frames(&frames(width, height, frame_count).pattern(
        ScenePattern::FadeOutIn {
            start_frame: 0,
            end_frame: frame_count,
            high: 240,
            low: 0,
        },
    ));

    let texture = synthetic_frames(
        &frames(width, height, frame_count)
            .pattern(ScenePattern::TextureShift {
                start_frame: 0,
                end_frame: frame_count / 2,
                offset: 0,
            })
            .pattern(ScenePattern::TextureShift {
                start_frame: frame_count / 2,
                end_frame: frame_count,
                offset: 5,
            }),
    );

    group.bench_function(format!("content_{suffix}_{frame_count}_frames"), |b| {
        b.iter(|| run_detector(ContentDetector::new(27.0, 15), black_box(&hard_cut)))
    });
    group.bench_function(format!("adaptive_{suffix}_{frame_count}_frames"), |b| {
        b.iter(|| {
            run_detector(
                AdaptiveDetector::new(3.0, 15, 2, 15.0),
                black_box(&hard_cut),
            )
        })
    });
    group.bench_function(
        format!("threshold_fade_{suffix}_{frame_count}_frames"),
        |b| b.iter(|| run_detector(ThresholdDetector::new(12.0, 15), black_box(&fade))),
    );
    group.bench_function(format!("histogram_{suffix}_{frame_count}_frames"), |b| {
        b.iter(|| run_detector(HistogramDetector::new(0.05, 256, 15), black_box(&hard_cut)))
    });
    group.bench_function(format!("hash_{suffix}_{frame_count}_frames"), |b| {
        b.iter(|| run_detector(HashDetector::new(0.395, 16, 2, 15), black_box(&texture)))
    });
    group.bench_function(
        format!("weighted_composite_content_histogram_{suffix}_{frame_count}_frames"),
        |b| {
            b.iter(|| {
                let detector = WeightedCompositeDetector::builder()
                    .weighted_component(
                        WeightedComponent::new(ContentScoreAlgorithm::new(27.0), 1.0).unwrap(),
                    )
                    .weighted_component(
                        WeightedComponent::new(HistogramScoreAlgorithm::new(0.05, 256), 1.0)
                            .unwrap(),
                    )
                    .threshold(0.5)
                    .min_scene_len(15)
                    .build()
                    .unwrap();
                run_detector(detector, black_box(&hard_cut))
            })
        },
    );
    group.bench_function(
        format!("pipeline_content_metrics_{suffix}_{frame_count}_frames"),
        |b| b.iter(|| run_pipeline(ContentDetector::new(27.0, 15), black_box(&hard_cut))),
    );
    group.finish();
}

fn frames(width: u32, height: u32, frame_count: u64) -> SyntheticVideoSpec {
    SyntheticVideoSpec::new(width, height, frame_count, Rational64::new(30, 1))
}

fn bench_content_hot_paths(c: &mut Criterion, width: u32, height: u32) {
    let suffix = format!("{width}x{height}");
    let frame_count = 2;
    let frames = synthetic_frames(
        &frames(width, height, frame_count)
            .pattern(ScenePattern::TextureShift {
                start_frame: 0,
                end_frame: 1,
                offset: 0,
            })
            .pattern(ScenePattern::TextureShift {
                start_frame: 1,
                end_frame: 2,
                offset: 7,
            }),
    );
    let first = frames[0].as_frame();
    let second = frames[1].as_frame();
    let mut group = c.benchmark_group(format!("content_hot_path_{suffix}"));
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(1));

    group.bench_function(format!("rgb_to_hsv_{suffix}"), |b| {
        b.iter(|| {
            let mut total = 0usize;
            for pixel in first.data.chunks_exact(3) {
                let (h, s, v) = rgb_to_hsv_compat(pixel[0], pixel[1], pixel[2]);
                total += h as usize + s as usize + v as usize;
            }
            black_box(total)
        })
    });
    group.bench_function(
        format!("content_score_frame_data_from_frame_{suffix}"),
        |b| b.iter(|| black_box(frame_data_from_frame(first))),
    );
    group.bench_function(format!("content_score_mean_abs_diff_{suffix}"), |b| {
        let left = frame_data_from_frame(first);
        let right = frame_data_from_frame(second);
        b.iter(|| black_box(mean_abs_diff_compat(&left.2, &right.2)))
    });
    group.bench_function(format!("content_detector_process_frame_{suffix}"), |b| {
        b.iter_batched(
            || ContentDetector::new(27.0, 15),
            |mut detector| {
                detector.process_frame(&first, None).unwrap();
                black_box(detector.process_frame(&second, None).unwrap().len())
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn run_detector<D: SceneDetector>(mut detector: D, frames: &[OwnedVideoFrame]) -> usize {
    let mut cuts = 0usize;
    for frame in frames {
        cuts += detector
            .process_frame(&frame.as_frame(), None)
            .unwrap()
            .len();
    }
    if let Some(last) = frames.last() {
        cuts += detector.finish(last.position, None).unwrap().len();
    }
    black_box(cuts)
}

fn run_pipeline<D: SceneDetector + 'static>(detector: D, frames: &[OwnedVideoFrame]) -> usize {
    let mut pipeline = ScenePipeline::builder()
        .detector(detector)
        .start_in_scene(true)
        .build()
        .unwrap();
    for frame in frames {
        pipeline.process_frame(frame.clone()).unwrap();
    }
    let result = pipeline.finish_detection().unwrap();
    black_box(result.cuts.len() + result.metrics.rows().len())
}

fn frame_data_from_frame(frame: VideoFrame<'_>) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut hue = Vec::with_capacity(frame.pixel_count());
    let mut sat = Vec::with_capacity(frame.pixel_count());
    let mut lum = Vec::with_capacity(frame.pixel_count());
    let row_len = frame.width as usize * 3;
    for y in 0..frame.height {
        let start = y as usize * frame.stride;
        let row = &frame.data[start..start + row_len];
        for pixel in row.chunks_exact(3) {
            let (r, g, b) = match frame.pixel_format {
                PixelFormat::Rgb24 => (pixel[0], pixel[1], pixel[2]),
                PixelFormat::Bgr24 => (pixel[2], pixel[1], pixel[0]),
            };
            let (h, s, v) = rgb_to_hsv_compat(r, g, b);
            hue.push(h);
            sat.push(s);
            lum.push(v);
        }
    }
    (hue, sat, lum)
}

fn rgb_to_hsv_compat(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r = r as f32 / 255.0;
    let g = g as f32 / 255.0;
    let b = b as f32 / 255.0;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let hue = if delta == 0.0 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / delta) % 6.0)
    } else if max == g {
        60.0 * (((b - r) / delta) + 2.0)
    } else {
        60.0 * (((r - g) / delta) + 4.0)
    };
    let hue = if hue < 0.0 { hue + 360.0 } else { hue };
    let sat = if max == 0.0 { 0.0 } else { delta / max };
    (
        (hue / 360.0 * 255.0) as u8,
        (sat * 255.0) as u8,
        (max * 255.0) as u8,
    )
}

fn mean_abs_diff_compat(left: &[u8], right: &[u8]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| (*left as i16 - *right as i16).unsigned_abs() as u64)
        .sum::<u64>() as f32
        / left.len() as f32
}

criterion_group!(benches, bench_scene_detectors);
criterion_main!(benches);
