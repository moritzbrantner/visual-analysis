//! Same-process legacy/candidate comparison. Correctness and allocation budgets
//! are tested separately; machine-dependent timings are evidence, not CI gates.
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use num_rational::Rational64;
use video_analysis_core::{
    ContentDetector, FramePosition, OwnedVideoFrame, PixelFormat, Result, ScenePipeline,
};
use video_analysis_ingest::surface::detect_content_scenes;
use video_analysis_ingest::{MediaSourceInfo, VideoFrameSource, VideoStreamInfo};

struct Generated {
    info: MediaSourceInfo,
    next: u64,
    count: u64,
}
impl Generated {
    fn new(count: u64) -> Self {
        Self {
            info: MediaSourceInfo::recorded("benchmark").with_video(VideoStreamInfo {
                width: 64,
                height: 36,
                pixel_format: PixelFormat::Rgb24,
                frame_rate: Some(Rational64::new(30, 1)),
            }),
            next: 0,
            count,
        }
    }
}
impl VideoFrameSource for Generated {
    fn source_info(&self) -> &MediaSourceInfo {
        &self.info
    }
    fn next_video_frame(&mut self) -> Result<Option<OwnedVideoFrame>> {
        if self.next == self.count {
            return Ok(None);
        }
        let index = self.next;
        self.next += 1;
        Ok(Some(OwnedVideoFrame {
            position: FramePosition::from_frame_index(index, Rational64::new(30, 1)),
            width: 64,
            height: 36,
            pixel_format: PixelFormat::Rgb24,
            data: vec![if index < self.count / 2 { 16 } else { 240 }; 64 * 36 * 3],
            stride: 64 * 3,
        }))
    }
}
fn legacy(count: u64) -> video_analysis_core::DetectionResult {
    let mut source = Generated::new(count);
    let mut pipeline = ScenePipeline::builder()
        .detector(ContentDetector::new(20.0, 1))
        .start_in_scene(true)
        .build()
        .unwrap();
    while let Some(frame) = source.next_video_frame().unwrap() {
        pipeline.process_frame(frame).unwrap();
    }
    pipeline.finish_detection().unwrap()
}
fn candidate(count: u64) -> video_analysis_core::DetectionResult {
    detect_content_scenes(&mut Generated::new(count), 20.0, 1).unwrap()
}
fn benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("scene_ingest");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_millis(100));
    group.measurement_time(std::time::Duration::from_millis(500));
    for count in [16_u64, 64, 128] {
        let old = legacy(count);
        let new = candidate(count);
        assert_eq!(new.scenes, old.scenes);
        assert_eq!(new.cuts, old.cuts);
        assert_eq!(new.frames_processed, old.frames_processed);
        // The typed stats path additionally exposes component metrics. Compare
        // every previously returned score, not an artificially reduced workload.
        for frame in 0..count {
            assert_eq!(
                new.metrics.get(frame, "content_val"),
                old.metrics.get(frame, "content_val")
            );
        }
        group.throughput(Throughput::Elements(count));
        group.bench_with_input(
            BenchmarkId::new("legacy_prefix", count),
            &count,
            |b, &count| b.iter(|| black_box(legacy(count))),
        );
        group.bench_with_input(BenchmarkId::new("one_pass", count), &count, |b, &count| {
            b.iter(|| black_box(candidate(count)))
        });
    }
    group.finish();
}
criterion_group!(benches, benchmark);
criterion_main!(benches);
