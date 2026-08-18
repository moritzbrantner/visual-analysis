use std::time::Instant;

use num_rational::Rational64;
use video_analysis_core::{FramePosition, OwnedVideoFrame, SceneDetector};
use video_analysis_detectors::{
    ContentDetector, HashDetector, HistogramDetector, WeightedComponent, WeightedCompositeDetector,
};

fn main() -> video_analysis_core::Result<()> {
    let frames = synthetic_frames(600);

    run_detector("content", ContentDetector::new(27.0, 15), &frames)?;
    run_detector("histogram", HistogramDetector::new(0.05, 256, 15), &frames)?;
    run_detector("hash", HashDetector::new(0.395, 16, 2, 15), &frames)?;

    let composite = WeightedCompositeDetector::builder()
        .weighted_component(WeightedComponent::new(
            video_analysis_detectors::ContentScoreAlgorithm::new(27.0),
            1.0,
        )?)
        .weighted_component(WeightedComponent::new(
            video_analysis_detectors::HistogramScoreAlgorithm::new(0.05, 256),
            1.0,
        )?)
        .threshold(0.5)
        .min_scene_len(15)
        .build()?;
    run_detector("weighted-composite", composite, &frames)?;

    Ok(())
}

fn run_detector<D: SceneDetector>(
    name: &str,
    mut detector: D,
    frames: &[OwnedVideoFrame],
) -> video_analysis_core::Result<()> {
    let started = Instant::now();
    let mut cuts = 0;
    for frame in frames {
        cuts += detector.process_frame(&frame.as_frame(), None)?.len();
    }
    if let Some(last) = frames.last() {
        cuts += detector.finish(last.position, None)?.len();
    }
    println!(
        "{name:>18}: {:>4} frames, {:>3} cuts, {:?}",
        frames.len(),
        cuts,
        started.elapsed()
    );
    Ok(())
}

fn synthetic_frames(count: u64) -> Vec<OwnedVideoFrame> {
    (0..count)
        .map(|frame_index| {
            let value = if frame_index % 180 < 90 { 32 } else { 224 };
            let mut data = Vec::with_capacity(64 * 64 * 3);
            for pixel in 0..(64 * 64) {
                let offset = ((pixel + frame_index as usize) % 31) as u8;
                data.extend_from_slice(&[
                    value + offset / 4,
                    value,
                    value.saturating_sub(offset / 4),
                ]);
            }
            OwnedVideoFrame {
                position: FramePosition::from_frame_index(frame_index, Rational64::new(30, 1)),
                width: 64,
                height: 64,
                pixel_format: video_analysis_core::PixelFormat::Rgb24,
                data,
                stride: 64 * 3,
            }
        })
        .collect()
}
