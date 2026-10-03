//! Lightweight canonical detector comparison; use `cargo bench --bench
//! scene_detectors` for the retained legacy-vs-one-pass regression evidence.
use scenedetect_core::{
    detect_frames, DetectionOptions, DetectorConfig, Frame, FrameIndex, FrameRate,
};
use std::time::Instant;
fn main() -> scenedetect_core::Result<()> {
    let frames = (0..600)
        .map(|index| Frame {
            index: FrameIndex(index),
            width: 64,
            height: 36,
            rgb: vec![if index % 180 < 90 { 32 } else { 224 }; 64 * 36 * 3],
        })
        .collect::<Vec<_>>();
    for (name, config) in [
        ("content", DetectorConfig::Content(Default::default())),
        ("histogram", DetectorConfig::Histogram(Default::default())),
        ("hash", DetectorConfig::Hash(Default::default())),
    ] {
        let start = Instant::now();
        let result = detect_frames(
            config,
            FrameRate(30.0),
            &frames,
            DetectionOptions::default(),
        )?;
        println!(
            "{name}: {} frames, {} scenes, {:?}",
            frames.len(),
            result.scene_list.scenes.len(),
            start.elapsed()
        );
    }
    Ok(())
}
