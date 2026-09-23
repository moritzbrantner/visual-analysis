//! Canonical adapter contracts replacing unbuildable pre-extraction tests.
//! This filename is retained for provenance. These tests do not claim parity
//! with PySceneDetect's removed streaming FlashFilter or ceiling-threshold API.
//! The registered scenedetect-core configuration is the executable authority.
use num_rational::Rational64;
use scenedetect_core::{DetectionOptions, DetectorConfig, MinSceneLenPolicy};
use video_analysis_core::{FramePosition, OwnedVideoFrame, PixelFormat, Result, VideoSource};
use video_analysis_detectors::detect_source;
struct Fixture(std::vec::IntoIter<OwnedVideoFrame>);
impl VideoSource for Fixture {
    fn frame_rate(&self) -> Rational64 {
        Rational64::new(30, 1)
    }
    fn next_frame(&mut self) -> Result<Option<OwnedVideoFrame>> {
        Ok(self.0.next())
    }
}
fn frames(colors: &[[u8; 3]], length: usize) -> Vec<OwnedVideoFrame> {
    (0..colors.len() * length)
        .map(|index| OwnedVideoFrame {
            position: FramePosition::from_frame_index(index as u64, Rational64::new(30, 1)),
            width: 32,
            height: 32,
            pixel_format: PixelFormat::Rgb24,
            data: colors[index / length].repeat(32 * 32),
            stride: 96,
        })
        .collect()
}
fn starts(
    config: DetectorConfig,
    frames: Vec<OwnedVideoFrame>,
    minimum: u64,
    policy: MinSceneLenPolicy,
) -> Vec<u64> {
    detect_source(
        Fixture(frames.into_iter()),
        config,
        DetectionOptions {
            min_scene_len: minimum,
            min_scene_len_policy: policy,
        },
    )
    .unwrap()
    .scene_list
    .scenes
    .into_iter()
    .map(|scene| scene.start.0)
    .collect()
}
#[test]
fn content_hard_cuts_preserve_default_minimum_length() {
    assert_eq!(
        starts(
            DetectorConfig::Content(Default::default()),
            frames(&[[16; 3], [240, 32, 32], [32, 240, 32], [32, 32, 240]], 20),
            15,
            MinSceneLenPolicy::Suppress
        ),
        [0, 20, 40, 60]
    );
}
#[test]
fn content_hard_cuts_preserve_longer_minimum_length() {
    assert_eq!(
        starts(
            DetectorConfig::Content(Default::default()),
            frames(&[[16; 3], [240, 32, 32], [32, 240, 32]], 35),
            30,
            MinSceneLenPolicy::Suppress
        ),
        [0, 35, 70]
    );
}
#[test]
fn adaptive_output_refers_to_target_frame_not_lookahead_frame() {
    assert_eq!(
        starts(
            DetectorConfig::Adaptive(Default::default()),
            frames(&[[16; 3], [240; 3]], 10),
            1,
            MinSceneLenPolicy::Suppress
        ),
        [0, 10]
    );
}
#[test]
fn all_five_canonical_configurations_execute_without_monolith_test_support() {
    for config in [
        DetectorConfig::Content(Default::default()),
        DetectorConfig::Adaptive(Default::default()),
        DetectorConfig::Threshold(Default::default()),
        DetectorConfig::Histogram(Default::default()),
        DetectorConfig::Hash(Default::default()),
    ] {
        assert_eq!(
            starts(
                config,
                frames(&[[96; 3]], 20),
                1,
                MinSceneLenPolicy::Suppress
            ),
            [0]
        );
    }
}
#[test]
fn histogram_distinguishes_brightness_distributions() {
    assert_eq!(
        starts(
            DetectorConfig::Histogram(Default::default()),
            frames(&[[32; 3], [200; 3]], 20),
            10,
            MinSceneLenPolicy::Suppress
        ),
        [0, 20]
    );
}
#[test]
fn hash_ignores_uniform_brightness_shifts() {
    assert_eq!(
        starts(
            DetectorConfig::Hash(Default::default()),
            frames(&[[80; 3], [120; 3]], 25),
            10,
            MinSceneLenPolicy::Suppress
        ),
        [0]
    );
}
#[test]
fn canonical_merge_and_suppress_options_have_explicit_tail_semantics() {
    let config = DetectorConfig::Content(Default::default());
    let mut input = frames(&[[16; 3], [240; 3]], 6);
    input.truncate(8);
    assert_eq!(
        starts(
            config.clone(),
            input.clone(),
            5,
            MinSceneLenPolicy::Suppress
        ),
        [0, 6]
    );
    assert_eq!(starts(config, input, 5, MinSceneLenPolicy::MergeLast), [0]);
}
#[test]
fn adapter_preserves_declared_frame_rate() {
    let result = detect_source(
        Fixture(frames(&[[96; 3]], 2).into_iter()),
        DetectorConfig::Content(Default::default()),
        DetectionOptions::default(),
    )
    .unwrap();
    assert_eq!(result.scene_list.frame_rate.0, 30.0);
    assert_eq!(result.stats.rows.len(), 2);
}
