use num_rational::Rational64;
use video_analysis_core::{FramePosition, SceneDetector};
use video_analysis_detectors::{
    AdaptiveDetector, ContentDetector, FlashFilter, FlashFilterMode, HashDetector,
    HistogramDetector, ThresholdDetector, ThresholdMethod,
};
use video_analysis_test_support::{
    assert_scene_boundaries, run_detector_on_frames, synthetic_frames, ScenePattern,
    SyntheticVideoSpec,
};

fn fps() -> Rational64 {
    Rational64::new(30, 1)
}

fn spec(frame_count: u64) -> SyntheticVideoSpec {
    SyntheticVideoSpec::new(64, 64, frame_count, fps())
}

fn scene_starts<D: SceneDetector + 'static>(
    detector: D,
    frames: &[video_analysis_core::OwnedVideoFrame],
) -> Vec<u64> {
    run_detector_on_frames(detector, frames)
        .unwrap()
        .scenes
        .iter()
        .map(|scene| scene.start.frame_index)
        .collect()
}

#[test]
fn content_detector_matches_hard_cut_boundaries_at_default_min_scene_len() {
    let frames = synthetic_frames(
        &spec(80)
            .pattern(ScenePattern::SolidColor {
                start_frame: 0,
                end_frame: 20,
                rgb: [16, 16, 16],
            })
            .pattern(ScenePattern::SolidColor {
                start_frame: 20,
                end_frame: 40,
                rgb: [240, 32, 32],
            })
            .pattern(ScenePattern::SolidColor {
                start_frame: 40,
                end_frame: 60,
                rgb: [32, 240, 32],
            })
            .pattern(ScenePattern::SolidColor {
                start_frame: 60,
                end_frame: 80,
                rgb: [32, 32, 240],
            }),
    );

    let result = run_detector_on_frames(ContentDetector::new(27.0, 15), &frames).unwrap();
    assert_scene_boundaries(&result, &[0, 20, 40, 60]);
    assert_eq!(
        result
            .cuts
            .iter()
            .map(|cut| cut.position.frame_index)
            .collect::<Vec<_>>(),
        [20, 40, 60]
    );
}

#[test]
fn content_detector_respects_longer_min_scene_len() {
    let frames = synthetic_frames(
        &spec(90)
            .pattern(ScenePattern::SolidColor {
                start_frame: 0,
                end_frame: 35,
                rgb: [16, 16, 16],
            })
            .pattern(ScenePattern::SolidColor {
                start_frame: 35,
                end_frame: 70,
                rgb: [240, 32, 32],
            })
            .pattern(ScenePattern::SolidColor {
                start_frame: 70,
                end_frame: 90,
                rgb: [32, 240, 32],
            }),
    );

    assert_eq!(
        scene_starts(ContentDetector::new(27.0, 30), &frames),
        [0, 35, 70]
    );
}

#[test]
fn adaptive_detector_reports_target_frame_after_window_latency() {
    let frames = synthetic_frames(
        &spec(24)
            .pattern(ScenePattern::SolidColor {
                start_frame: 0,
                end_frame: 10,
                rgb: [16, 16, 16],
            })
            .pattern(ScenePattern::SolidColor {
                start_frame: 10,
                end_frame: 24,
                rgb: [240, 240, 240],
            }),
    );
    let mut detector = AdaptiveDetector::new(3.0, 1, 2, 15.0);
    let mut emitted_on_processing_frame = None;
    let mut cut_frame = None;
    for frame in &frames {
        let cuts = detector.process_frame(&frame.as_frame(), None).unwrap();
        if let Some(cut) = cuts.first() {
            emitted_on_processing_frame = Some(frame.position.frame_index);
            cut_frame = Some(cut.position.frame_index);
            break;
        }
    }

    assert_eq!(cut_frame, Some(10));
    assert_eq!(emitted_on_processing_frame, Some(12));
}

#[test]
fn threshold_detector_finds_floor_and_ceiling_fade_boundaries() {
    let floor_frames = synthetic_frames(&spec(80).pattern(ScenePattern::FadeOutIn {
        start_frame: 0,
        end_frame: 60,
        high: 240,
        low: 0,
    }));
    let floor = run_detector_on_frames(
        ThresholdDetector::new(12.0, 1)
            .method(ThresholdMethod::Floor)
            .add_final_scene(true),
        &floor_frames,
    )
    .unwrap();
    assert_eq!(
        floor
            .cuts
            .iter()
            .map(|cut| cut.position.frame_index)
            .collect::<Vec<_>>(),
        [30]
    );

    let ceiling_frames = synthetic_frames(
        &spec(80)
            .pattern(ScenePattern::SolidColor {
                start_frame: 0,
                end_frame: 20,
                rgb: [0, 0, 0],
            })
            .pattern(ScenePattern::FadeOutIn {
                start_frame: 20,
                end_frame: 70,
                high: 255,
                low: 0,
            }),
    );
    let ceiling = run_detector_on_frames(
        ThresholdDetector::new(243.0, 1)
            .method(ThresholdMethod::Ceiling)
            .fade_bias(0.0)
            .add_final_scene(true),
        &ceiling_frames,
    )
    .unwrap();
    assert_eq!(
        ceiling
            .cuts
            .iter()
            .map(|cut| cut.position.frame_index)
            .collect::<Vec<_>>(),
        [21, 69]
    );
}

#[test]
fn histogram_detector_triggers_on_distribution_change_not_small_drift() {
    let unchanged = synthetic_frames(&spec(50).pattern(ScenePattern::SolidColor {
        start_frame: 0,
        end_frame: 50,
        rgb: [96, 96, 96],
    }));
    assert_eq!(
        scene_starts(HistogramDetector::new(0.05, 32, 10), &unchanged),
        [0]
    );

    let shifted = synthetic_frames(
        &spec(50)
            .pattern(ScenePattern::SolidColor {
                start_frame: 0,
                end_frame: 20,
                rgb: [96, 96, 96],
            })
            .pattern(ScenePattern::HistogramShift {
                start_frame: 20,
                end_frame: 50,
            }),
    );
    assert_eq!(
        scene_starts(HistogramDetector::new(0.05, 32, 10), &shifted),
        [0, 20]
    );
}

#[test]
fn hash_detector_triggers_on_texture_shift_not_uniform_brightness() {
    let uniform = synthetic_frames(
        &spec(50)
            .pattern(ScenePattern::SolidColor {
                start_frame: 0,
                end_frame: 25,
                rgb: [80, 80, 80],
            })
            .pattern(ScenePattern::SolidColor {
                start_frame: 25,
                end_frame: 50,
                rgb: [120, 120, 120],
            }),
    );
    assert_eq!(
        scene_starts(HashDetector::new(0.395, 16, 2, 10), &uniform),
        [0]
    );

    let textured = synthetic_frames(
        &spec(50)
            .pattern(ScenePattern::TextureShift {
                start_frame: 0,
                end_frame: 25,
                offset: 0,
            })
            .pattern(ScenePattern::TextureShift {
                start_frame: 25,
                end_frame: 50,
                offset: 5,
            }),
    );
    assert_eq!(
        scene_starts(HashDetector::new(0.1, 16, 2, 10), &textured),
        [0, 25]
    );
}

#[test]
fn flash_filter_merge_and_suppress_match_min_scene_len_contracts() {
    let mut suppress = FlashFilter::new(FlashFilterMode::Suppress, 3);
    assert_eq!(suppress.filter(0, true), Vec::<u64>::new());
    assert_eq!(suppress.filter(1, true), Vec::<u64>::new());
    assert_eq!(suppress.filter(3, true), vec![3]);
    assert_eq!(suppress.filter(5, true), Vec::<u64>::new());
    assert_eq!(suppress.filter(6, true), vec![6]);

    let mut merge = FlashFilter::new(FlashFilterMode::Merge, 3);
    assert_eq!(merge.filter(0, false), Vec::<u64>::new());
    assert_eq!(merge.filter(3, true), vec![3]);
    assert_eq!(merge.filter(4, true), Vec::<u64>::new());
    assert_eq!(merge.filter(5, false), Vec::<u64>::new());
    assert_eq!(merge.filter(7, true), Vec::<u64>::new());
    assert_eq!(merge.filter(10, false), vec![7]);
}

#[test]
fn detector_positions_keep_frame_rate_timestamps() {
    let frames = synthetic_frames(&spec(2));
    assert_eq!(
        frames[1].position,
        FramePosition::from_frame_index(1, Rational64::new(30, 1))
    );
}
