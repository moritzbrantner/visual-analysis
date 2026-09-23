use num_rational::Rational64;
use video_analysis_core::{BoundingBox, FramePosition, Observation, ObservationKind, Scene};
use video_analysis_recognition::{
    analyze_video_text_semantics, VideoTextRole, VideoTextSemanticContext,
};
fn position(frame: u64) -> FramePosition {
    FramePosition::from_frame_index(frame, Rational64::new(30, 1))
}
fn observation(frame: u64, x: u32) -> Observation {
    Observation::new("ocr", ObservationKind::Text)
        .text("OK")
        .at_frame(position(frame))
        .region(BoundingBox::new(x, 900, 40, 30).unwrap())
}
fn analyze(observations: &[Observation]) -> video_analysis_recognition::VideoTextSemanticAnalysis {
    analyze_video_text_semantics(
        VideoTextSemanticContext {
            video_id: "audit",
            width: 1920,
            height: 1080,
            duration_seconds: Some(120.0),
            scenes: &[Scene {
                start: position(0),
                end: position(3600),
            }],
        },
        observations,
    )
    .unwrap()
}
#[test]
fn neighboring_identical_text_in_one_frame_is_not_temporal_evidence() {
    let result = analyze(&[observation(0, 800), observation(0, 860)]);
    assert_eq!(result.tracks.len(), 2);
    assert!(result
        .tracks
        .iter()
        .all(|track| track.role != VideoTextRole::Subtitle && track.sample_count == 1));
    let mut observations = vec![
        observation(0, 800),
        observation(0, 860),
        observation(30, 800),
        observation(30, 860),
    ];
    let result = analyze(&observations);
    assert_eq!(result.tracks.len(), 2);
    assert!(result
        .tracks
        .iter()
        .all(|track| track.sample_count == 2 && track.role == VideoTextRole::Subtitle));
    observations.reverse();
    assert_eq!(analyze(&observations), result);
}
#[test]
fn duplicate_same_frame_observations_do_not_create_a_subtitle() {
    let result = analyze(&[observation(0, 800), observation(0, 800)]);
    assert_eq!(result.tracks.len(), 1);
    assert_ne!(result.tracks[0].role, VideoTextRole::Subtitle);
}
#[test]
fn missing_time_is_not_evidence_of_persistence() {
    let mut observation = observation(0, 800);
    observation.frame = None;
    observation.timestamp = None;
    let result = analyze(&[observation.clone(), observation]);
    assert!(result
        .tracks
        .iter()
        .all(|track| track.role != VideoTextRole::Subtitle));
}
#[test]
fn a_disappearance_longer_than_the_gap_starts_a_new_track() {
    let result = analyze(&[
        observation(0, 800),
        observation(30, 800),
        observation(360, 800),
        observation(390, 800),
    ]);
    assert_eq!(result.tracks.len(), 2);
}
#[test]
#[cfg(feature = "ocr")]
fn long_scene_sampling_is_bounded_and_compatible_with_tracking() {
    use video_analysis_recognition::representative_scene_frames;
    for fps in [5_u64, 30, 60] {
        let pos = |index| FramePosition::from_frame_index(index, Rational64::new(fps as i64, 1));
        let scenes = [Scene {
            start: pos(0),
            end: pos(60 * fps),
        }];
        let plan = representative_scene_frames(&scenes)
            .unwrap()
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(plan[0], 0);
        assert_eq!(*plan.last().unwrap(), 60 * fps - 1);
        assert!(
            plan.len() <= 34,
            "recognition work must remain bounded, {plan:?}"
        );
        for pair in plan.windows(2) {
            assert!(pair[1] - pair[0] <= 120);
            assert!((pair[1] - pair[0]) as f64 / fps as f64 <= 4.0);
        }
        let observations = plan
            .iter()
            .map(|index| {
                Observation::new("ocr", ObservationKind::Text)
                    .text("Stable title")
                    .at_frame(pos(*index))
                    .region(BoundingBox::new(10, 10, 300, 50).unwrap())
            })
            .collect::<Vec<_>>();
        let result = analyze_video_text_semantics(
            VideoTextSemanticContext {
                video_id: "long-scene",
                width: 1920,
                height: 1080,
                duration_seconds: Some(120.0),
                scenes: &scenes,
            },
            &observations,
        )
        .unwrap();
        assert_eq!(result.tracks.len(), 1);
        assert_eq!(result.tracks[0].sample_count, plan.len() as u32);
    }
    let enormous = [Scene {
        start: position(0),
        end: position(10_000_000),
    }];
    assert!(
        representative_scene_frames(&enormous).is_err(),
        "oversized sampling plans must fail before allocation"
    );
}
