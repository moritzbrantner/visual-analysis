use num_rational::Rational64;
use video_analysis_core::{FramePosition, Scene};
use video_analysis_split::{build_split_plan, SplitOptions, DEFAULT_TEMPLATE};

fn scene(start: u64, end: u64) -> Scene {
    Scene {
        start: FramePosition::from_frame_index(start, Rational64::new(10, 1)),
        end: FramePosition::from_frame_index(end, Rational64::new(10, 1)),
    }
}

#[test]
fn default_template_stays_pyscenedetect_like() {
    assert_eq!(DEFAULT_TEMPLATE, "$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4");
}

#[test]
fn template_expands_frame_tokens_and_zero_padded_scene_numbers() {
    let options = SplitOptions {
        output_dir: "out".into(),
        template: "$VIDEO_NAME-Scene-$SCENE_NUMBER-$START_FRAME-$END_FRAME.mp4".to_string(),
        ..SplitOptions::default()
    };
    let scenes = vec![scene(0, 10), scene(10, 25), scene(25, 40)];

    let plan = build_split_plan("clip.mov", &scenes, &options).unwrap();

    assert_eq!(
        plan.jobs[0].output_path,
        std::path::PathBuf::from("out/clip-Scene-001-0-10.mp4")
    );
    assert_eq!(
        plan.jobs[1].output_path,
        std::path::PathBuf::from("out/clip-Scene-002-10-25.mp4")
    );
    assert_eq!(
        plan.jobs[2].output_path,
        std::path::PathBuf::from("out/clip-Scene-003-25-40.mp4")
    );
}

#[test]
fn split_duration_uses_scene_end_as_exclusive_media_time() {
    let plan = build_split_plan("clip.mov", &[scene(10, 25)], &SplitOptions::default()).unwrap();

    assert_eq!(plan.jobs[0].start_seconds, 1.0);
    assert_eq!(plan.jobs[0].duration_seconds, 1.5);
}
