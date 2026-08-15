#[path = "../examples/scene_dataset_eval.rs"]
#[allow(clippy::duplicate_mod)]
#[allow(dead_code)]
mod scene_dataset_eval;
#[path = "../examples/scene_dataset_eval_support/mod.rs"]
#[allow(dead_code)]
mod scene_dataset_eval_support;

use scene_dataset_eval_support::{
    parse_args_from, suppress_nearby_cuts, FilterMode, BBC_CONTENT_TUNED_ADAPTIVE_MIN_CONTENT_VAL,
    BBC_CONTENT_TUNED_ADAPTIVE_THRESHOLD, BBC_CONTENT_TUNED_ADAPTIVE_WINDOW_WIDTH,
    BBC_CONTENT_TUNED_MIN_SCENE_LEN, BBC_CONTENT_TUNED_POST_FILTER_WINDOW,
    BBC_CONTENT_TUNED_PRESET,
};
use video_analysis_core::PixelFormat;

fn base_args() -> Vec<&'static str> {
    vec![
        "--dataset",
        "bbc",
        "--root",
        ".test-corpora/video-scene/BBC",
    ]
}

#[test]
fn defaults_preserve_current_content_detector_settings() {
    let args = parse_args_from(base_args()).unwrap();

    assert_eq!(args.detector, "content");
    assert_eq!(args.config.content_threshold, 27.0);
    assert_eq!(args.config.min_scene_len, 15);
    assert_eq!(args.config.filter_mode, FilterMode::Merge);
    assert_eq!(args.config.adaptive_threshold, 3.0);
    assert_eq!(args.config.adaptive_window_width, 2);
    assert_eq!(args.config.adaptive_min_content_val, 15.0);
    assert_eq!(args.config.post_filter_window, 0);
}

#[test]
fn parses_content_tuning_flags() {
    let mut args = base_args();
    args.extend([
        "--content-threshold",
        "33",
        "--min-scene-len",
        "20",
        "--filter-mode",
        "merge",
    ]);

    let args = parse_args_from(args).unwrap();

    assert_eq!(args.config.content_threshold, 33.0);
    assert_eq!(args.config.min_scene_len, 20);
    assert_eq!(args.config.filter_mode, FilterMode::Merge);
}

#[test]
fn parses_adaptive_tuning_flags() {
    let mut args = base_args();
    args.extend([
        "--detector",
        "adaptive",
        "--adaptive-threshold",
        "3.5",
        "--adaptive-window-width",
        "3",
        "--adaptive-min-content-val",
        "18",
    ]);

    let args = parse_args_from(args).unwrap();

    assert_eq!(args.detector, "adaptive");
    assert_eq!(args.config.adaptive_threshold, 3.5);
    assert_eq!(args.config.adaptive_window_width, 3);
    assert_eq!(args.config.adaptive_min_content_val, 18.0);
}

#[test]
fn invalid_filter_mode_fails() {
    let mut args = base_args();
    args.extend(["--filter-mode", "invalid"]);

    let error = parse_args_from(args).unwrap_err();

    assert!(error.contains("invalid --filter-mode"));
}

#[test]
fn parses_post_filter_window() {
    let mut args = base_args();
    args.extend(["--post-filter-window", "12"]);

    let args = parse_args_from(args).unwrap();

    assert_eq!(args.config.post_filter_window, 12);
}

#[test]
fn parses_bbc_content_tuned_preset() {
    let mut args = base_args();
    args.extend(["--preset", BBC_CONTENT_TUNED_PRESET]);

    let args = parse_args_from(args).unwrap();

    assert_eq!(args.detector, "adaptive");
    assert_eq!(args.preset.as_deref(), Some(BBC_CONTENT_TUNED_PRESET));
    assert_eq!(
        args.config.adaptive_threshold,
        BBC_CONTENT_TUNED_ADAPTIVE_THRESHOLD
    );
    assert_eq!(
        args.config.adaptive_window_width,
        BBC_CONTENT_TUNED_ADAPTIVE_WINDOW_WIDTH
    );
    assert_eq!(
        args.config.adaptive_min_content_val,
        BBC_CONTENT_TUNED_ADAPTIVE_MIN_CONTENT_VAL
    );
    assert_eq!(args.config.min_scene_len, BBC_CONTENT_TUNED_MIN_SCENE_LEN);
    assert_eq!(
        args.config.post_filter_window,
        BBC_CONTENT_TUNED_POST_FILTER_WINDOW
    );
}

#[test]
fn explicit_flags_after_preset_override_preset_values() {
    let mut args = base_args();
    args.extend([
        "--preset",
        BBC_CONTENT_TUNED_PRESET,
        "--adaptive-threshold",
        "4.5",
        "--adaptive-window-width",
        "4",
        "--adaptive-min-content-val",
        "21",
        "--min-scene-len",
        "36",
        "--post-filter-window",
        "20",
    ]);

    let args = parse_args_from(args).unwrap();

    assert_eq!(args.config.adaptive_threshold, 4.5);
    assert_eq!(args.config.adaptive_window_width, 4);
    assert_eq!(args.config.adaptive_min_content_val, 21.0);
    assert_eq!(args.config.min_scene_len, 36);
    assert_eq!(args.config.post_filter_window, 20);
}

#[test]
fn parses_pixel_format() {
    let mut args = base_args();
    args.extend(["--pixel-format", "bgr24"]);

    let args = parse_args_from(args).unwrap();

    assert_eq!(args.pixel_format, PixelFormat::Bgr24);
}

#[test]
fn invalid_non_positive_values_fail() {
    for (flag, value) in [
        ("--content-threshold", "0"),
        ("--min-scene-len", "0"),
        ("--adaptive-threshold", "-1"),
        ("--adaptive-window-width", "0"),
        ("--adaptive-min-content-val", "0"),
    ] {
        let mut args = base_args();
        if flag.starts_with("--adaptive") {
            args.extend(["--detector", "adaptive"]);
        }
        args.extend([flag, value]);

        assert!(parse_args_from(args).is_err(), "{flag} {value} should fail");
    }
}

#[test]
fn rejects_detector_specific_flags_on_wrong_detector() {
    let mut args = base_args();
    args.extend(["--detector", "adaptive", "--content-threshold", "33"]);

    assert!(parse_args_from(args)
        .unwrap_err()
        .contains("content tuning"));

    let mut args = base_args();
    args.extend(["--detector", "content", "--adaptive-threshold", "3.5"]);

    assert!(parse_args_from(args)
        .unwrap_err()
        .contains("adaptive tuning"));
}

#[test]
fn resume_report_deserializes_missing_implementation_as_rust() {
    let report = serde_json::from_str::<scene_dataset_eval::EvalReport>(
        r#"{
            "dataset": "bbc",
            "detector": "content",
            "videos": [],
            "summary": {
                "recall": 0.0,
                "precision": 0.0,
                "f1": 0.0,
                "avgElapsedMs": 0.0
            }
        }"#,
    )
    .unwrap();

    assert_eq!(report.implementation, "rust");
    assert_eq!(
        report.pyscenedetect_baseline,
        scene_dataset_eval::PYSCENEDETECT_BASELINE
    );
}

#[test]
fn suppresses_nearby_cuts_with_inclusive_window_rule() {
    assert_eq!(suppress_nearby_cuts(vec![10, 14, 30], 0), vec![10, 14, 30]);
    assert_eq!(suppress_nearby_cuts(vec![10, 14, 30], 8), vec![10, 30]);
    assert_eq!(suppress_nearby_cuts(vec![10, 18, 30], 8), vec![10, 18, 30]);
    assert_eq!(
        suppress_nearby_cuts(vec![10, 17, 18, 40], 8),
        vec![10, 18, 40]
    );
}
