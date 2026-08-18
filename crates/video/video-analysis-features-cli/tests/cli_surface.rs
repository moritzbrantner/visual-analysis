#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        video_analysis_features_cli::LIBRARY_CRATE,
        "video-analysis-features"
    );
    let surface = video_analysis_features_cli::package_surface();
    assert_eq!(surface.library, "moenarch-video-analysis-features");
    assert!(!surface.operations.is_empty());
}
