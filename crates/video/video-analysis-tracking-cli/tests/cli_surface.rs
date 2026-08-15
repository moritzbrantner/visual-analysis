#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        video_analysis_tracking_cli::LIBRARY_CRATE,
        "video-analysis-tracking"
    );
    let surface = video_analysis_tracking_cli::package_surface();
    assert_eq!(surface.library, "moenarch-video-analysis-tracking");
    assert!(!surface.operations.is_empty());
}
