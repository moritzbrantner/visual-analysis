#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        video_analysis_transform_cli::LIBRARY_CRATE,
        "video-analysis-transform"
    );
    let surface = video_analysis_transform_cli::package_surface();
    assert_eq!(surface.library, "moenarch-video-analysis-transform");
    assert!(!surface.operations.is_empty());
}
