#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        video_analysis_split_cli::LIBRARY_CRATE,
        "video-analysis-split"
    );
    let surface = video_analysis_split_cli::package_surface();
    assert_eq!(surface.library, "moenarch-video-analysis-split");
    assert!(!surface.operations.is_empty());
}
