#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        video_analysis_synthesis_cli::LIBRARY_CRATE,
        "video-analysis-synthesis"
    );
    let surface = video_analysis_synthesis_cli::package_surface();
    assert_eq!(surface.library, "moenarch-video-analysis-synthesis");
    assert!(!surface.operations.is_empty());
}
