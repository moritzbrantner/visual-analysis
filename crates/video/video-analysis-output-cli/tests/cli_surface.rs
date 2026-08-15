#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        video_analysis_output_cli::LIBRARY_CRATE,
        "video-analysis-output"
    );
    let surface = video_analysis_output_cli::package_surface();
    assert_eq!(surface.library, "moenarch-video-analysis-output");
    assert!(!surface.operations.is_empty());
}
