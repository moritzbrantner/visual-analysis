#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        video_analysis_editing_cli::LIBRARY_CRATE,
        "video-analysis-editing"
    );
    let surface = video_analysis_editing_cli::package_surface();
    assert_eq!(surface.library, "moenarch-video-analysis-editing");
    assert!(!surface.operations.is_empty());
}
