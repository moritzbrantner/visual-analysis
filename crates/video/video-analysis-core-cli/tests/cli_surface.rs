#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        video_analysis_core_cli::LIBRARY_CRATE,
        "video-analysis-core"
    );
    let surface = video_analysis_core_cli::package_surface();
    assert_eq!(surface.library, "moenarch-video-analysis-core");
    assert!(!surface.operations.is_empty());
}
