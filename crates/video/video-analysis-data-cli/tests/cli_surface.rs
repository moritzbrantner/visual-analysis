#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        video_analysis_data_cli::LIBRARY_CRATE,
        "video-analysis-data"
    );
    let surface = video_analysis_data_cli::package_surface();
    assert_eq!(surface.library, "moenarch-video-analysis-data");
    assert!(!surface.operations.is_empty());
}
