#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        video_analysis_detectors_cli::LIBRARY_CRATE,
        "video-analysis-detectors"
    );
    let surface = video_analysis_detectors_cli::package_surface();
    assert_eq!(surface.library, "moenarch-video-analysis-detectors");
    assert!(!surface.operations.is_empty());
}
