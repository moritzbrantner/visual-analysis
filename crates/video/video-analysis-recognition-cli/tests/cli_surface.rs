#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        video_analysis_recognition_cli::LIBRARY_CRATE,
        "video-analysis-recognition"
    );
    let surface = video_analysis_recognition_cli::package_surface();
    assert_eq!(surface.library, "moenarch-video-analysis-recognition");
    assert!(!surface.operations.is_empty());
}
