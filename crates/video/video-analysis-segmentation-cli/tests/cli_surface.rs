#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        video_analysis_segmentation_cli::LIBRARY_CRATE,
        "video-analysis-segmentation"
    );
    let surface = video_analysis_segmentation_cli::package_surface();
    assert_eq!(surface.library, "moenarch-video-analysis-segmentation");
    assert!(!surface.operations.is_empty());
}
