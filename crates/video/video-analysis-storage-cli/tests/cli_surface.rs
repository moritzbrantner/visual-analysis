#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        video_analysis_storage_cli::LIBRARY_CRATE,
        "video-analysis-storage"
    );
    let surface = video_analysis_storage_cli::package_surface();
    assert_eq!(surface.library, "moenarch-video-analysis-storage");
    assert!(!surface.operations.is_empty());
}
