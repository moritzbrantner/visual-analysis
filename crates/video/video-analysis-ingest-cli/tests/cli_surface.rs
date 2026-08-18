#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        video_analysis_ingest_cli::LIBRARY_CRATE,
        "video-analysis-ingest"
    );
    let surface = video_analysis_ingest_cli::package_surface();
    assert_eq!(surface.library, "moenarch-video-analysis-ingest");
    assert!(!surface.operations.is_empty());
}
