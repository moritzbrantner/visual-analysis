#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        video_analysis_dataset_cli::LIBRARY_CRATE,
        "video-analysis-dataset"
    );
    let surface = video_analysis_dataset_cli::package_surface();
    assert_eq!(surface.library, "moenarch-video-analysis-dataset");
    assert!(!surface.operations.is_empty());
}
