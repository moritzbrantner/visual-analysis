#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        image_analysis_processing_cli::LIBRARY_CRATE,
        "image-analysis-processing"
    );
    let surface = image_analysis_processing_cli::package_surface();
    assert_eq!(surface.library, "moenarch-image-analysis-processing");
    assert!(!surface.operations.is_empty());
}
