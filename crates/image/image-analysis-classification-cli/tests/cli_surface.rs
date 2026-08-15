#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        image_analysis_classification_cli::LIBRARY_CRATE,
        "image-analysis-classification"
    );
    let surface = image_analysis_classification_cli::package_surface();
    assert_eq!(surface.library, "moenarch-image-analysis-classification");
    assert!(!surface.operations.is_empty());
}
