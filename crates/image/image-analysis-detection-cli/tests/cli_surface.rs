#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        image_analysis_detection_cli::LIBRARY_CRATE,
        "image-analysis-detection"
    );
    let surface = image_analysis_detection_cli::package_surface();
    assert_eq!(surface.library, "moenarch-image-analysis-detection");
    assert!(!surface.operations.is_empty());
}
