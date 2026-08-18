#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(image_analysis_ocr_cli::LIBRARY_CRATE, "image-analysis-ocr");
    let surface = image_analysis_ocr_cli::package_surface();
    assert_eq!(surface.library, "moenarch-image-analysis-ocr");
    assert!(!surface.operations.is_empty());
}
