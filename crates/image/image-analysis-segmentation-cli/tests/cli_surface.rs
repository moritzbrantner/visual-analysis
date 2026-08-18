#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        image_analysis_segmentation_cli::LIBRARY_CRATE,
        "image-analysis-segmentation"
    );
    let surface = image_analysis_segmentation_cli::package_surface();
    assert_eq!(surface.library, "moenarch-image-analysis-segmentation");
    assert!(!surface.operations.is_empty());
}
