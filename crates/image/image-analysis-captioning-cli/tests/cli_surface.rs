#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        image_analysis_captioning_cli::LIBRARY_CRATE,
        "image-analysis-captioning"
    );
    let surface = image_analysis_captioning_cli::package_surface();
    assert_eq!(surface.library, "moenarch-image-analysis-captioning");
    assert!(!surface.operations.is_empty());
}
