#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        image_analysis_synthesis_cli::LIBRARY_CRATE,
        "image-analysis-synthesis"
    );
    let surface = image_analysis_synthesis_cli::package_surface();
    assert_eq!(surface.library, "moenarch-image-analysis-synthesis");
    assert!(!surface.operations.is_empty());
}
