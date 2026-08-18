#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        image_analysis_core_cli::LIBRARY_CRATE,
        "image-analysis-core"
    );
    let surface = image_analysis_core_cli::package_surface();
    assert_eq!(surface.library, "moenarch-image-analysis-core");
    assert!(!surface.operations.is_empty());
}
