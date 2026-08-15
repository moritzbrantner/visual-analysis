#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        image_analysis_embeddings_cli::LIBRARY_CRATE,
        "image-analysis-embeddings"
    );
    let surface = image_analysis_embeddings_cli::package_surface();
    assert_eq!(surface.library, "moenarch-image-analysis-embeddings");
    assert!(!surface.operations.is_empty());
}
