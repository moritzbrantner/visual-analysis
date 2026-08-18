#[test]
fn package_endpoint_reports_wrapped_library() {
    let response = image_analysis_processing_server::response_for("GET", "/api/package", "");
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("image-analysis-processing"));
}

#[test]
fn run_endpoint_calls_library_surface() {
    let response = image_analysis_processing_server::response_for(
        "POST",
        "/api/run",
        r#"{"operation":"describe","input":{"includeOperations":true}}"#,
    );
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains(r#""operation""#));
}
