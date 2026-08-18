#[test]
fn package_endpoint_reports_wrapped_library() {
    let response = video_analysis_recognition_server::response_for("GET", "/api/package", "");
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("video-analysis-recognition"));
}

#[test]
fn run_endpoint_calls_library_surface() {
    let response = video_analysis_recognition_server::response_for(
        "POST",
        "/api/run",
        r#"{"operation":"describe","input":{"includeOperations":true}}"#,
    );
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains(r#""operation""#));
}
