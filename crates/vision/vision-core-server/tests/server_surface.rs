#[test]
fn server_surface_exposes_package_metadata() {
    let response = vision_core_server::response_for("GET", "/api/package", "");
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("vision-core"));
}

#[test]
fn run_endpoint_calls_library_surface() {
    let response = vision_core_server::response_for(
        "POST",
        "/api/run",
        r#"{"operation":"vision.validateIdentityMatch","input":{"identityMatch":{"referenceId":"alice","score":0.99,"attributes":{}}}}"#,
    );
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains(r#""operation""#));
}
