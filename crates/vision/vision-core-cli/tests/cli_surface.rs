#[test]
fn cli_surface_wraps_library_surface() {
    let surface = vision_core_cli::package_surface();
    assert_eq!(surface.library, "moenarch-vision-core");
    assert!(surface
        .operations
        .iter()
        .any(|operation| operation.id.as_str() == "vision.validateDetection"));
}
