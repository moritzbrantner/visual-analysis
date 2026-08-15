#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        video_analysis_ffmpeg_cli::LIBRARY_CRATE,
        "video-analysis-ffmpeg"
    );
    let surface = video_analysis_ffmpeg_cli::package_surface();
    assert_eq!(surface.library, "moenarch-video-analysis-ffmpeg");
    assert!(!surface.operations.is_empty());
}
