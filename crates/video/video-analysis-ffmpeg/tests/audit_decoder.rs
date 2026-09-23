//! Real decoder regressions; no downloaded media or learned models.
#![cfg(feature = "ffmpeg-tests")]
use std::io::Write;
use std::process::{Command, Stdio};
use video_analysis_core::VideoSource;
use video_analysis_ffmpeg::{FfmpegSourceOptions, FfmpegVideoSource};

fn fixture(width: u32, height: u32) -> tempfile::NamedTempFile {
    let file = tempfile::Builder::new().suffix(".mkv").tempfile().unwrap();
    let mut child = Command::new("ffmpeg")
        .args([
            "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
        ])
        .arg(format!("{width}x{height}"))
        .args(["-r", "30", "-i", "pipe:0", "-c:v", "ffv1", "-threads", "1"])
        .arg(file.path())
        .stdin(Stdio::piped())
        .spawn()
        .expect("FFmpeg is required by ffmpeg-tests");
    {
        let mut input = child.stdin.take().unwrap();
        for color in [[255, 0, 0], [0, 255, 0], [0, 0, 255]] {
            input
                .write_all(&color.repeat((width * height) as usize))
                .unwrap();
        }
    }
    assert!(child.wait().unwrap().success());
    file
}

#[test]
fn resized_frames_keep_color_identity_at_awkward_aspect_ratios() {
    for (width, height, target) in [(1920, 1080, 500), (720, 1280, 321), (9, 16, 4)] {
        let file = fixture(width, height);
        let mut source = FfmpegVideoSource::open_path_with_options(
            file.path(),
            FfmpegSourceOptions::recorded().resize_width(target),
        )
        .unwrap();
        for expected in [[255, 0, 0], [0, 255, 0], [0, 0, 255]] {
            let frame = source
                .next_frame()
                .unwrap()
                .expect("all three frames survive");
            assert_eq!(frame.width, target);
            assert_eq!(frame.data.len(), frame.stride * frame.height as usize);
            for pixel in frame.data.chunks_exact(3) {
                assert!(
                    pixel
                        .iter()
                        .zip(expected)
                        .all(|(actual, expected)| if expected == 255 {
                            *actual > 140
                        } else {
                            *actual < 80
                        }),
                    "{width}x{height}->{target}: expected {expected:?}, got {pixel:?}"
                );
            }
        }
        assert!(source.next_frame().unwrap().is_none());
        assert!(source.next_frame().unwrap().is_none());
    }
}

#[test]
fn decoder_failure_before_first_frame_is_not_successful_eof() {
    let file = fixture(16, 16);
    let options = FfmpegSourceOptions::recorded()
        .extra_output_arg("-vf")
        .extra_output_arg("nonexistent_visual_audit_filter");
    let mut source = FfmpegVideoSource::open_path_with_options(file.path(), options).unwrap();
    let error = source
        .next_frame()
        .expect_err("decoder errors must reach the caller");
    assert!(
        error
            .to_string()
            .contains("nonexistent_visual_audit_filter"),
        "{error}"
    );
    assert!(
        source.next_frame().is_err(),
        "a failed decoder must not turn into successful EOF"
    );
}
