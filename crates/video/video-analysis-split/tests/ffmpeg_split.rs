#[cfg(feature = "external-tests")]
mod external {
    use std::path::{Path, PathBuf};

    use num_rational::Rational64;
    use video_analysis_core::{FramePosition, Scene};
    use video_analysis_split::{split_video_ffmpeg, SplitOptions};

    fn require_command(command: &str) {
        find_command(command)
            .unwrap_or_else(|| panic!("required command `{command}` is unavailable"));
    }

    fn find_command(command: &str) -> Option<PathBuf> {
        let path = Path::new(command);
        if path.components().count() > 1 && path.is_file() {
            return Some(path.to_path_buf());
        }
        std::env::var_os("PATH").and_then(|paths| {
            std::env::split_paths(&paths)
                .map(|dir| dir.join(command))
                .find(|candidate| candidate.is_file())
        })
    }

    fn assert_nonempty_file(path: impl AsRef<Path>) {
        let path = path.as_ref();
        let metadata = std::fs::metadata(path)
            .unwrap_or_else(|err| panic!("expected `{}` metadata: {err}", path.display()));
        assert!(
            metadata.is_file() && metadata.len() > 0,
            "expected `{}` to be a non-empty file",
            path.display()
        );
    }

    #[test]
    #[ignore = "requires real ffmpeg and ffprobe"]
    fn real_ffmpeg_splits_generated_two_scene_video() {
        require_command("ffmpeg");
        require_command("ffprobe");

        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("two-scenes.mp4");
        video_analysis_ffmpeg::write_two_scene_test_video(&input).unwrap();

        let rate = Rational64::new(10, 1);
        let scenes = vec![
            Scene {
                start: FramePosition::from_frame_index(0, rate),
                end: FramePosition::from_frame_index(4, rate),
            },
            Scene {
                start: FramePosition::from_frame_index(4, rate),
                end: FramePosition::from_frame_index(8, rate),
            },
        ];
        let options = SplitOptions {
            output_dir: dir.path().join("splits"),
            ..SplitOptions::default()
        };

        let outputs = split_video_ffmpeg(&input, &scenes, &options).unwrap();
        assert_eq!(outputs.len(), 2);
        for output in outputs {
            assert_nonempty_file(output);
        }
    }
}
