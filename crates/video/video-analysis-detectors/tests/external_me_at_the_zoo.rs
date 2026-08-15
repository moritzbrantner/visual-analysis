use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use video_analysis_core::ScenePipeline;
use video_analysis_detectors::ContentDetector;
use video_analysis_ffmpeg::{is_ffmpeg_available, is_ffprobe_available, FfmpegVideoSource};
use video_analysis_ingest::VideoFrameSource;

const ME_AT_THE_ZOO_WIKIMEDIA_URL: &str =
    "https://upload.wikimedia.org/wikipedia/commons/e/e0/Me_at_the_zoo.webm";

fn ensure_me_at_the_zoo_fixture() -> PathBuf {
    if let Some(path) = std::env::var_os("ME_AT_THE_ZOO_VIDEO_PATH") {
        let path = PathBuf::from(path);
        assert_nonempty_file(&path);
        return path;
    }

    let url = std::env::var("ME_AT_THE_ZOO_VIDEO_URL")
        .unwrap_or_else(|_| ME_AT_THE_ZOO_WIKIMEDIA_URL.to_string());
    let cache_dir = std::env::temp_dir().join("video-analysis-fixtures");
    fs::create_dir_all(&cache_dir).expect("expected fixture cache directory to be writable");
    let path = cache_dir.join("me-at-the-zoo.webm");
    if path.exists() {
        assert_nonempty_file(&path);
        return path;
    }

    download_to_path(&url, &path);
    assert_nonempty_file(&path);
    path
}

fn assert_nonempty_file(path: impl AsRef<Path>) {
    let path = path.as_ref();
    let metadata = fs::metadata(path)
        .unwrap_or_else(|err| panic!("expected `{}` metadata: {err}", path.display()));
    assert!(
        metadata.is_file() && metadata.len() > 0,
        "expected `{}` to be a non-empty file",
        path.display()
    );
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

fn download_to_path(url: &str, path: &Path) {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("expected system time after unix epoch")
        .as_nanos();
    let tmp = path.with_extension(format!("part-{unique}-{}", std::process::id()));
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|err| panic!("expected `{}` directory: {err}", parent.display()));
    }

    if path.exists() {
        assert_nonempty_file(path);
        return;
    }

    let status = if let Some(curl) = find_command("curl") {
        Command::new(curl)
            .args(["-L", "--fail", "--silent", "--show-error", "-o"])
            .arg(&tmp)
            .arg(url)
            .status()
    } else if let Some(wget) = find_command("wget") {
        Command::new(wget).arg("-O").arg(&tmp).arg(url).status()
    } else {
        panic!("required command `curl` or `wget` is unavailable");
    }
    .unwrap_or_else(|err| panic!("expected fixture download command to start: {err}"));

    assert!(
        status.success(),
        "expected fixture download from `{url}` to succeed"
    );
    if path.exists() {
        let _ = fs::remove_file(&tmp);
        assert_nonempty_file(path);
        return;
    }
    fs::rename(&tmp, path).unwrap_or_else(|err| {
        panic!(
            "expected downloaded fixture `{}` to move into place: {err}",
            path.display()
        )
    });
}

#[test]
#[ignore = "downloads Me at the zoo from Wikimedia Commons and requires ffmpeg"]
fn me_at_the_zoo_stays_a_single_scene() {
    if !(is_ffmpeg_available() && is_ffprobe_available()) {
        eprintln!("skipping Me at the Zoo scene test because ffmpeg/ffprobe is unavailable");
        return;
    }

    let path = ensure_me_at_the_zoo_fixture();
    let mut source = FfmpegVideoSource::open(&path).unwrap();
    let mut pipeline = ScenePipeline::builder()
        .detector(ContentDetector::new(27.0, 15))
        .start_in_scene(true)
        .build()
        .unwrap();

    while let Some(frame) = source.next_video_frame().unwrap() {
        pipeline.process_frame(frame).unwrap();
    }

    let result = pipeline.finish_detection().unwrap();
    assert_eq!(result.frames_processed, 285);
    assert!(
        result.cuts.is_empty(),
        "expected no scene cuts in Me at the Zoo"
    );
    assert_eq!(result.scenes.len(), 1);
    assert_eq!(result.scenes[0].start.frame_index, 0);
    assert_eq!(result.scenes[0].end.frame_index, 285);
}
