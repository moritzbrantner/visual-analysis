use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use image_analysis_segmentation::{BinaryMask, ImageSegment};
use model_runtime::{DownloadedModel, HuggingFaceModelSpec, ModelTask};
use video_analysis_core::{BoundingBox, Result, VideoFrame};
use video_analysis_ffmpeg::{is_ffmpeg_available, is_ffprobe_available, FfmpegVideoSource};
use video_analysis_ingest::VideoFrameSource;
use video_analysis_recognition::{ExternalCommandModel, RawPrediction, VisionModelBackend};
use video_analysis_segmentation::{
    TrackedSegment, VideoSegmentationBackend, VideoSegmentationRequest, VideoSegmenter,
};

const ME_AT_THE_ZOO_WIKIMEDIA_URL: &str =
    "https://upload.wikimedia.org/wikipedia/commons/e/e0/Me_at_the_zoo.webm";

fn find_python_with_modules(modules: &[&str]) -> Option<PathBuf> {
    let import = format!("import {}", modules.join(", "));
    [
        PathBuf::from("python3"),
        PathBuf::from("python"),
        PathBuf::from(".external-test-tools/model-python-venv/bin/python"),
    ]
    .into_iter()
    .find(|candidate| {
        Command::new(candidate)
            .args(["-c", &import])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    })
}

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

fn assert_nonempty_file(path: impl AsRef<std::path::Path>) {
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
    let path = std::path::Path::new(command);
    if path.components().count() > 1 && path.is_file() {
        return Some(path.to_path_buf());
    }
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|dir| dir.join(command))
            .find(|candidate| candidate.is_file())
    })
}

fn download_to_path(url: &str, path: &std::path::Path) {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("expected system time after unix epoch")
        .as_nanos();
    let tmp = path.with_extension(format!("part-{unique}-{}", std::process::id()));
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|err| panic!("expected `{}` directory: {err}", parent.display()));
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
    fs::rename(&tmp, path).unwrap_or_else(|err| {
        panic!(
            "expected downloaded fixture `{}` to move into place: {err}",
            path.display()
        )
    });
}

struct PersonMaskBackend<B> {
    detector: B,
    min_score: f32,
}

impl<B> PersonMaskBackend<B> {
    fn new(detector: B) -> Self {
        Self {
            detector,
            min_score: 0.5,
        }
    }
}

impl<B: VisionModelBackend> VideoSegmentationBackend for PersonMaskBackend<B> {
    fn segment_frame(
        &mut self,
        frame: &VideoFrame<'_>,
        request: &VideoSegmentationRequest,
    ) -> Result<Vec<TrackedSegment>> {
        let object_id = request
            .prompt
            .object_id
            .clone()
            .unwrap_or_else(|| "person".to_string());
        let mut segments = Vec::new();

        for prediction in self.detector.predict_frame(frame)? {
            if prediction
                .label
                .as_deref()
                .map(|label| !label.eq_ignore_ascii_case("person"))
                .unwrap_or(true)
            {
                continue;
            }
            if prediction.score.unwrap_or(0.0) < self.min_score {
                continue;
            }
            let Some(region) = raw_bbox_to_region(&prediction, frame) else {
                continue;
            };
            let mask = BinaryMask::filled_rect(frame.width, frame.height, region)?;
            if mask.active_pixels() < request.min_mask_pixels.max(1) {
                continue;
            }
            let segment = ImageSegment::new(mask)?.label("person");
            segments.push(TrackedSegment::new(segment).object_id(object_id.clone()));
        }

        Ok(segments)
    }
}

fn raw_bbox_to_region(prediction: &RawPrediction, frame: &VideoFrame<'_>) -> Option<BoundingBox> {
    let region = prediction.region?;
    let x = region.x?.round().max(0.0) as u32;
    let y = region.y?.round().max(0.0) as u32;
    let mut width = region.width?.round().max(0.0) as u32;
    let mut height = region.height?.round().max(0.0) as u32;
    if x >= frame.width || y >= frame.height {
        return None;
    }
    width = width.min(frame.width.saturating_sub(x));
    height = height.min(frame.height.saturating_sub(y));
    (width > 0 && height > 0)
        .then(|| BoundingBox::new(x, y, width, height).ok())
        .flatten()
}

fn opencv_person_detector_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../scripts/opencv_person_detector.py")
}

fn dummy_model() -> DownloadedModel {
    DownloadedModel {
        spec: HuggingFaceModelSpec::new("opencv-haar-person", ModelTask::ObjectDetection)
            .name("opencv-haar-person"),
        files: BTreeMap::new(),
    }
}

#[test]
#[ignore = "downloads Me at the zoo from Wikimedia Commons and requires ffmpeg + python3 + cv2"]
fn me_at_the_zoo_person_count_stays_at_one() {
    if !(is_ffmpeg_available() && is_ffprobe_available()) {
        eprintln!("skipping Me at the Zoo segmentation test because ffmpeg/ffprobe is unavailable");
        return;
    }

    let Some(python) = find_python_with_modules(&["cv2", "numpy"]) else {
        eprintln!(
            "skipping Me at the Zoo segmentation test because python cv2/numpy is unavailable"
        );
        return;
    };

    let path = ensure_me_at_the_zoo_fixture();
    let script = opencv_person_detector_script();
    let backend =
        ExternalCommandModel::new(python, dummy_model()).arg(script.display().to_string());
    let mut segmenter = VideoSegmenter::new(PersonMaskBackend::new(backend))
        .request(VideoSegmentationRequest::default().min_mask_pixels(24 * 24));
    let mut source = FfmpegVideoSource::open(&path).unwrap();

    let mut sampled_frames = 0_u32;
    let mut frames_with_person = 0_u32;
    let mut max_person_count = 0_usize;
    while let Some(frame) = source.next_video_frame().unwrap() {
        if frame.position.frame_index > 120 {
            break;
        }
        if frame.position.frame_index % 30 != 0 {
            continue;
        }

        sampled_frames += 1;
        let output = segmenter.process_frame(&frame.as_frame()).unwrap();
        let person_count = output
            .segments
            .iter()
            .filter(|segment| segment.object_id.as_deref() == Some("person"))
            .count();
        if person_count > 0 {
            frames_with_person += 1;
        }
        max_person_count = max_person_count.max(person_count);
    }

    assert_eq!(sampled_frames, 5);
    assert!(frames_with_person >= 4);
    assert_eq!(max_person_count, 1);
}
