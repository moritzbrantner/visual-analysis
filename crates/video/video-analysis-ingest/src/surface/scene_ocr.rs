use std::collections::BTreeSet;

use image_analysis_ocr::{OcrBackend, OcrRequest};
use video_analysis_core::{DetectError, Observation, Result, Scene, VideoAnalysisPipeline};
use video_analysis_recognition::{
    analyze_video_text_semantics, OcrVideoAnalyzer, VideoTextSemanticAnalysis,
    VideoTextSemanticContext,
};

use crate::VideoFrameSource;

/// Scene-aware OCR output produced from one decoded video pass.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneAwareOcrAnalysis {
    /// Sorted frame indices actually submitted to OCR.
    pub sampled_frames: Vec<u64>,
    /// Raw frame-level OCR evidence emitted by the configured backend.
    pub observations: Vec<Observation>,
    /// Temporal tracks and conservative semantic roles derived from the raw evidence.
    pub text_semantics: VideoTextSemanticAnalysis,
}

/// Runs OCR on representative frames from already-detected scenes and derives
/// temporal video-text semantics from the resulting observations.
///
/// This function deliberately owns no OCR algorithm and no scene detector. The
/// caller supplies canonical scenes and an `OcrBackend`; this layer only performs
/// reusable video sampling/composition. Each scene contributes its start, midpoint,
/// and end frame (deduplicated), allowing downstream applications to avoid running
/// expensive OCR on every decoded frame while retaining deterministic coverage.
pub fn analyze_scene_aware_ocr<S, B>(
    source: &mut S,
    backend: B,
    request: OcrRequest,
    video_id: &str,
    scenes: &[Scene],
    duration_seconds: Option<f64>,
) -> Result<SceneAwareOcrAnalysis>
where
    S: VideoFrameSource,
    B: OcrBackend + 'static,
{
    if video_id.trim().is_empty() {
        return Err(DetectError::InvalidArgument(
            "scene-aware OCR requires a non-empty video id".to_string(),
        ));
    }
    if scenes.is_empty() {
        return Err(DetectError::InvalidArgument(
            "scene-aware OCR requires at least one scene".to_string(),
        ));
    }
    if duration_seconds.is_some_and(|value| !value.is_finite() || value < 0.0) {
        return Err(DetectError::InvalidArgument(
            "scene-aware OCR duration must be finite and non-negative".to_string(),
        ));
    }

    let (width, height) = source
        .source_info()
        .video
        .as_ref()
        .map(|video| (video.width, video.height))
        .ok_or_else(|| DetectError::InvalidArgument("video source has no video stream".to_string()))?;
    if width == 0 || height == 0 {
        return Err(DetectError::InvalidArgument(
            "scene-aware OCR requires non-zero video dimensions".to_string(),
        ));
    }

    let targets = representative_scene_frames(scenes)?;
    let analyzer = OcrVideoAnalyzer::new("ocr", backend).request(request);
    let mut pipeline = VideoAnalysisPipeline::builder().analyzer(analyzer).build()?;
    let mut sampled_frames = Vec::with_capacity(targets.len());

    while let Some(frame) = source.next_video_frame()? {
        let frame_index = frame.position.frame_index;
        if targets.contains(&frame_index) {
            pipeline.process_frame(frame)?;
            sampled_frames.push(frame_index);
        }
        if sampled_frames.len() == targets.len() {
            break;
        }
    }

    let analysis = pipeline.finish_analysis()?;
    let effective_duration = duration_seconds.or_else(|| {
        scenes
            .iter()
            .map(|scene| scene.end.timestamp.seconds())
            .filter(|seconds| seconds.is_finite() && *seconds >= 0.0)
            .max_by(f64::total_cmp)
    });
    let text_semantics = analyze_video_text_semantics(
        VideoTextSemanticContext {
            video_id,
            width,
            height,
            duration_seconds: effective_duration,
            scenes,
        },
        &analysis.observations,
    )?;

    Ok(SceneAwareOcrAnalysis {
        sampled_frames,
        observations: analysis.observations,
        text_semantics,
    })
}

/// Returns the deterministic representative-frame plan for a scene list.
pub fn representative_scene_frames(scenes: &[Scene]) -> Result<BTreeSet<u64>> {
    let mut result = BTreeSet::new();
    for scene in scenes {
        let start = scene.start.frame_index;
        let end = scene.end.frame_index;
        if end < start {
            return Err(DetectError::InvalidArgument(format!(
                "scene end frame {end} precedes start frame {start}"
            )));
        }
        result.insert(start);
        result.insert(start + (end - start) / 2);
        result.insert(end);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::VecDeque;
    use std::rc::Rc;

    use image_analysis_core::ImageView;
    use image_analysis_ocr::OcrDocument;
    use num_rational::Rational64;
    use video_analysis_core::{FramePosition, OwnedVideoFrame, PixelFormat};

    use super::*;
    use crate::{MediaSourceInfo, VideoStreamInfo};

    struct FixtureSource {
        info: MediaSourceInfo,
        frames: VecDeque<OwnedVideoFrame>,
    }

    impl VideoFrameSource for FixtureSource {
        fn source_info(&self) -> &MediaSourceInfo {
            &self.info
        }

        fn next_video_frame(&mut self) -> Result<Option<OwnedVideoFrame>> {
            Ok(self.frames.pop_front())
        }
    }

    struct FixtureOcr {
        calls: Rc<Cell<usize>>,
    }

    impl OcrBackend for FixtureOcr {
        fn recognize_image(
            &mut self,
            image: &ImageView<'_>,
            _request: &OcrRequest,
        ) -> Result<OcrDocument> {
            let call = self.calls.get();
            self.calls.set(call + 1);
            let text = if call < 3 { "First Slide" } else { "Second Slide" };
            OcrDocument::new(text, image.width, image.height)
        }
    }

    fn position(frame_index: u64) -> FramePosition {
        FramePosition::from_frame_index(frame_index, Rational64::new(30, 1))
    }

    fn frame(frame_index: u64) -> OwnedVideoFrame {
        OwnedVideoFrame {
            position: position(frame_index),
            width: 2,
            height: 2,
            pixel_format: PixelFormat::Rgb24,
            data: vec![0_u8; 12],
            stride: 6,
        }
    }

    fn scene(start: u64, end: u64) -> Scene {
        Scene {
            start: position(start),
            end: position(end),
        }
    }

    #[test]
    fn representative_sampling_covers_scene_edges_and_midpoints() {
        let frames = representative_scene_frames(&[scene(0, 4), scene(5, 8)]).unwrap();
        assert_eq!(frames.into_iter().collect::<Vec<_>>(), vec![0, 2, 4, 5, 6, 8]);
    }

    #[test]
    fn scene_aware_ocr_runs_only_on_representative_frames_and_builds_tracks() -> Result<()> {
        let info = MediaSourceInfo::recorded("fixture").with_video(VideoStreamInfo {
            width: 2,
            height: 2,
            frame_rate: Some(Rational64::new(30, 1)),
            pixel_format: PixelFormat::Rgb24,
        });
        let frames = (0..=8).map(frame).collect::<VecDeque<_>>();
        let mut source = FixtureSource { info, frames };
        let calls = Rc::new(Cell::new(0));
        let scenes = vec![scene(0, 4), scene(5, 8)];

        let analysis = analyze_scene_aware_ocr(
            &mut source,
            FixtureOcr {
                calls: calls.clone(),
            },
            OcrRequest::default(),
            "video-1",
            &scenes,
            Some(8.0 / 30.0),
        )?;

        assert_eq!(analysis.sampled_frames, vec![0, 2, 4, 5, 6, 8]);
        assert_eq!(calls.get(), 6);
        assert_eq!(analysis.observations.len(), 6);
        assert_eq!(analysis.text_semantics.tracks.len(), 2);
        assert_eq!(analysis.text_semantics.tracks[0].sample_count, 3);
        assert_eq!(analysis.text_semantics.tracks[1].sample_count, 3);
        assert!(analysis
            .text_semantics
            .tracks
            .iter()
            .all(|track| track.scene_indices.len() == 1));
        Ok(())
    }

    #[test]
    fn scene_aware_ocr_rejects_missing_scene_plan() {
        let info = MediaSourceInfo::recorded("fixture").with_video(VideoStreamInfo {
            width: 2,
            height: 2,
            frame_rate: Some(Rational64::new(30, 1)),
            pixel_format: PixelFormat::Rgb24,
        });
        let mut source = FixtureSource {
            info,
            frames: VecDeque::new(),
        };
        let error = analyze_scene_aware_ocr(
            &mut source,
            FixtureOcr {
                calls: Rc::new(Cell::new(0)),
            },
            OcrRequest::default(),
            "video-1",
            &[],
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("at least one scene"));
    }
}
