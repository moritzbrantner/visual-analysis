use scenedetect_core::{
    boundary_review_from_content_detection_stats, detect_content_stats,
    scene_list_from_content_detection_stats, BoundaryReview, BoundaryReviewOptions,
    ContentDetectionStats, ContentDetectorConfig, DetectionOptions, Frame, FrameIndex, FrameRate,
    FrameSource, SceneList,
};
use video_analysis_core::{DetectError, Result, VideoSource};

/// Canonical scene-analysis outputs derived from one Detection Stats pass.
#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalContentAnalysis {
    /// Reusable typed Detection Stats produced by `scenedetect-core`.
    pub detection_stats: ContentDetectionStats,
    /// Scene List derived from the Detection Stats.
    pub scene_list: SceneList,
    /// Boundary Candidate review derived from the same Detection Stats.
    pub boundary_review: BoundaryReview,
}

/// Runs the canonical content detector once over an owned visual source.
///
/// The visual source remains responsible for decoding, pixel format, timestamps,
/// and frame ownership. This adapter converts each decoded RGB/BGR frame into the
/// narrow `scenedetect-core::FrameSource` contract and then derives all public
/// scene outputs from the resulting Detection Stats.
pub fn analyze_content_source<S>(
    source: S,
    config: ContentDetectorConfig,
    options: DetectionOptions,
    review_options: BoundaryReviewOptions,
) -> Result<CanonicalContentAnalysis>
where
    S: VideoSource,
{
    let detection_stats = detect_content_stats(VisualFrameSource::new(source), config, options)
        .map_err(|error| DetectError::Source(error.to_string()))?;
    let scene_list = scene_list_from_content_detection_stats(&detection_stats);
    let boundary_review =
        boundary_review_from_content_detection_stats(&detection_stats, review_options);

    Ok(CanonicalContentAnalysis {
        detection_stats,
        scene_list,
        boundary_review,
    })
}

struct VisualFrameSource<S> {
    source: S,
    frame_rate: FrameRate,
}

impl<S> VisualFrameSource<S>
where
    S: VideoSource,
{
    fn new(source: S) -> Self {
        let rate = source.frame_rate();
        let frame_rate = FrameRate(*rate.numer() as f64 / *rate.denom() as f64);
        Self { source, frame_rate }
    }
}

impl<S> FrameSource for VisualFrameSource<S>
where
    S: VideoSource,
{
    fn frame_rate(&self) -> FrameRate {
        self.frame_rate
    }

    fn next_frame(&mut self) -> scenedetect_core::Result<Option<Frame>> {
        let Some(frame) = self
            .source
            .next_frame()
            .map_err(|error| scenedetect_core::SceneDetectError::FrameSource(error.to_string()))?
        else {
            return Ok(None);
        };

        let borrowed = frame.as_frame();
        let rgb = (0..borrowed.height)
            .flat_map(|y| (0..borrowed.width).flat_map(move |x| borrowed.pixel_rgb(x, y)))
            .collect();

        Ok(Some(Frame {
            index: FrameIndex(borrowed.position.frame_index),
            width: borrowed.width,
            height: borrowed.height,
            rgb,
        }))
    }
}
