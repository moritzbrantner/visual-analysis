use scenedetect_core::{
    boundary_review_from_content_detection_stats, detect_content_stats,
    scene_list_from_content_detection_stats, BoundaryReview, BoundaryReviewOptions,
    ContentDetectionStats, ContentDetectorConfig, DetectionOptions, Frame, FrameIndex, FrameRate,
    FrameSource, SceneList,
};
use video_analysis_core::{DetectError, PixelFormat, Result, VideoFrame, VideoSource};

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

        VideoFrame::packed(
            frame.position,
            frame.width,
            frame.height,
            frame.pixel_format,
            &frame.data,
            frame.stride,
        )
        .map_err(|error| scenedetect_core::SceneDetectError::FrameSource(error.to_string()))?;
        let row_bytes = frame.width as usize * 3;
        let mut rgb = frame.data;
        if frame.stride != row_bytes {
            // Repack padded rows in place; no second full-frame allocation.
            for y in 1..frame.height as usize {
                rgb.copy_within(
                    y * frame.stride..y * frame.stride + row_bytes,
                    y * row_bytes,
                );
            }
        }
        rgb.truncate(row_bytes * frame.height as usize);
        if frame.pixel_format == PixelFormat::Bgr24 {
            for pixel in rgb.chunks_exact_mut(3) {
                pixel.swap(0, 2);
            }
        }
        Ok(Some(Frame {
            index: FrameIndex(frame.position.frame_index),
            width: frame.width,
            height: frame.height,
            rgb,
        }))
    }
}

/// Executes any canonical detector configuration in one decoded-source pass.
/// The algorithms and their parameter semantics remain owned by scenedetect-core.
pub fn detect_source<S: VideoSource>(
    source: S,
    config: scenedetect_core::DetectorConfig,
    options: DetectionOptions,
) -> Result<scenedetect_core::DetectionResult> {
    scenedetect_core::detect_scenes(config, VisualFrameSource::new(source), options)
        .map_err(|error| DetectError::Source(error.to_string()))
}
