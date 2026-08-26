use std::cell::Cell;
use std::rc::Rc;

use num_rational::Rational64;
use scenedetect_core::{
    BoundaryCandidateStatus, BoundaryReviewOptions, ContentDetectorConfig, DetectionOptions,
    DetectionStatsDecision, FrameIndex, MinSceneLenPolicy, SceneSpan,
};
use video_analysis_core::{
    FramePosition, OwnedVideoFrame, PixelFormat, Result, VideoSource,
};
use video_analysis_detectors::analyze_content_source;

struct CountingVideoSource {
    frame_rate: Rational64,
    frames: std::vec::IntoIter<OwnedVideoFrame>,
    reads: Rc<Cell<usize>>,
}

impl CountingVideoSource {
    fn new(frames: Vec<OwnedVideoFrame>, reads: Rc<Cell<usize>>) -> Self {
        Self {
            frame_rate: Rational64::new(10, 1),
            frames: frames.into_iter(),
            reads,
        }
    }
}

impl VideoSource for CountingVideoSource {
    fn next_frame(&mut self) -> Result<Option<OwnedVideoFrame>> {
        self.reads.set(self.reads.get() + 1);
        Ok(self.frames.next())
    }

    fn frame_rate(&self) -> Rational64 {
        self.frame_rate
    }
}

fn solid_frame(index: u64, pixel_format: PixelFormat, color: [u8; 3]) -> OwnedVideoFrame {
    let width = 2;
    let height = 2;
    let pixel = match pixel_format {
        PixelFormat::Rgb24 => color,
        PixelFormat::Bgr24 => [color[2], color[1], color[0]],
    };
    let mut data = Vec::new();
    for _ in 0..(width * height) {
        data.extend_from_slice(&pixel);
    }
    OwnedVideoFrame {
        position: FramePosition::from_frame_index(index, Rational64::new(10, 1)),
        width,
        height,
        pixel_format,
        data,
        stride: width as usize * 3,
    }
}

fn options() -> DetectionOptions {
    DetectionOptions {
        min_scene_len: 1,
        min_scene_len_policy: MinSceneLenPolicy::Suppress,
    }
}

#[test]
fn canonical_content_analysis_consumes_rgb_source_once() {
    let reads = Rc::new(Cell::new(0));
    let source = CountingVideoSource::new(
        vec![
            solid_frame(0, PixelFormat::Rgb24, [0, 0, 0]),
            solid_frame(1, PixelFormat::Rgb24, [255, 255, 255]),
            solid_frame(2, PixelFormat::Rgb24, [255, 255, 255]),
        ],
        Rc::clone(&reads),
    );

    let analysis = analyze_content_source(
        source,
        ContentDetectorConfig {
            threshold: 20.0,
            ..Default::default()
        },
        options(),
        BoundaryReviewOptions::default(),
    )
    .unwrap();

    assert_eq!(reads.get(), 4, "three frames plus one EOF read");
    assert_eq!(analysis.detection_stats.total_frames, 3);
    assert_eq!(analysis.detection_stats.rows.len(), 3);
    assert_eq!(
        analysis.detection_stats.rows[1].decision,
        DetectionStatsDecision::Accepted
    );
    assert_eq!(
        analysis.scene_list.scenes,
        vec![
            SceneSpan {
                start: FrameIndex(0),
                end: FrameIndex(1),
            },
            SceneSpan {
                start: FrameIndex(1),
                end: FrameIndex(3),
            },
        ]
    );
    assert!(analysis.boundary_review.candidates.iter().any(|candidate| {
        candidate.status == BoundaryCandidateStatus::Accepted && candidate.frame == FrameIndex(1)
    }));
}

#[test]
fn canonical_content_analysis_normalizes_bgr_through_visual_frame_contract() {
    let reads = Rc::new(Cell::new(0));
    let source = CountingVideoSource::new(
        vec![
            solid_frame(0, PixelFormat::Bgr24, [255, 0, 0]),
            solid_frame(1, PixelFormat::Bgr24, [0, 255, 0]),
        ],
        Rc::clone(&reads),
    );

    let analysis = analyze_content_source(
        source,
        ContentDetectorConfig {
            threshold: 20.0,
            ..Default::default()
        },
        options(),
        BoundaryReviewOptions::default(),
    )
    .unwrap();

    assert_eq!(reads.get(), 3, "two frames plus one EOF read");
    assert_eq!(analysis.detection_stats.total_frames, 2);
    assert_eq!(analysis.detection_stats.rows[1].frame, FrameIndex(1));
    assert!(analysis.detection_stats.rows[1].score > 20.0);
}
