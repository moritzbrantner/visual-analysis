#![cfg(feature = "ocr")]
use image_analysis_core::ImageView;
use image_analysis_ocr::{OcrBackend, OcrDocument, OcrRequest, OcrTextBlock, OcrTextLine};
use num_rational::Rational64;
use std::cell::Cell;
use std::rc::Rc;
use video_analysis_core::{
    BoundingBox, FramePosition, OwnedVideoFrame, PixelFormat, Result, Scene,
};
use video_analysis_ingest::surface::scene_ocr::{
    analyze_scene_aware_ocr, representative_scene_frames,
};
use video_analysis_ingest::{MediaSourceInfo, VideoFrameSource, VideoStreamInfo};
use video_analysis_recognition::VideoTextRole;
struct Source {
    info: MediaSourceInfo,
    index: u64,
    repeated_first: bool,
}
impl VideoFrameSource for Source {
    fn source_info(&self) -> &MediaSourceInfo {
        &self.info
    }
    fn next_video_frame(&mut self) -> Result<Option<OwnedVideoFrame>> {
        if self.index == 300 {
            return Ok(None);
        }
        let index = self.index;
        if !self.repeated_first {
            self.repeated_first = true;
        } else {
            self.index += 1;
        }
        Ok(Some(OwnedVideoFrame {
            position: position(index),
            width: 160,
            height: 100,
            pixel_format: PixelFormat::Rgb24,
            data: vec![0; 160 * 100 * 3],
            stride: 480,
        }))
    }
}
fn position(index: u64) -> FramePosition {
    FramePosition::from_frame_index(index, Rational64::new(5, 1))
}
struct Slide(Rc<Cell<usize>>);
impl OcrBackend for Slide {
    fn recognize_image(&mut self, image: &ImageView<'_>, _: &OcrRequest) -> Result<OcrDocument> {
        self.0.set(self.0.get() + 1);
        let mut block = OcrTextBlock::paragraph("Title\nFirst topic\nSecond topic")?;
        for (text, y) in [("Title", 10), ("First topic", 35), ("Second topic", 60)] {
            block = block.line(OcrTextLine::new(text)?.region(BoundingBox::new(10, y, 120, 10)?));
        }
        Ok(OcrDocument::new(
            "Title\nFirst topic\nSecond topic",
            image.width,
            image.height,
        )?
        .block(block))
    }
}
#[test]
fn long_scene_composition_retains_slide_tracks_without_duplicate_model_work() {
    let scenes = [Scene {
        start: position(0),
        end: position(300),
    }];
    let targets = representative_scene_frames(&scenes).unwrap();
    let calls = Rc::new(Cell::new(0));
    let mut source = Source {
        info: MediaSourceInfo::recorded("long-slide").with_video(VideoStreamInfo {
            width: 160,
            height: 100,
            pixel_format: PixelFormat::Rgb24,
            frame_rate: Some(Rational64::new(5, 1)),
        }),
        index: 0,
        repeated_first: false,
    };
    let result = analyze_scene_aware_ocr(
        &mut source,
        Slide(calls.clone()),
        OcrRequest::default(),
        "slide",
        &scenes,
        Some(120.0),
    )
    .unwrap();
    assert_eq!(
        result.sampled_frames,
        targets.into_iter().collect::<Vec<_>>()
    );
    assert_eq!(calls.get(), result.sampled_frames.len());
    assert!(
        calls.get() <= 18,
        "OCR must not accidentally run on all 300 frames"
    );
    assert_eq!(result.observations.len(), 3 * calls.get());
    assert_eq!(result.text_semantics.tracks.len(), 3);
    assert!(result
        .text_semantics
        .tracks
        .iter()
        .all(|track| track.role == VideoTextRole::PresentationSlide
            && track.sample_count as usize == calls.get()));
}
