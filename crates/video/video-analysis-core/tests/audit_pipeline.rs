use num_rational::Rational64;
use std::cell::RefCell;
use std::rc::Rc;
use video_analysis_core::*;

fn frame(index: u64, value: u8) -> OwnedVideoFrame {
    OwnedVideoFrame {
        position: FramePosition::from_frame_index(index, Rational64::new(30, 1)),
        width: 8,
        height: 4,
        pixel_format: PixelFormat::Rgb24,
        data: vec![value; 8 * 4 * 3],
        stride: 24,
    }
}
fn run(pipeline: &mut ScenePipeline, colors: &[u8]) -> DetectionResult {
    pipeline.reset();
    for (index, color) in colors.iter().enumerate() {
        pipeline.process_frame(frame(index as u64, *color)).unwrap();
    }
    pipeline.finish_detection().unwrap()
}
fn pipeline() -> ScenePipeline {
    ScenePipeline::builder()
        .detector(ContentDetector::new(10.0, 1))
        .start_in_scene(true)
        .build()
        .unwrap()
}
#[test]
fn detector_history_and_emitted_cuts_do_not_survive_reset() {
    let mut reused = pipeline();
    let first = run(&mut reused, &[0, 0, 255, 255]);
    assert!(!first.cuts.is_empty());
    assert_eq!(run(&mut reused, &[0, 0, 255, 255]), first);
    assert_eq!(
        run(&mut reused, &[255, 0, 0]),
        run(&mut pipeline(), &[255, 0, 0])
    );
}
struct Spy(Rc<RefCell<Vec<OwnedVideoFrame>>>);
impl SceneDetector for Spy {
    fn name(&self) -> &'static str {
        "spy"
    }
    fn metric_keys(&self) -> &'static [&'static str] {
        &[]
    }
    fn process_frame(
        &mut self,
        frame: &VideoFrame<'_>,
        _: Option<&mut dyn MetricsSink>,
    ) -> Result<Vec<Cut>> {
        self.0.borrow_mut().push(OwnedVideoFrame {
            position: frame.position,
            width: frame.width,
            height: frame.height,
            pixel_format: frame.pixel_format,
            data: frame.data.to_vec(),
            stride: frame.stride,
        });
        Ok(vec![])
    }
}
#[test]
fn crop_and_downscale_apply_once_on_owned_and_borrowed_paths() {
    let frames = Rc::new(RefCell::new(Vec::new()));
    let mut pipeline = ScenePipeline::builder()
        .detector(Spy(frames.clone()))
        .crop(Some(CropRegion::new(2, 0, 8, 4).unwrap()))
        .auto_downscale_min_width(3)
        .build()
        .unwrap();
    let mut input = frame(0, 0);
    input.pixel_format = PixelFormat::Bgr24;
    input.stride = 27;
    input.data = (0..108).map(|x| x as u8).collect();
    pipeline.process_frame_ref(&input.as_frame()).unwrap();
    pipeline.process_frame(input.clone()).unwrap();
    let actual = frames.borrow();
    assert_eq!(actual[0], actual[1]);
    assert_eq!(
        (actual[0].width, actual[0].height, actual[0].stride),
        (3, 2, 9)
    );
    let expected = [6_usize, 12, 18, 60, 66, 72]
        .into_iter()
        .flat_map(|offset| input.data[offset..offset + 3].iter().copied())
        .collect::<Vec<_>>();
    assert_eq!(actual[0].data, expected);
    assert_eq!(actual[0].pixel_format, PixelFormat::Bgr24);
}
#[test]
fn crop_ignores_changes_outside_the_requested_region() {
    let mut pipeline = ScenePipeline::builder()
        .detector(ContentDetector::new(1.0, 1))
        .crop(Some(CropRegion::new(0, 0, 2, 4).unwrap()))
        .start_in_scene(true)
        .build()
        .unwrap();
    pipeline.process_frame(frame(0, 0)).unwrap();
    let mut changed = frame(1, 255);
    for row in changed.data.chunks_mut(24) {
        row[..6].fill(0);
    }
    pipeline.process_frame(changed).unwrap();
    assert!(pipeline.finish_detection().unwrap().cuts.is_empty());
}
#[test]
fn invalid_frame_options_fail_before_the_detector_runs() {
    for crop in [
        CropRegion {
            x0: 0,
            y0: 0,
            x1: 9,
            y1: 2,
        },
        CropRegion {
            x0: 4,
            y0: 0,
            x1: 2,
            y1: 2,
        },
    ] {
        let frames = Rc::new(RefCell::new(Vec::new()));
        let mut pipeline = ScenePipeline::builder()
            .detector(Spy(frames.clone()))
            .crop(Some(crop))
            .build()
            .unwrap();
        assert!(pipeline.process_frame(frame(0, 0)).is_err());
        assert!(frames.borrow().is_empty());
    }
    let mut invalid = ScenePipeline::builder()
        .detector(ContentDetector::default())
        .auto_downscale_min_width(0)
        .build()
        .unwrap();
    assert!(invalid.process_frame(frame(0, 0)).is_err());
    let mut input = frame(0, 0);
    input.stride = 1;
    assert!(pipeline().process_frame(input).is_err());
}
