//! Structural performance ratchet: real production ingestion, not a timing gate.
use num_rational::Rational64;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use video_analysis_core::{FramePosition, OwnedVideoFrame, PixelFormat, Result};
use video_analysis_ingest::surface::detect_content_scenes;
use video_analysis_ingest::{MediaSourceInfo, VideoFrameSource, VideoStreamInfo};

const WIDTH: u32 = 257;
const HEIGHT: u32 = 127;
const FRAME_BYTES: usize = WIDTH as usize * HEIGHT as usize * 3;
#[derive(Clone, Copy, Default, Debug)]
struct Counts {
    active: bool,
    current: usize,
    peak: usize,
    allocations: usize,
}
thread_local! { static COUNTS: Cell<Counts> = const { Cell::new(Counts { active: false, current: 0, peak: 0, allocations: 0 }) }; }
struct CountingAllocator;
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;
fn allocated(size: usize) {
    COUNTS.with(|cell| {
        let mut c = cell.get();
        if c.active && size >= FRAME_BYTES {
            c.current += size;
            c.peak = c.peak.max(c.current);
            c.allocations += 1;
            cell.set(c);
        }
    });
}
fn released(size: usize) {
    COUNTS.with(|cell| {
        let mut c = cell.get();
        if c.active && size >= FRAME_BYTES {
            c.current = c.current.saturating_sub(size);
            cell.set(c);
        }
    });
}
// Test-only allocator delegates every operation to System unchanged. The
// thread-local counters exclude the Rust test harness and other test threads.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            allocated(layout.size());
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            allocated(layout.size());
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        released(layout.size());
        unsafe {
            System.dealloc(ptr, layout);
        }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let next = unsafe { System.realloc(ptr, layout, size) };
        if !next.is_null() {
            released(layout.size());
            allocated(size);
        }
        next
    }
}
struct Source {
    info: MediaSourceInfo,
    next: u64,
    count: u64,
    reads: u64,
    offset: u64,
    padded_bgr: bool,
}
impl Source {
    fn new(count: u64) -> Self {
        Self {
            info: MediaSourceInfo::recorded("generated-audit").with_video(VideoStreamInfo {
                width: WIDTH,
                height: HEIGHT,
                pixel_format: PixelFormat::Rgb24,
                frame_rate: Some(Rational64::new(30, 1)),
            }),
            next: 0,
            count,
            reads: 0,
            offset: 0,
            padded_bgr: false,
        }
    }
}
impl VideoFrameSource for Source {
    fn source_info(&self) -> &MediaSourceInfo {
        &self.info
    }
    fn next_video_frame(&mut self) -> Result<Option<OwnedVideoFrame>> {
        self.reads += 1;
        if self.next == self.count {
            return Ok(None);
        }
        let index = self.next;
        self.next += 1;
        let color = if index < self.count / 2 {
            [255, 0, 0]
        } else {
            [0, 255, 0]
        };
        let stride = WIDTH as usize * 3 + if self.padded_bgr { 7 } else { 0 };
        let mut data = vec![0; stride * HEIGHT as usize];
        for row in data.chunks_exact_mut(stride) {
            for pixel in row[..WIDTH as usize * 3].chunks_exact_mut(3) {
                pixel.copy_from_slice(&color);
                if self.padded_bgr {
                    pixel.swap(0, 2);
                }
            }
        }
        Ok(Some(OwnedVideoFrame {
            position: FramePosition::from_frame_index(index + self.offset, Rational64::new(30, 1)),
            width: WIDTH,
            height: HEIGHT,
            pixel_format: if self.padded_bgr {
                PixelFormat::Bgr24
            } else {
                PixelFormat::Rgb24
            },
            data,
            stride,
        }))
    }
}
#[test]
fn public_ingest_retains_at_most_two_raw_frames_and_allocates_one_per_input() {
    for frame_count in [16_u64, 64, 128] {
        let mut source = Source::new(frame_count);
        COUNTS.with(|counts| {
            counts.set(Counts {
                active: true,
                ..Counts::default()
            })
        });
        let result = detect_content_scenes(&mut source, 20.0, 1);
        let counts = COUNTS.with(|cell| {
            let c = cell.get();
            cell.set(Counts::default());
            c
        });
        let result = result.unwrap();
        assert_eq!(source.reads, frame_count + 1);
        assert_eq!(result.frames_processed, frame_count);
        assert_eq!(
            result
                .cuts
                .iter()
                .map(|cut| cut.position.frame_index)
                .collect::<Vec<_>>(),
            [frame_count / 2]
        );
        assert!(
            counts.peak <= 2 * FRAME_BYTES,
            "unbounded raw frame retention: {counts:?}"
        );
        assert_eq!(
            counts.allocations, frame_count as usize,
            "raw frame copies or growing-prefix work: {counts:?}"
        );
        assert_eq!(counts.current, 0, "raw pixels escaped analysis");
    }
}
#[test]
fn padded_bgr_and_nonzero_source_indices_preserve_cut_positions_and_scores() {
    let rgb = detect_content_scenes(&mut Source::new(4), 20.0, 1).unwrap();
    let mut source = Source::new(4);
    source.padded_bgr = true;
    source.offset = 100;
    let bgr = detect_content_scenes(&mut source, 20.0, 1).unwrap();
    assert_eq!(bgr.scenes[0].start.frame_index, 100);
    assert_eq!(bgr.cuts[0].position.frame_index, 102);
    assert_eq!(
        bgr.cuts[0].position.timestamp,
        FramePosition::from_frame_index(102, Rational64::new(30, 1)).timestamp
    );
    assert_eq!(
        rgb.metrics.get(2, "content_val"),
        bgr.metrics.get(102, "content_val")
    );
}
#[test]
fn source_without_declared_frame_rate_uses_frame_timebase() {
    let mut source = Source::new(4);
    source.info.video.as_mut().unwrap().frame_rate = None;
    source.offset = 100;

    let result = detect_content_scenes(&mut source, 20.0, 1).unwrap();

    assert_eq!(source.reads, 5);
    assert_eq!(result.frames_processed, 4);
    assert_eq!(
        result
            .cuts
            .iter()
            .map(|cut| cut.position.frame_index)
            .collect::<Vec<_>>(),
        vec![102]
    );
    assert_eq!(
        result.cuts[0].position.timestamp,
        FramePosition::from_frame_index(102, Rational64::new(30, 1)).timestamp
    );
}

#[test]
fn empty_and_invalid_inputs_do_not_invent_scenes_or_decode() {
    assert!(detect_content_scenes(&mut Source::new(0), 20.0, 1)
        .unwrap()
        .scenes
        .is_empty());
    for (threshold, minimum) in [(f32::NAN, 1), (-1.0, 1), (20.0, 0)] {
        let mut source = Source::new(4);
        assert!(detect_content_scenes(&mut source, threshold, minimum).is_err());
        assert_eq!(source.reads, 0);
    }
}
