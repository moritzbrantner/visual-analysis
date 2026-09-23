use crate::{CropRegion, DetectError, OwnedVideoFrame, Result, VideoFrame};

/// Visual frame preparation, not a scene-detection algorithm. The unchanged
/// path borrows the caller's pixels; transformed frames use a tight packed buffer.
pub(super) fn prepare(
    frame: &VideoFrame<'_>,
    crop: Option<CropRegion>,
    minimum_width: Option<u32>,
) -> Result<Option<OwnedVideoFrame>> {
    VideoFrame::packed(
        frame.position,
        frame.width,
        frame.height,
        frame.pixel_format,
        frame.data,
        frame.stride,
    )?;
    let crop = crop.unwrap_or(CropRegion {
        x0: 0,
        y0: 0,
        x1: frame.width,
        y1: frame.height,
    });
    if crop.x0 >= crop.x1 || crop.y0 >= crop.y1 || crop.x1 > frame.width || crop.y1 > frame.height {
        return Err(DetectError::InvalidArgument(
            "crop must be a non-empty region inside the frame".into(),
        ));
    }
    if minimum_width == Some(0) {
        return Err(DetectError::InvalidArgument(
            "automatic downscale minimum width must be positive".into(),
        ));
    }
    let source_width = crop.x1 - crop.x0;
    let source_height = crop.y1 - crop.y0;
    let factor = minimum_width.map_or(1, |minimum| (source_width / minimum).max(1));
    let width = source_width / factor;
    let height = (source_height / factor).max(1);
    if crop.x0 == 0 && crop.y0 == 0 && width == frame.width && height == frame.height {
        return Ok(None);
    }
    let stride = width as usize * 3;
    let mut data = vec![0; stride * height as usize];
    // Nearest-neighbor sampling is deterministic and preserves channel order.
    for y in 0..height {
        let source_y =
            crop.y0 + (u64::from(y) * u64::from(source_height) / u64::from(height)) as u32;
        for x in 0..width {
            let source_x =
                crop.x0 + (u64::from(x) * u64::from(source_width) / u64::from(width)) as u32;
            let source_offset = source_y as usize * frame.stride + source_x as usize * 3;
            let target_offset = y as usize * stride + x as usize * 3;
            data[target_offset..target_offset + 3]
                .copy_from_slice(&frame.data[source_offset..source_offset + 3]);
        }
    }
    Ok(Some(OwnedVideoFrame {
        position: frame.position,
        width,
        height,
        pixel_format: frame.pixel_format,
        data,
        stride,
    }))
}
