#!/usr/bin/env python3
"""Generate the tiny deterministic Pages workbench fixtures.

The PNG is produced byte-for-byte by the Python standard library. The video
frames are produced deterministically here (raw Y4M) and encoded losslessly as
VP9/WebM by ffmpeg with bit-exact flags; WebM container bytes can still differ
between ffmpeg releases, so the committed file is the fixture and
``manifest.json`` records its SHA-256. ``--check`` regenerates the PNG and the
decoded video frame values and verifies them against the committed files
without rewriting anything.

Pixel design (see manifest.json for the derived expectations):

* workbench-quadrants.png, 16x16 RGB, no colour-management chunks:
  top half (rows 0-7) red (255,0,0); bottom-left quarter blue (0,0,255);
  bottom-right quarter white (255,255,255).
* workbench-steps.webm, 64x48, 10 fps, 24 frames (2.4 s), uniform grey frames:
  frames 0-7 grey 36, frames 8-15 grey 124, frames 16-23 grey 220. These greys
  are the centres of 32-bin luma histogram bins 4, 15 and 27, so codec rounding
  of +-3 cannot move a sample into a different bin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
PNG_NAME = "workbench-quadrants.png"
VIDEO_NAME = "workbench-steps.webm"
PNG_SIZE = 16
VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS = 64, 48, 10
VIDEO_SEGMENTS = ((8, 36), (8, 124), (8, 220))


def png_pixel(x: int, y: int) -> tuple[int, int, int]:
    half = PNG_SIZE // 2
    if y < half:
        return (255, 0, 0)
    return (0, 0, 255) if x < half else (255, 255, 255)


def png_bytes() -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    raw = bytearray()
    for y in range(PNG_SIZE):
        raw.append(0)  # filter type: none
        for x in range(PNG_SIZE):
            raw.extend(png_pixel(x, y))
    header = struct.pack(">IIBBBBB", PNG_SIZE, PNG_SIZE, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def video_greys() -> list[int]:
    return [grey for count, grey in VIDEO_SEGMENTS for _ in range(count)]


def limited_range_luma(grey: int) -> int:
    return round(16 + grey * 219 / 255)


def y4m_bytes() -> bytes:
    out = bytearray(
        f"YUV4MPEG2 W{VIDEO_WIDTH} H{VIDEO_HEIGHT} F{VIDEO_FPS}:1 Ip A1:1 C420jpeg\n".encode()
    )
    chroma = bytes([128]) * ((VIDEO_WIDTH // 2) * (VIDEO_HEIGHT // 2))
    for grey in video_greys():
        out += b"FRAME\n"
        out += bytes([limited_range_luma(grey)]) * (VIDEO_WIDTH * VIDEO_HEIGHT)
        out += chroma + chroma
    return bytes(out)


def encode_video(target: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg with libvpx-vp9 is required to (re)generate the video fixture")
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "frames.y4m"
        source.write_bytes(y4m_bytes())
        subprocess.run(
            [
                ffmpeg, "-loglevel", "error", "-y", "-i", str(source),
                "-c:v", "libvpx-vp9", "-lossless", "1", "-row-mt", "0", "-threads", "1",
                "-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt470bg",
                "-color_primaries", "bt470bg", "-color_trc", "smpte170m",
                "-g", "8", "-an", "-fflags", "+bitexact", "-flags", "+bitexact",
                "-map_metadata", "-1", str(target),
            ],
            check=True,
        )


def decoded_video_luma(path: Path) -> list[int]:
    """Decode the committed WebM and return each frame's (uniform) limited-range Y value."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg is required to verify the video fixture")
    raw = subprocess.run(
        [ffmpeg, "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
        check=True,
        capture_output=True,
    ).stdout
    luma = VIDEO_WIDTH * VIDEO_HEIGHT
    frame = luma * 3 // 2
    if len(raw) % frame:
        raise SystemExit(f"{path.name}: decoded size {len(raw)} is not a whole number of frames")
    frames = [raw[index:index + luma] for index in range(0, len(raw), frame)]
    for index, data in enumerate(frames):
        if len(set(data)) != 1:
            raise SystemExit(f"{path.name}: frame {index} is not uniform")
    return [data[0] for data in frames]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest() -> dict:
    return {
        "schemaVersion": 1,
        "generator": "tests/fixtures/pages/generate.py",
        "image": {
            "file": PNG_NAME,
            "sha256": sha256(HERE / PNG_NAME),
            "mimeType": "image/png",
            "width": PNG_SIZE,
            "height": PNG_SIZE,
            "regions": [
                {"rgb": [255, 0, 0], "pixels": 128},
                {"rgb": [0, 0, 255], "pixels": 64},
                {"rgb": [255, 255, 255], "pixels": 64},
            ],
        },
        "video": {
            "file": VIDEO_NAME,
            "sha256": sha256(HERE / VIDEO_NAME),
            "mimeType": "video/webm",
            "codec": "vp9 lossless, yuv420p limited range",
            "width": VIDEO_WIDTH,
            "height": VIDEO_HEIGHT,
            "fps": VIDEO_FPS,
            "durationSeconds": sum(count for count, _ in VIDEO_SEGMENTS) / VIDEO_FPS,
            "segments": [
                {"frames": count, "grey": grey} for count, grey in VIDEO_SEGMENTS
            ],
        },
    }


def check() -> int:
    errors = []
    if (HERE / PNG_NAME).read_bytes() != png_bytes():
        errors.append(f"{PNG_NAME} differs from the generator output")
    expected_luma = [limited_range_luma(grey) for grey in video_greys()]
    if not shutil.which("ffmpeg"):
        print(f"ffmpeg unavailable: verifying {VIDEO_NAME} by its manifest SHA-256 only", file=sys.stderr)
    elif decoded_video_luma(HERE / VIDEO_NAME) != expected_luma:
        errors.append(f"{VIDEO_NAME} does not decode to the generated frame values")
    committed = json.loads((HERE / "manifest.json").read_text())
    if committed != manifest():
        errors.append("manifest.json is stale; rerun tests/fixtures/pages/generate.py")
    for error in errors:
        print(error, file=sys.stderr)
    if not errors:
        print("Pages fixtures match their generator.")
    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify committed fixtures instead of writing")
    args = parser.parse_args()
    if args.check:
        return check()
    (HERE / PNG_NAME).write_bytes(png_bytes())
    encode_video(HERE / VIDEO_NAME)
    expected_luma = [limited_range_luma(grey) for grey in video_greys()]
    if decoded_video_luma(HERE / VIDEO_NAME) != expected_luma:
        raise SystemExit("encoded video is not lossless; refusing to write the manifest")
    (HERE / "manifest.json").write_text(json.dumps(manifest(), indent=2) + "\n")
    print(f"wrote {PNG_NAME}, {VIDEO_NAME} and manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
