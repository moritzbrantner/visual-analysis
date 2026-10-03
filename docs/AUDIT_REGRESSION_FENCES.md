# Visual audit regression fences

Audit baseline: `281d4c553beed5c151e5a7e66a41309631ae6fff`.
The September 23, 2026 maintainer request authorizes repairs, behavioral tests,
benchmarks, and integration. No package version or release is changed.

## Blocking evidence

| Finding | Production repair | Regression fence |
| --- | --- | --- |
| FFmpeg resized frame boundaries | One explicit size drives both `scale=w:h` and raw framing | `audit_decoder`: three distinguishable real FFV1 frames, landscape/portrait/tiny aspect ratios, complete EOF |
| Failed decoder looks like EOF / stderr deadlock | Shared video/audio process owner drains bounded 64 KiB diagnostic tail and checks exit status | Nonzero exit before output, complete frame followed by failure, repeat EOF, 1 MiB stderr saturation, bounded diagnostic retention |
| Public scene ingest reanalyzes every prefix | Existing canonical one-pass source adapter, moved/repacked RGB/BGR buffer, original frame-position projection | Actual production ingest allocation counter at 16/64/128 frames: at most two raw buffers alive, exactly one input pixel allocation per frame, no retained pixels; output and offset assertions |
| Reset retains detector state | Detector reset lifecycle hook and ContentDetector state cleanup | Same clip twice; A then B versus fresh B, including cut suppression/history |
| Stale browser inference changes another image | Selection/operation generation identities, post-await guards, session lease/disposal | Seven real Chromium tests: reordered completion, stale failures, A-B-A, stale embedding, in-flight mask, interrupted box sequence, Choose another |
| Crop/downscale settings ignored | Shared owned/borrowed frame preparation and full validation | Spy checks exact padded BGR pixels and dimensions, one application on both paths, outside-crop changes, invalid-input short-circuit |
| Co-frame text counted as temporal persistence | Independent neighboring occurrences; persistence requires different temporal anchors | Same text in two nearby boxes, duplicate observation, absent anchors, reversed input, disappearance gap |
| Long-scene OCR samples too far apart | Bounded periodic samples compatible with tracking gaps; duplicate decoded targets ignored | 60-second, three-line slide through real OCR composition; one stable track per line, bounded backend calls; oversized plans rejected |

The source-owned tracking heuristics remain conservative, not calibrated semantic
probabilities. Sampling estimates a uniform interval from scene endpoints; VFR
sources still use actual timestamps for association and may break tracks across
large gaps. The plan is capped at 10,000 unique targets and rejects oversized
requests before generating an unbounded sequence.

The legacy incremental ContentDetector remains a compatibility adapter. Its
prefix reanalysis is intentionally retained as a benchmark reference; consumers
processing whole sources should use `detect_content_scenes` or the canonical
source adapter. Metadata/statistics may grow with frame count; the raw-frame
memory bound does not claim constant total report memory.

## Commands and CI

Prepare the exact declared Foundation source graph with `scripts/source-deps`.
No NLP or scene source checkout is introduced. Then run:

```sh
bash scripts/check-audit-regressions.sh
node --test tests/pages/*.test.mjs
python -m pip install -r tests/browser/requirements.txt
python -m playwright install chromium
python tests/browser/vision_races.py
```

`Audit regression fences` runs native correctness, strict Clippy, and untimed
benchmark execution on affected PRs. These checks enable both ingest OCR and real
FFmpeg fixtures, avoiding the previous uncompiled/missing `ocr` feature gap.
FFmpeg/ffprobe are required, not silently skipped by the audit integration tests.
Browser fixtures exercise the shipped HTML/CSS/controller in Chromium with only
the model adapter deferred; they prohibit network access and do not claim live
SAM/WebGPU accuracy. Screenshots and logs are uploaded after execution, including
on failure. Existing structural, Pages, and native ONNX acceptance remain separate.

Measured release benchmarks run on main or an explicit workflow dispatch:

```sh
cargo bench --locked -p moenarch-video-analysis-detectors --bench scene_detectors -- --noplot
cargo bench --locked -p moenarch-video-analysis-recognition --features ocr --bench text_regressions -- --noplot
python3 scripts/report-audit-benchmarks.py
```

The scene comparison executes legacy growing-prefix and one-pass production
paths on identical generated input in one process. It first compares cuts,
source timestamps, frame counts, and existing `content_val` scores. The new
canonical statistics also expose component metrics; those additional fields are
not falsely required to equal the narrower legacy statistics.

Text benchmarks cover neighboring tracks and bounded sampling plans at growing
sizes, asserting their semantic contracts before timing. JSON records include
confidence intervals and source identity. Timing comparisons are diagnostic
artifacts, never machine-sensitive pass/fail thresholds. Source-read counts alone
are insufficient; the allocation ratchet detects the original frame retention.

## Before/after validation

The new decoder tests fail against the baseline's mismatched framing and silent
EOF. On that same original source, all four new pipeline tests fail; the scene
allocation/offset tests fail; four of five new text-track tests fail. Correctness
is not inferred from a faster benchmark. The repairs are tested through actual
Rust entrypoints, and browser race tests use controlled completion ordering.

Previously unbuildable detector examples/tests referred to removed monolith-only
interfaces. They now call the registered canonical configurations. The retained
`pyscenedetect_detector_parity.rs` filename is provenance, not a claim of external
PySceneDetect or removed streaming-filter numerical parity. External media/model
fixtures remain explicit/opt-in; no synthetic stub replaces native ONNX acceptance.
