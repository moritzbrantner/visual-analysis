# video-analysis-detectors

Scene detection algorithms and detector adapters for `moritzbrantner-video-analysis`.

The crate tracks PySceneDetect behavior parity for detector outputs and common
workflows while keeping Rust-first APIs and workspace crate boundaries.

## Feature flags

- No optional feature flags today.

## Runtime surface

- `video.detectors.registry` summarizes detector families and score algorithms.
- `video.detectors.flashFilter` runs the deterministic flash filter over frame
  threshold decisions.
- `video.detectors.compositePlan` validates weighted composite detector
  configuration without processing media.

## Detector families

- `ContentDetector`: HSV/luma/edge frame-difference cuts, matching the
  PySceneDetect content detector workflow.
- `AdaptiveDetector`: content-score ratio over a delayed window.
- `ThresholdDetector`: fade out/in detection with floor and ceiling methods.
- `HistogramDetector`: luma histogram correlation cuts.
- `HashDetector`: perceptual hash distance cuts.
- `WeightedCompositeDetector`: Rust-native composite scoring over detector
  algorithms.

Known intentional differences:

- Public APIs are idiomatic Rust and do not clone PySceneDetect class names or
  CLI flags exactly.
- Default tests use synthetic fixtures. Large PySceneDetect resource videos,
  BBC, and AutoShot are opt-in local corpora.
- Dataset evaluation reports include both predictions and loaded ground-truth
  cuts so downstream scripts can recompute metrics.

## Example

```rust,ignore
use video_analysis_detectors::ContentDetector;

let detector = ContentDetector::new(27.0, 15);
let _ = detector;
```

## Verification

Default detector tests:

```bash
cargo test -p moritzbrantner-video-analysis-detectors
```

Ignored external smoke test:

```bash
cargo test -p moritzbrantner-video-analysis-detectors --test external_me_at_the_zoo -- --ignored
```

Criterion benchmarks:

```bash
cargo bench -p moritzbrantner-video-analysis-detectors
bun run video-native:bench
```

Opt-in dataset evaluation:

```bash
scripts/setup_video_scene_benchmarks.sh
cargo run --release -p moritzbrantner-video-analysis-detectors --example scene_dataset_eval -- --dataset bbc --root .test-corpora/video-scene/BBC --detector content --video-id bbc_10 --resize-width 320 --progress --resume --max-runtime-seconds 3300 --output target/video-scene/content-bbc-smoke.json
python3 scripts/check_video_scene_eval.py target/video-scene/content-bbc-smoke.json --allow-partial
```

For the current three-video BBC baseline, run:

```bash
cargo run --release -p moritzbrantner-video-analysis-detectors --example scene_dataset_eval -- --dataset bbc --root .test-corpora/video-scene/BBC --detector content --video-id bbc_03 --video-id bbc_06 --video-id bbc_10 --resize-width 320 --progress --resume --max-runtime-seconds 3300 --output target/video-scene/content-bbc-subset-broad.json
python3 scripts/check_video_scene_eval.py target/video-scene/content-bbc-subset-broad.json --allow-partial --tolerance-frames 0
python3 scripts/summarize_video_scene_eval.py target/video-scene/content-bbc-subset-broad.json --tolerance-frames 0 --output target/video-scene/content-bbc-subset-broad-summary.json
```

The baseline report uses PySceneDetect-compatible content defaults:
`contentThreshold=27.0`, `minSceneLen=15`, `filterMode=merge`, and
`postFilterWindow=0`. The evaluator also accepts explicit content/adaptive
tuning flags for local experiments:

```bash
cargo run --release -p moritzbrantner-video-analysis-detectors --example scene_dataset_eval -- --dataset bbc --root .test-corpora/video-scene/BBC --detector content --video-id bbc_03 --video-id bbc_06 --video-id bbc_10 --resize-width 320 --content-threshold 33 --min-scene-len 20 --filter-mode merge --post-filter-window 8 --progress --resume --max-runtime-seconds 3300 --output target/video-scene/content-bbc-subset-tuned.json
```

Run the reproducible BBC subset sweep with:

```bash
python3 scripts/sweep_video_scene_eval.py \
  --dataset bbc \
  --root .test-corpora/video-scene/BBC \
  --video-id bbc_03 \
  --video-id bbc_06 \
  --video-id bbc_10 \
  --resize-width 320 \
  --max-runtime-seconds 3300 \
  --output target/video-scene/bbc-subset-sweep.json
```

The full BBC corpus is still available as an explicit long-running benchmark:

```bash
cargo run --release -p moritzbrantner-video-analysis-detectors --example scene_dataset_eval -- --dataset bbc --root .test-corpora/video-scene/BBC --detector content --progress --resume --output target/video-scene/content-bbc-full.json
python3 scripts/check_video_scene_eval.py target/video-scene/content-bbc-full.json
```

Benchmark outputs are local artifacts under `target/video-scene/`. They are
kept out of version control unless a specific reviewed output class is added to
the repository policy.

Equal Rust vs PySceneDetect speed comparison:

```bash
cargo run --release -p moritzbrantner-video-analysis-detectors --example scene_dataset_eval -- \
  --dataset bbc \
  --root .test-corpora/video-scene/BBC \
  --detector content \
  --video-id bbc_03 \
  --video-id bbc_06 \
  --video-id bbc_10 \
  --resize-width 320 \
  --progress \
  --resume \
  --max-runtime-seconds 3300 \
  --output target/video-scene/rust-content-bbc-subset-broad.json

python3 scripts/pyscenedetect_scene_dataset_eval.py \
  --dataset bbc \
  --root .test-corpora/video-scene/BBC \
  --detector content \
  --video-id bbc_03 \
  --video-id bbc_06 \
  --video-id bbc_10 \
  --resize-width 320 \
  --output target/video-scene/pyscenedetect-content-bbc-subset-broad.json

python3 scripts/compare_scene_detector_speed.py \
  --rust-report target/video-scene/rust-content-bbc-subset-broad.json \
  --pyscenedetect-report target/video-scene/pyscenedetect-content-bbc-subset-broad.json \
  --output target/video-scene/content-bbc-speed-compare.json
```

`elapsedMs` is end-to-end video evaluator time. `decodeResizeElapsedMs`
measures frame pulling from the decode/resize source, and `detectorElapsedMs`
measures detector processing plus finalization.

Detector-only real-frame timing:

```bash
cargo run --release -p moritzbrantner-video-analysis-detectors --example scene_detector_real_frame_benchmark -- \
  --input .test-corpora/video-scene/BBC/videos/bbc_03.mp4 \
  --resize-width 320 \
  --detector content \
  --iterations 5 \
  --output target/video-scene/rust-real-frame-content-bbc_03.json
```

## Related crates

- `video-analysis-core`
- `video-analysis-cli`
- `video-analysis-split`
