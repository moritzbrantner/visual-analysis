#!/usr/bin/env bash
# Run after activating the declared Foundation source graph.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .artifacts/audit
packages=(
  -p moenarch-video-analysis-core
  -p moenarch-video-analysis-detectors
  -p moenarch-video-analysis-ffmpeg
  -p moenarch-video-analysis-ingest
  -p moenarch-video-analysis-recognition
)
features=moenarch-video-analysis-ingest/ocr,moenarch-video-analysis-ffmpeg/ffmpeg-tests
cargo test --locked "${packages[@]}" --features "$features" --lib --tests 2>&1 | tee .artifacts/audit/native.log
cargo clippy --locked "${packages[@]}" --features "$features" --all-targets -- -D warnings 2>&1 | tee .artifacts/audit/clippy.log
# One untimed execution of each benchmark also catches stale/unbuildable harnesses.
cargo test --locked -p moenarch-video-analysis-detectors --bench scene_detectors -- --test 2>&1 | tee .artifacts/audit/scene-bench-smoke.log
cargo test --locked -p moenarch-video-analysis-recognition --features ocr --bench text_regressions -- --test 2>&1 | tee .artifacts/audit/text-bench-smoke.log
