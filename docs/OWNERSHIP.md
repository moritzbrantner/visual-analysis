# Ownership

The source-derived inventory consists of:

- 10 image families: captioning, classification, core, detection, embeddings,
  IO, OCR, processing, segmentation, synthesis.
- 16 video families: core, data, dataset, detectors, editing, features, FFmpeg,
  ingest, output, recognition, segmentation, split, storage, synthesis,
  tracking, transform.
- 1 shared vision family: vision-core.

Each family retains library, CLI, server, Rust WASM, app, and Bun WASM surfaces:
108 Cargo packages and 54 Bun packages. The focused UI is one additional
destination-authored private package and is not part of the source count.

ComfyUI and spatial packages are excluded. `scenedetect-core` owns canonical
scene algorithms; visual-analysis owns adapters, visual pipeline contracts,
general FFmpeg integration, output extensions, and split planning.

The machine-readable boundary contract is
`docs/ownership/visual-analysis-boundary.json`. `scripts/check-visual-boundary.py`
enforces it in the structural CI gate: only the declared adapter crates may
depend on `scenedetect-core`, and no package may depend on other scene-detection
crates or on downstream corpus/product repositories.
