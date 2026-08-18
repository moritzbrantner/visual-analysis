# Repository context

This repository is the visual capability layer of the Moenarch Rust ecosystem.
It owns image analysis, shared visual contracts, and non-spatial video
processing. It may depend on exact published foundation and NLP contracts.

It must not own ComfyUI, posture, reconstruction, radiance-field, MVS, SfM, or
other spatial packages. It must not depend on spatial-analysis or on the
rust-packages test-support crate. `scenedetect-core` is the canonical owner of
the five shared scene-detector algorithms; visual packages retain compatibility
adapters and broader visual workflows.

The extraction is additive. `moritzbrantner/rust-packages` remains untouched and
continues to contain its source until a separate source-removal issue is
authorized.
