# Repository context

This repository is the visual capability layer of the Moenarch Rust ecosystem.
It owns image analysis, shared visual contracts, and non-spatial video
processing. It may depend on foundation and NLP contracts. During normal development, the committed source-dependency declaration may replace registry packages with exact reviewed source revisions; distributed builds remain registry-only.

It must not own ComfyUI, reconstruction, radiance-field, MVS, SfM, or other
spatial packages. Posture estimation (`video-analysis-posture`,
`video-analysis-posture-io`) moved here from rust-packages by owner decision in
rust-packages#179: it is pose estimation over video frames, not spatial
reconstruction, and it keeps its own small 3D skeleton geometry. It must not depend on spatial-analysis or on the
rust-packages test-support crate. `scenedetect-core` is the canonical owner of
the five shared scene-detector algorithms; visual packages retain compatibility
adapters and broader visual workflows.

The extraction is additive. `moritzbrantner/rust-packages` remains untouched and
continues to contain its source until a separate source-removal issue is
authorized.
