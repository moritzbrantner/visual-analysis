# visual-analysis

Rust-first image, vision, and non-spatial video analysis packages extracted
from `moritzbrantner/rust-packages`.

The bootstrap freezes source commit
`b8b29cf8db0b86ed1b133a18155adf24992f9483` and owns exactly 27 visual
families: 108 Cargo packages and 54 source-derived Bun packages. A private,
destination-authored `@moritzbrantner/visual-app-ui` adapter serves the 27 apps.

This extraction is structural. Behavioral, build, test, lint, documentation,
WASM, package, consumer, and benchmark evidence was deliberately not run or
claimed. See [docs/PROVENANCE.md](docs/PROVENANCE.md) and
[docs/repository-split/package-ownership.json](docs/repository-split/package-ownership.json).

## Migration status

The workspace keeps exact registry coordinates in committed package manifests. Normal feature work may activate only the exact `moenarch-foundation` source revision declared in `.coding-tooling.source-deps.json`; the generated Cargo configuration remains ignored and uncommitted. `moenarch-text-core` and `scenedetect-core` remain registry-level compatibility/adapter dependencies rather than sibling source-workspace requirements.

This keeps the normal source graph directional: visual capabilities may build on foundation, while NLP enrichment and scene-detection interoperability cross explicit compatibility or adapter seams. `video-analysis-detectors::canonical` remains the concrete adapter around `scenedetect-core`; scene algorithms stay owned by `scenedetect-rs`.

This records the distribution baseline only; it does not make registry publication a prerequisite for source-mode implementation work and does not replace the separate behavioral verification required for a release. Run `bash scripts/source-deps activate` for the exact foundation source graph and `bash scripts/source-deps deactivate` before registry-only release verification. See [docs/SOURCE_DEVELOPMENT.md](docs/SOURCE_DEVELOPMENT.md).

No publication or source removal is authorized by this bootstrap.
