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

The workspace keeps exact registry coordinates in committed package manifests. Normal feature work may activate the exact foundation, NLP, and scene-detection source revisions declared in `.coding-tooling.source-deps.json`; the generated Cargo configuration remains ignored and uncommitted. The checked-in lockfile records registry sources and checksums for the former bootstrap prerequisites
`moenarch-audio-contracts =0.1.0` and `scenedetect-core =0.1.0`, alongside the
released foundation and narrow NLP contracts declared in the workspace
manifest.

This records the distribution baseline only; it does not make registry publication a prerequisite for source-mode implementation work and does not replace the separate behavioral verification required for a release. Run `bash scripts/source-deps activate` for the exact source graph and `bash scripts/source-deps deactivate` before registry-only release verification. See [docs/SOURCE_DEVELOPMENT.md](docs/SOURCE_DEVELOPMENT.md).

No publication or source removal is authorized by this bootstrap.
