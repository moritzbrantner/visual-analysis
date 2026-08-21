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

The workspace intentionally uses exact registry coordinates and does not fall
back to sibling paths or Git dependencies. The checked-in lockfile records
registry sources and checksums for the former bootstrap prerequisites
`moenarch-audio-contracts =0.1.0` and `scenedetect-core =0.1.0`, alongside the
released foundation and narrow NLP contracts declared in the workspace
manifest.

This records dependency-resolution provenance only; it does not replace the
separate behavioral verification required for a release. No publication or
source removal is authorized by this bootstrap.
