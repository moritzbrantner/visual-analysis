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

## Current blockers

The workspace intentionally uses exact registry coordinates and does not fall
back to sibling paths or Git dependencies. Merge/lock resolution remains
blocked until these packages are registry-visible:

- `moenarch-audio-contracts =0.1.0`
- `scenedetect-core =0.1.0`

No publication or source removal is authorized by this bootstrap.
