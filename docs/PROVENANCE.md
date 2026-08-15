# Provenance

## Frozen source

- Source repository: `moritzbrantner/rust-packages`
- Exact extraction commit: `b8b29cf8db0b86ed1b133a18155adf24992f9483`
- Destination bootstrap issue: `moritzbrantner/visual-analysis#2`
- Parent PRD: `moritzbrantner/visual-analysis#1`

The extraction enumerates each family and surface explicitly. It does not copy
the broad `crates/image`, `crates/video`, `crates/bindings`, or `packages`
trees. The source checkout was never modified or removed.

`docs/repository-split/source-byte-identity.json` records SHA-256 for all 911
source files selected at the frozen commit. Unadapted files are verified byte
for byte. Adapted files are identified separately and comprise dependency
manifests, scene adapter boundaries, and app references to the focused UI.

## Intentional adaptations

- Cross-repository Cargo dependencies use exact registry coordinates.
- Dataset and recognition manifests no longer declare four spatial reverse
  edges; detector dev-dependencies no longer use monolith test support.
- Canonical shared scene-detector implementations are not copied as active
  visual source. Visual compatibility code delegates to or names the exact
  `scenedetect-core =0.1.0` seam.
- All 27 source-derived apps use the private destination-authored
  `@moritzbrantner/visual-app-ui` workspace adapter.

No behavioral equivalence is asserted by this structural bootstrap.
