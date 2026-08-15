# Agent instructions

<!-- verification-harness:start -->
## Verification harness
Run the installed `moenarch-verification-harness` skill's `audit` command before changing verification surfaces.
Early selection is advisory; `full` remains the handoff gate. See `.harness/README.md`.
<!-- verification-harness:end -->

Read `CONTEXT.md`, `docs/PROVENANCE.md`, `docs/OWNERSHIP.md`, and the checked
inventories before changing package boundaries.

## Boundaries

- Keep exactly the 27 source families classified in the ownership inventory.
- Do not add ComfyUI or spatial packages, spatial reverse dependencies, sibling
  repository paths, moving Git dependencies, or rust-packages test support.
- Use exact registry coordinates for cross-repository Rust dependencies.
- Treat `scenedetect-core` as the canonical owner of shared scene algorithms.
- Keep `@moritzbrantner/visual-app-ui` private and destination-owned.
- Never publish, tag, release, remove monolith source, or merge the bootstrap
  without a separate exact authorization.

## Bootstrap verification policy

Issue #2 explicitly authorizes structural verification only. Do not run or
claim workspace, unit, integration, consumer, WASM, clippy, docs, build,
package, or benchmark suites for this bootstrap. Run only:

```bash
git diff --check
python3 scripts/check_visual_extraction.py --check
cargo metadata --format-version 1 --no-deps  # only when blockers resolve
```

The `.harness/` profile is draft and non-authoritative. Activation and any
later behavioral gate policy require an explicit maintainer decision.
