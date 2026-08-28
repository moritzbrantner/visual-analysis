# Agent instructions

<!-- verification-harness:start -->
## Verification harness
Run the installed `moenarch-verification-harness` skill's `audit` command before changing verification surfaces.
Early selection is advisory; `full` remains the handoff gate. See `.harness/README.md`.
<!-- verification-harness:end -->

Read `CONTEXT.md`, `docs/PROVENANCE.md`, `docs/OWNERSHIP.md`, `docs/SOURCE_DEVELOPMENT.md`, and the checked inventories before changing package boundaries or cross-repository dependencies.

## Boundaries

- Keep exactly the 27 source families classified in the ownership inventory.
- Do not add ComfyUI or spatial packages, spatial reverse dependencies, sibling repository paths, moving Git dependencies, or rust-packages test support to committed package manifests.
- Keep exact registry coordinates in package manifests. Normal source development temporarily replaces only the declared `moenarch-foundation` packages through the managed, ignored Cargo configuration.
- Do not add NLP or scene-detection repositories to the normal source workspace. Existing `text-core` usage is a compatibility/data-contract surface; `scenedetect-core` crosses through explicit detector adapters. New cross-domain behavior belongs in an adapter or consuming application.
- The outer coding workspace owns the declared foundation worktree. It must exist at its exact pinned revision before source activation; do not add private-repository tokens or authenticated Git fallback to hosted CI.
- Ordinary feature work is source-first. Do not publish crates, bump package versions, create tags, or start a release train merely to unblock development.
- Keep package versions stable during source work when compatibility permits; a dedicated release task owns version bumps, publication, and registry-only verification.
- If a consumer task expands beyond the consumer plus two upstream repositories, treat that as an architecture boundary problem unless broader migration scope was explicitly assigned.
- Treat `scenedetect-core` as the canonical owner of shared scene algorithms; keep `video-analysis-detectors::canonical` as the integration seam rather than copying scene algorithms into visual core.
- Keep `@moritzbrantner/visual-app-ui` private and destination-owned.
- Never publish, tag, release, remove monolith source, or merge the bootstrap without a separate exact authorization.
- Registry-only dependency verification is release evidence; it is not required before exact source-mode implementation evidence is useful.

## Bootstrap verification policy

Issue #2 explicitly authorizes structural verification only. Do not run or
claim workspace, unit, integration, consumer, WASM, clippy, docs, build,
package, or benchmark suites for this bootstrap. Run only:

```bash
git diff --check c5d6b02a708837cf01ee6e033faa41b0a667b38b..HEAD
python3 scripts/check_visual_extraction.py --check  # authoritative structural and registry-coordinate audit
cargo metadata --format-version 1 --no-deps  # registry by default; exact managed source graph when activated
```

For focused extraction iteration, `bun run structural:prepare` checks the
static inventory without querying the registry. It is not an authoritative
gate and cannot make the bootstrap ready while either exact registry
prerequisite is absent.

The `.harness/` profile is draft and non-authoritative. Activation and any
later behavioral gate policy require an explicit maintainer decision.
