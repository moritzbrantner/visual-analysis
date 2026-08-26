# Visual source development

`visual-analysis` keeps registry coordinates as its distribution contract, but ordinary cross-repository development does not require publishing foundation, NLP, or scene-detection changes first.

The committed `.coding-tooling.source-deps.json` pins the external package closure used by this workspace to exact revisions of `moenarch-foundation`, `nlp-stack`, and `scenedetect-rs` and enables local-only resolution. Run `bash scripts/source-deps activate` only after every declared sibling checkout exists at its exact pinned Git `HEAD`. Missing local source is an error; source mode never falls back to cloning private repositories or authenticated Git fetches.

The outer coding loop owns those sibling repositories/worktrees and may advance their pins when a task deliberately validates newer source heads. Use `bash scripts/source-deps status` to inspect the mode and `bash scripts/source-deps deactivate` before registry-only release verification.

## Development contract

- Feature work may change visual code and immediate upstream source without starting a crates.io release.
- Keep package versions compatible during source work. Version bumps and publication belong to a dedicated release task.
- Update all entries from an upstream repository to the same exact revision when its validated source head changes.
- Do not commit generated Cargo configuration, sibling paths, or moving Git dependencies to package manifests.
- Do not add private-repository credentials or authenticated Git fallback merely to reproduce the local multi-repository workspace in hosted CI.
- Keep a normal consumer task to the consumer plus at most two upstream repositories unless broader migration scope was explicitly assigned.

## Verification boundary

Source-mode verification in the local multi-repository workspace proves the exact source graph under development and is valid implementation evidence before publication. Hosted CI remains repository-local. The structural checker continues to enforce registry coordinates in manifests; source mode does not weaken that boundary. Distribution still requires a later release task to deactivate source mode, publish or select the minimal required crate closure, and prove clean registry-only resolution.
