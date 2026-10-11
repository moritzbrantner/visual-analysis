# Visual source development

`visual-analysis` keeps registry coordinates as its distribution contract. Ordinary cross-repository source development is intentionally limited to the lower-level `moenarch-foundation` graph.

The committed `.coding-tooling.source-deps.json` pins the foundation package closure used by this workspace to one exact `moenarch-foundation` revision and enables local-only resolution.

## Fresh-clone bootstrap (default path)

On a clean clone, run:

```bash
bash scripts/bootstrap-source-deps.sh
cargo test --locked --workspace
```

The bootstrap (#66, owner decision option B) needs no manually prepared sibling repository. It:

1. reads the declared pins: `coding-tooling` from `scripts/coding-tooling.rev`, `moenarch-foundation` from the single revision all patches in `.coding-tooling.source-deps.json` share. Each must be a full 40-character commit id; a branch, tag, mixed or missing pin fails.
2. fetches each public repository at exactly that commit into `../coding-tooling` and `../moenarch-foundation` (shallow, no credentials). A sibling that already exists is reused only when its `HEAD` is the pinned commit and it has no tracked modifications; otherwise the bootstrap stops with the expected and actual revision instead of moving it.
3. activates source mode with the pinned `coding-tooling` (`source-deps activate`), which writes the ignored `.cargo/config.toml`.

Provenance: the fetched inputs are exactly the committed pins, so the source graph of a build is identified by this repository's commit. The `Fresh-clone bootstrap` workflow checks this from an otherwise empty directory: invalid pins fail (`scripts/test_bootstrap_source_deps.sh`), the siblings are at the pinned commits and nothing else is next to the checkout, the workspace builds and tests, and the build leaves the checkout unchanged. The other hosted workflows use the same bootstrap script.

Missing local source is an error for `source-deps activate`; source mode never falls back to cloning private repositories or authenticated Git fetches. To advance a pin, change `scripts/coding-tooling.rev` or the `rev` fields of the manifest in one commit, together with the validation of that source head.

`moenarch-text-core` and `scenedetect-core` remain versioned registry dependencies where their existing compatibility/adapter surfaces require them. They are not part of the ordinary visual source workspace: NLP enrichment belongs downstream, and scene detection crosses the boundary through the explicit detector adapter rather than coordinated repository heads.

The outer coding loop owns the foundation sibling repository/worktree and may advance its pin when a task deliberately validates a newer foundation source head. Use `bash scripts/source-deps status` to inspect the mode and `bash scripts/source-deps deactivate` before registry-only release verification.

## Development contract

- Feature work may change visual code and immediate foundation source without starting a crates.io release.
- Keep package versions compatible during source work. Version bumps and publication belong to a dedicated release task.
- Do not add an NLP or scene-detection source pin merely because a registry dependency exists; cross-domain co-development requires an explicit adapter/migration task.
- Keep `video-analysis-detectors::canonical` as the explicit adapter around `scenedetect-core`; do not move scene algorithms into visual core.
- Treat the existing `text-core` uses as compatibility/data-contract surfaces, not permission for visual algorithms to depend on NLP implementation behavior.
- Do not commit generated Cargo configuration, sibling paths, or moving Git dependencies to package manifests.
- Do not add private-repository credentials or authenticated Git fallback merely to reproduce the local multi-repository workspace in hosted CI.
- Keep a normal consumer task to the consumer plus at most two upstream repositories unless broader migration scope was explicitly assigned.

## Verification boundary

Source-mode verification in the local multi-repository workspace proves the exact foundation+visual source graph under development and is valid implementation evidence before publication. Hosted CI remains repository-local. The structural checker continues to enforce registry coordinates in manifests; source mode does not weaken that boundary. Distribution still requires a later release task to deactivate source mode, publish or select the minimal required crate closure, and prove clean registry-only resolution.
