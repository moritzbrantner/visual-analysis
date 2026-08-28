# Visual source development

`visual-analysis` keeps registry coordinates as its distribution contract. Ordinary cross-repository source development is intentionally limited to the lower-level `moenarch-foundation` graph.

The committed `.coding-tooling.source-deps.json` pins the foundation package closure used by this workspace to one exact `moenarch-foundation` revision and enables local-only resolution. Run `bash scripts/source-deps activate` only after the sibling foundation checkout exists at that exact Git `HEAD`. Missing local source is an error; source mode never falls back to cloning private repositories or authenticated Git fetches.

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
