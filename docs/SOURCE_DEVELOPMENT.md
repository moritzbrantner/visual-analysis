# Visual source development

`visual-analysis` keeps registry coordinates as its distribution contract, but ordinary cross-repository development does not require publishing foundation, NLP, or scene-detection changes first.

The committed `.coding-tooling.source-deps.json` pins the external package closure used by this workspace to exact revisions of `moenarch-foundation`, `nlp-stack`, and `scenedetect-rs`. Run `bash scripts/source-deps activate` to materialize the ignored `.cargo/config.toml`. For each available sibling checkout, coding-tooling requires Git `HEAD` to equal the declared revision; otherwise it uses that exact Git revision when the repository is accessible.

Use `bash scripts/source-deps status` to inspect the mode and `bash scripts/source-deps deactivate` before registry-only release verification.

## Development contract

- Feature work may change visual code and immediate upstream source without starting a crates.io release.
- Keep package versions compatible during source work. Version bumps and publication belong to a dedicated release task.
- Update all entries from an upstream repository to the same exact revision when its validated source head changes.
- Do not commit generated Cargo configuration, sibling paths, or moving Git dependencies to package manifests.
- Keep a normal consumer task to the consumer plus at most two upstream repositories unless broader migration scope was explicitly assigned.

## Verification boundary

Source-mode verification proves the exact source graph under development and is valid implementation evidence before publication. The structural checker continues to enforce registry coordinates in manifests; source mode does not weaken that boundary. Distribution still requires a later release task to deactivate source mode, publish or select the minimal required crate closure, and prove clean registry-only resolution.
