# Codex cloud environment

`visual-analysis` uses a repository-owned setup script so Codex cloud, local development, and future CI can converge on the same prerequisites instead of relying on an undocumented container state.

## Codex environment settings

Create one Codex cloud environment with:

- **Name:** `visual-analysis`
- **Repository:** `moritzbrantner/visual-analysis`
- **Setup script:** `bash scripts/codex-environment.sh setup`
- **Maintenance script:** `bash scripts/codex-environment.sh maintenance`
- **Agent internet access:** off by default
- **Environment variables:** none required initially
- **Secrets:** none required initially

Codex setup scripts have internet access, so dependency and tool acquisition happens before the agent phase. The normal agent phase should remain offline unless an individual task explicitly requires network access.

## What setup installs and pins

The setup script:

1. installs native media/document prerequisites used by visual consumers and acceptance work: FFmpeg, Poppler, Tesseract English/German, compiler/build prerequisites, and Python venv support;
2. installs the exact Bun version declared by `package.json`;
3. installs the exact Rust toolchain, rustfmt, and Clippy declared by `rust-toolchain.toml`;
4. prepares ONNX Runtime 1.24.4 and persists `ORT_DYLIB_PATH` for agent shells;
5. reads `.coding-tooling.source-deps.json`, verifies that it declares exactly one source repository, and checks out the exact `moenarch-foundation` revision as a sibling workspace;
6. checks out a pinned `coding-tooling` revision as tooling rather than a domain source dependency;
7. installs frozen Bun dependencies;
8. activates the managed exact source-development graph through `scripts/source-deps`;
9. warms Cargo dependencies; and
10. runs structural/environment preflight checks.

The setup deliberately does **not** download large visual model bundles or quality-corpus assets. Those belong to the explicit visual-acceptance setup introduced by the behavioral-verification work, where model revisions and fixture checksums can be reviewed and pinned independently.

## Source-boundary rationale

`visual-analysis` declares `maxExactSourceRepositories: 1`. The Codex environment therefore materializes only `moenarch-foundation` as an exact source dependency. `nlp-stack` and `scenedetect-rs` remain adapter/registry boundaries unless a task explicitly changes that architecture.

`coding-tooling` is checked out separately because `scripts/source-deps` delegates the deterministic source-patch mechanics to it. It is tooling and does not expand the domain dependency graph.

## Maintenance behavior

Codex may resume a cached container on a newer `visual-analysis` commit. The maintenance script therefore:

- re-reads the current exact foundation revision;
- resets the sibling foundation checkout to that revision;
- resets `coding-tooling` to its pinned tooling revision;
- refreshes frozen Bun dependencies;
- re-activates managed source dependencies;
- refreshes the Cargo cache; and
- reruns the environment preflight.

If the setup script, maintenance command, environment variables, or secrets change, reset the Codex environment cache so the new configuration is applied from a clean cached image.

## Local reproduction

The same setup can be exercised in a disposable Linux development environment with:

```bash
bash scripts/codex-environment.sh setup
```

A normal environment refresh uses:

```bash
bash scripts/codex-environment.sh maintenance
```

After setup, source mode should report the exact managed graph and Cargo metadata should resolve without depending on unpublished intermediate packages.
