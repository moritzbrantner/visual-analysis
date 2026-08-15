# Optional verification profile

This `draft` profile describes optional independent verification for
`moritzbrantner/visual-analysis`. It supplements the repository's normal verification
contract and does not add a worker-scheduling gate.

The active command and surface map comes from `.harness/checks.toml`. The full
tier runs every blocking check declared there with `tier = "full"`; targeted
selection is early feedback only.

## Workflow

Invoke the Harness from the installed skill:

```bash
python3 "${CODEX_HOME:-$HOME/.codex}/skills/moenarch-verification-harness/scripts/verification_harness.py" \
  audit --repo-root . --json
```

Use `select` or the targeted tier only for optional early feedback. At a clean
exact candidate head, run the full tier with an independently resolved base and
exact head:

```bash
python3 "${CODEX_HOME:-$HOME/.codex}/skills/moenarch-verification-harness/scripts/verification_harness.py" \
  run --repo-root . --tier full \
  --base-ref BASE_SHA --expected-base-sha BASE_SHA --head-sha HEAD_SHA --json
```

Report any failure or evidence gap honestly. A Harness result does not replace
the repository's normal exact-head verification evidence.

Generated output is limited to the ignored paths declared by the checks.
Harness, Git, review, and result state are never generated output. Keep raw
logs, command output, prompts, secrets, environment values, and private issue
bodies out of committed files.
