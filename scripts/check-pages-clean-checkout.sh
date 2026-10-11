#!/usr/bin/env bash
# Build the GitHub Pages artifact from a clean checkout of the committed HEAD.
#
# Issue #69 (owner decision: option 3, revision-pinned source inputs): the
# check clones HEAD into an otherwise empty temporary directory, so no
# prepared sibling repositories, ignored build outputs, managed Cargo
# configuration or other untracked files are visible. It then fetches exactly
# the declared pins with scripts/bootstrap-source-deps.sh (the only entries
# allowed next to the checkout afterwards are coding-tooling and
# moenarch-foundation at their pinned commits), runs the repository's Pages
# entry point, and requires that the build left every tracked file unchanged
# and created no untracked (non-ignored) files -- a rewritten Cargo.lock, for
# example, means the build resolved dependencies the lockfile does not declare.
#
# The artifact carries build-provenance.json (commit and source pins) and is
# what pages.yml uploads and deploys; scripts/check-pages-provenance.mjs
# verifies it against a checkout.
#
# Usage: bash scripts/check-pages-clean-checkout.sh [OUTPUT_DIR]
# OUTPUT_DIR (default: _site) receives the clean-checkout artifact. Uncommitted
# changes in the working tree are deliberately NOT part of the build.
set -euo pipefail

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUTPUT_DIR="$(realpath -m "${1:-$ROOT_DIR/_site}")"
HEAD_SHA="$(git -C "$ROOT_DIR" rev-parse HEAD)"

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/visual-pages-clean.XXXXXX")"
cleanup() { rm -rf "$WORK_DIR"; }
trap cleanup EXIT
CHECKOUT="$WORK_DIR/visual-analysis"

git clone --quiet --no-local --no-checkout "$ROOT_DIR" "$CHECKOUT"
git -C "$CHECKOUT" checkout --quiet --detach "$HEAD_SHA"

# Nothing but the checkout may exist next to it before the bootstrap.
siblings="$(find "$WORK_DIR" -mindepth 1 -maxdepth 1 ! -path "$CHECKOUT" -printf '%f\n')"
if [[ -n "$siblings" ]]; then
  printf 'unexpected entries next to the clean checkout:\n%s\n' "$siblings" >&2
  exit 1
fi

# Inputs a developer machine may provide implicitly must not be used.
unset CODING_TOOLING_DIR CARGO_TARGET_DIR
bash "$CHECKOUT/scripts/bootstrap-source-deps.sh"

# Afterwards only the declared source inputs may exist next to it.
siblings="$(find "$WORK_DIR" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort | tr '\n' ' ')"
if [[ "$siblings" != "coding-tooling moenarch-foundation visual-analysis " ]]; then
  printf 'undeclared inputs next to the clean checkout: %s\n' "$siblings" >&2
  exit 1
fi

printf 'Building Pages from a clean checkout of %s in %s\n' "$HEAD_SHA" "$CHECKOUT"
(
  cd "$CHECKOUT"
  node --test tests/pages/*.test.mjs
  bash scripts/build-pages.sh "$WORK_DIR/site"
  bash scripts/source-deps deactivate >/dev/null
)

dirty="$(git -C "$CHECKOUT" status --porcelain --untracked-files=all)"
if [[ -n "$dirty" ]]; then
  printf 'the Pages build modified tracked files or left untracked inputs/outputs in the clean checkout:\n%s\n' "$dirty" >&2
  git -C "$CHECKOUT" --no-pager diff --stat >&2 || true
  exit 1
fi

node "$CHECKOUT/scripts/check-pages-artifact.mjs" "$WORK_DIR/site"

# Record which declared inputs built this artifact, then check the record
# against the clean checkout and its fetched siblings.
python3 - "$CHECKOUT" "$WORK_DIR/site/build-provenance.json" <<'PY'
import json
import subprocess
import sys

checkout, target = sys.argv[1], sys.argv[2]

def head(path):
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()

record = {
    "schema_version": 1,
    "repository": "moritzbrantner/visual-analysis",
    "commit": head(checkout),
    "built_from": "clean-checkout",
    "source_inputs": {
        "coding-tooling": head(f"{checkout}/../coding-tooling"),
        "moenarch-foundation": head(f"{checkout}/../moenarch-foundation"),
    },
}
with open(target, "w") as handle:
    json.dump(record, handle, indent=2)
    handle.write("\n")
PY
node "$CHECKOUT/scripts/check-pages-provenance.mjs" "$WORK_DIR/site" "$CHECKOUT"

rm -rf "$OUTPUT_DIR"
mkdir -p "$(dirname "$OUTPUT_DIR")"
cp -R "$WORK_DIR/site" "$OUTPUT_DIR"
printf 'Clean-checkout Pages artifact: %s\n' "$OUTPUT_DIR"
