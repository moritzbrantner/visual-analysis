#!/usr/bin/env bash
# Build the GitHub Pages artifact from a clean checkout of the committed HEAD.
#
# NOT wired into CI: whether Pages must build without the pinned source-deps
# sibling checkouts (publish moenarch-semantic-core, a minimal Pages crate
# closure, or keep pinned checkouts) awaits the owner decision in
# moritzbrantner/visual-analysis#69. On the current baseline this check fails
# because moenarch-semantic-core is not on crates.io. Run it manually.
#
# Issue #67: Pages must build from the repository alone, using declared
# dependencies. This check clones HEAD into an otherwise empty temporary
# directory (so no sibling repositories, ignored build outputs, managed Cargo
# configuration or other untracked files are visible), runs the repository's
# Pages entry point there, and then requires that the build left every tracked
# file unchanged and created no untracked (non-ignored) files -- for example a
# rewritten Cargo.lock means the build resolved dependencies the lockfile does
# not declare.
#
# Usage: bash scripts/check-pages-clean-checkout.sh [OUTPUT_DIR]
# OUTPUT_DIR (default: _site) receives the clean-checkout artifact for the
# browser acceptance suite. Uncommitted changes in the working tree are
# deliberately NOT part of the build.
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

# Nothing but the checkout may exist next to it: sibling source workspaces
# (../moenarch-foundation, ../coding-tooling, ...) are not declared inputs.
siblings="$(find "$WORK_DIR" -mindepth 1 -maxdepth 1 ! -path "$CHECKOUT" -printf '%f\n')"
if [[ -n "$siblings" ]]; then
  printf 'unexpected entries next to the clean checkout:\n%s\n' "$siblings" >&2
  exit 1
fi

printf 'Building Pages from a clean checkout of %s in %s\n' "$HEAD_SHA" "$CHECKOUT"
(
  cd "$CHECKOUT"
  # Inputs a developer machine may provide implicitly must not be used.
  unset CODING_TOOLING_DIR CARGO_TARGET_DIR
  bash scripts/build-pages.sh "$WORK_DIR/site"
)

dirty="$(git -C "$CHECKOUT" status --porcelain --untracked-files=all)"
if [[ -n "$dirty" ]]; then
  printf 'the Pages build modified tracked files or left untracked inputs/outputs in the clean checkout:\n%s\n' "$dirty" >&2
  git -C "$CHECKOUT" --no-pager diff --stat >&2 || true
  exit 1
fi

node "$CHECKOUT/scripts/check-pages-artifact.mjs" "$WORK_DIR/site"

rm -rf "$OUTPUT_DIR"
mkdir -p "$(dirname "$OUTPUT_DIR")"
cp -R "$WORK_DIR/site" "$OUTPUT_DIR"
printf 'Clean-checkout Pages artifact: %s\n' "$OUTPUT_DIR"
