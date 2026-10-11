#!/usr/bin/env bash
# Fresh-clone source bootstrap (#66).
#
# Fetches exactly the declared, revision-pinned source inputs next to this
# checkout and activates them through the existing coding-tooling source-deps
# mechanism, so a clean clone builds without manually prepared siblings:
#
#   - ../coding-tooling       at the revision in scripts/coding-tooling.rev
#   - ../moenarch-foundation  at the one revision every Foundation patch in
#                             .coding-tooling.source-deps.json declares
#
# A sibling that already exists is used only when its HEAD is exactly the
# pinned revision and it has no tracked modifications; anything else fails
# with the expected and actual revision. Revisions must be full 40-character
# commit ids: no branch, tag or other moving revision is accepted. Public
# repositories only; no credentials are used.
#
# Usage: bash scripts/bootstrap-source-deps.sh
# Afterwards `bash scripts/source-deps status` shows the active mode, and
# `bash scripts/source-deps deactivate` restores registry coordinates.
set -euo pipefail

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
PARENT_DIR="$(dirname "$ROOT_DIR")"
TOOLING_URL="https://github.com/moritzbrantner/coding-tooling"
FOUNDATION_URL="https://github.com/moritzbrantner/moenarch-foundation"

fail() {
  printf 'bootstrap-source-deps: %s\n' "$*" >&2
  exit 1
}

require_commit_id() {
  [[ "$2" =~ ^[0-9a-f]{40}$ ]] || fail "$1 pin '$2' is not a full 40-character commit id"
}

tooling_pin_file="$ROOT_DIR/scripts/coding-tooling.rev"
[[ -f "$tooling_pin_file" ]] || fail "missing $tooling_pin_file"
TOOLING_REV="$(tr -d '[:space:]' < "$tooling_pin_file")"
require_commit_id coding-tooling "$TOOLING_REV"

# The Foundation revision and sibling directory come from the committed
# source-deps manifest; every patch must name the same repository revision
# and a path inside ../moenarch-foundation.
FOUNDATION_REV="$(python3 - "$ROOT_DIR/.coding-tooling.source-deps.json" "$FOUNDATION_URL" <<'PY'
import json
import sys

path, url = sys.argv[1], sys.argv[2]
try:
    patches = json.load(open(path))["cargo"]["patches"]
except (OSError, KeyError, TypeError, ValueError) as error:
    sys.exit(f"bootstrap-source-deps: cannot read cargo patches from {path}: {error}")
if not patches:
    sys.exit(f"bootstrap-source-deps: {path} declares no cargo patches")
revisions = set()
for patch in patches:
    if patch.get("git") != url:
        sys.exit(f"bootstrap-source-deps: patch {patch.get('package')} is not from {url}")
    if not str(patch.get("localPath", "")).startswith("../moenarch-foundation/"):
        sys.exit(f"bootstrap-source-deps: patch {patch.get('package')} is not under ../moenarch-foundation/")
    revisions.add(patch.get("rev"))
if len(revisions) != 1:
    sys.exit(f"bootstrap-source-deps: expected one Foundation revision, found {sorted(map(str, revisions))}")
print(revisions.pop())
PY
)"
require_commit_id moenarch-foundation "$FOUNDATION_REV"

ensure_checkout() {
  local name="$1" url="$2" rev="$3"
  local dir="$PARENT_DIR/$name"
  if [[ -e "$dir" ]]; then
    local head
    head="$(git -C "$dir" rev-parse --verify HEAD 2>/dev/null)" \
      || fail "$dir exists but is not a Git checkout; remove it or check out $url at $rev"
    [[ "$head" == "$rev" ]] \
      || fail "$dir is at $head, but the declared pin is $rev; check out the pinned revision or move the sibling away"
    [[ -z "$(git -C "$dir" status --porcelain --untracked-files=no)" ]] \
      || fail "$dir has tracked modifications; the pinned source must be unmodified"
    printf 'source input %s: existing checkout at pinned %s\n' "$name" "$rev"
    return
  fi
  # Fetch into a temporary directory and move it into place only once it is
  # complete, so an interrupted or failed fetch never leaves a half-made
  # sibling that a rerun would refuse.
  local partial
  partial="$(mktemp -d "$PARENT_DIR/.$name.partial.XXXXXX")"
  git init --quiet "$partial"
  if ! git -C "$partial" fetch --quiet --depth=1 "$url" "$rev" \
    || ! git -C "$partial" checkout --quiet --detach FETCH_HEAD; then
    rm -rf "$partial"
    fail "cannot fetch $url at pinned $rev"
  fi
  local head
  head="$(git -C "$partial" rev-parse HEAD)"
  if [[ "$head" != "$rev" ]]; then
    rm -rf "$partial"
    fail "$url fetched $head instead of pinned $rev"
  fi
  mv "$partial" "$dir"
  printf 'source input %s: fetched %s at %s\n' "$name" "$url" "$rev"
}

ensure_checkout coding-tooling "$TOOLING_URL" "$TOOLING_REV"
ensure_checkout moenarch-foundation "$FOUNDATION_URL" "$FOUNDATION_REV"

(cd "$PARENT_DIR/coding-tooling" && bun install --frozen-lockfile --silent)

# Use the pinned tooling, never an installed `coding-tooling` of another version.
cd "$ROOT_DIR"
bun "$PARENT_DIR/coding-tooling/src/cli.ts" source-deps activate
