#!/usr/bin/env bash
# Negative checks for scripts/bootstrap-source-deps.sh (#66): moving, invalid
# or mismatched pins and a sibling at another revision must fail visibly and
# never activate source mode. Runs against a throwaway clone of HEAD in a
# temporary directory; the real checkout and its siblings are not touched.
set -euo pipefail

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/visual-bootstrap-test.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT
failures=0

fresh_clone() {
  rm -rf "$WORK_DIR/case"
  mkdir -p "$WORK_DIR/case"
  git clone --quiet --no-local "$ROOT_DIR" "$WORK_DIR/case/visual-analysis"
  # Uncommitted bootstrap changes under test are part of the exercised script.
  cp "$ROOT_DIR/scripts/bootstrap-source-deps.sh" "$WORK_DIR/case/visual-analysis/scripts/"
}

expect_failure() {
  local name="$1" pattern="$2"
  local log="$WORK_DIR/$name.log"
  if bash "$WORK_DIR/case/visual-analysis/scripts/bootstrap-source-deps.sh" >"$log" 2>&1; then
    printf 'FAIL %s: bootstrap succeeded\n' "$name" >&2
    failures=$((failures + 1))
  elif ! grep -q -- "$pattern" "$log"; then
    printf 'FAIL %s: expected "%s" in:\n' "$name" "$pattern" >&2
    cat "$log" >&2
    failures=$((failures + 1))
  elif [[ -e "$WORK_DIR/case/visual-analysis/.cargo/config.toml" ]]; then
    printf 'FAIL %s: source mode was activated\n' "$name" >&2
    failures=$((failures + 1))
  else
    printf 'ok %s\n' "$name"
  fi
}

manifest() { printf '%s' "$WORK_DIR/case/visual-analysis/.coding-tooling.source-deps.json"; }
foundation_rev() {
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["cargo"]["patches"][0]["rev"])' "$(manifest)"
}

fresh_clone
sed -i "s/$(foundation_rev)/main/g" "$(manifest)"
expect_failure moving-foundation-pin "is not a full 40-character commit id"

fresh_clone
printf 'main\n' >"$WORK_DIR/case/visual-analysis/scripts/coding-tooling.rev"
expect_failure moving-tooling-pin "is not a full 40-character commit id"

fresh_clone
sed -i "0,/$(foundation_rev)/s//0000000000000000000000000000000000000000/" "$(manifest)"
expect_failure mixed-foundation-pins "expected one Foundation revision"

fresh_clone
sed -i "s/$(foundation_rev)/0000000000000000000000000000000000000000/g" "$(manifest)"
expect_failure unresolvable-foundation-pin "cannot fetch"
leftover="$(find "$WORK_DIR/case" -mindepth 1 -maxdepth 1 ! -name visual-analysis ! -name coding-tooling -printf '%f\n')"
if [[ -n "$leftover" ]]; then
  printf 'FAIL unresolvable-foundation-pin: a failed fetch left %s behind\n' "$leftover" >&2
  failures=$((failures + 1))
else
  printf 'ok failed-fetch-leaves-no-sibling\n'
fi

fresh_clone
git init --quiet "$WORK_DIR/case/moenarch-foundation"
git -C "$WORK_DIR/case/moenarch-foundation" -c user.name=test -c user.email=test@invalid \
  commit --quiet --allow-empty -m "not the pinned revision"
expect_failure sibling-at-another-revision "but the declared pin is"

if ((failures > 0)); then
  printf '%d bootstrap check(s) failed\n' "$failures" >&2
  exit 1
fi
