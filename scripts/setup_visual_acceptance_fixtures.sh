#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURE_DIR="${VISUAL_ACCEPTANCE_FIXTURES_DIR:-$ROOT/.external-test-tools/visual-fixtures}"
TRANSFORMERS_REVISION="5fcf605fda6c7c3982f62807f2eda210aa37e9e1"
COCO_BLOB_SHA="a3b5225fc3cef5c492cc109aebe883f24941a156"
COCO_NAME="coco-000000039769.png"
COCO_PATH="$FIXTURE_DIR/$COCO_NAME"
COCO_URL="https://raw.githubusercontent.com/huggingface/transformers/$TRANSFORMERS_REVISION/tests/fixtures/tests_samples/COCO/000000039769.png"

mkdir -p "$FIXTURE_DIR"

verify_fixture() {
  [[ -s "$COCO_PATH" ]] && [[ "$(git hash-object "$COCO_PATH")" == "$COCO_BLOB_SHA" ]]
}

if ! verify_fixture; then
  rm -f "$COCO_PATH"
  curl --fail --location --retry 3 --retry-delay 2 --output "$COCO_PATH" "$COCO_URL"
fi

if ! verify_fixture; then
  echo "visual acceptance fixture failed pinned Git blob verification: $COCO_PATH" >&2
  exit 2
fi

printf 'visual acceptance fixture ready: %s\n' "$COCO_PATH"
