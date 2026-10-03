#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURE_DIR="${VISUAL_ACCEPTANCE_FIXTURES_DIR:-$ROOT/.external-test-tools/visual-fixtures}"
TRANSFORMERS_REVISION="5fcf605fda6c7c3982f62807f2eda210aa37e9e1"
COCO_BLOB_SHA="a3b5225fc3cef5c492cc109aebe883f24941a156"
COCO_NAME="coco-000000039769.png"
COCO_PATH="$FIXTURE_DIR/$COCO_NAME"
COCO_URL="https://raw.githubusercontent.com/huggingface/transformers/$TRANSFORMERS_REVISION/tests/fixtures/tests_samples/COCO/000000039769.png"
OPENCV_REVISION="c7b9dc388c5770f3457f07742f1223068b5e27d4"
FACE_BLOB_SHA="f06aa74a57ce3a4129340cd4407ef3c0558e3193"
FACE_NAME="opencv-lena.jpg"
FACE_PATH="$FIXTURE_DIR/$FACE_NAME"
FACE_URL="https://raw.githubusercontent.com/opencv/opencv/$OPENCV_REVISION/samples/data/lena.jpg"

mkdir -p "$FIXTURE_DIR"

verify_coco_fixture() {
  [[ -s "$COCO_PATH" ]] && [[ "$(git hash-object "$COCO_PATH")" == "$COCO_BLOB_SHA" ]]
}

verify_face_fixture() {
  [[ -s "$FACE_PATH" ]] && [[ "$(git hash-object "$FACE_PATH")" == "$FACE_BLOB_SHA" ]]
}

if ! verify_coco_fixture; then
  rm -f "$COCO_PATH"
  curl --fail --location --retry 3 --retry-delay 2 --output "$COCO_PATH" "$COCO_URL"
fi

if ! verify_face_fixture; then
  rm -f "$FACE_PATH"
  curl --fail --location --retry 3 --retry-delay 2 --output "$FACE_PATH" "$FACE_URL"
fi

if ! verify_coco_fixture; then
  echo "visual acceptance fixture failed pinned Git blob verification: $COCO_PATH" >&2
  exit 2
fi
if ! verify_face_fixture; then
  echo "visual acceptance fixture failed pinned Git blob verification: $FACE_PATH" >&2
  exit 2
fi

printf 'visual acceptance fixtures ready: %s %s\n' "$COCO_PATH" "$FACE_PATH"
