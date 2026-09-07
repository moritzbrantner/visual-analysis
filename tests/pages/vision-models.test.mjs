import test from "node:test";
import assert from "node:assert/strict";

import {
  BROWSER_VISION_MODELS,
  OPEN_VOCAB_MODEL_ID,
  SAM_MODEL_ID,
  normalizeOpenVocabularyDetection,
  summarizeBinaryMask,
} from "../../site/vision-models.js";

test("browser learned vision models are explicit and opt-in", () => {
  assert.deepEqual(
    BROWSER_VISION_MODELS.map(({ id, optIn }) => ({ id, optIn })),
    [
      { id: SAM_MODEL_ID, optIn: true },
      { id: OPEN_VOCAB_MODEL_ID, optIn: true },
    ],
  );
});

test("open-vocabulary detections normalize to the canonical pixel-region shape", () => {
  assert.deepEqual(
    normalizeOpenVocabularyDetection({
      label: "cat",
      score: 0.75,
      box: { xmin: 10.2, ymin: 20.6, xmax: 45.7, ymax: 70.1 },
    }),
    {
      label: "cat",
      score: 0.75,
      region: { x: 10, y: 21, width: 36, height: 49 },
      attributes: {
        backend: "transformers.js-zero-shot-object-detection",
        modelId: OPEN_VOCAB_MODEL_ID,
        promptKind: "text",
      },
    },
  );
});

test("binary mask summaries preserve active-pixel counts and tight bounds", () => {
  const data = Uint8Array.from([
    0, 0, 0, 0,
    0, 255, 255, 0,
    0, 255, 0, 0,
  ]);
  assert.deepEqual(summarizeBinaryMask(data, 4, 3), {
    activePixels: 3,
    region: { x: 1, y: 1, width: 2, height: 2 },
  });
});
