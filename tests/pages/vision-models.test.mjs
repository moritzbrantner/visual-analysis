import test from "node:test";
import assert from "node:assert/strict";

import {
  BROWSER_VISION_MODELS,
  OPEN_VOCAB_MODEL_ID,
  SAM_MODEL_ID,
  normalizeOpenVocabularyDetection,
  scalePixelBoxToSamInput,
  summarizeBinaryMask,
} from "../../site/vision-models.js";

test("browser learned vision models are explicit and opt-in", () => {
  assert.deepEqual(
    BROWSER_VISION_MODELS.map(({ id, optIn, prompts }) => ({ id, optIn, prompts })),
    [
      { id: SAM_MODEL_ID, optIn: true, prompts: ["point", "box"] },
      { id: OPEN_VOCAB_MODEL_ID, optIn: true, prompts: ["text"] },
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

test("SAM box prompts scale from original pixels into the resized prompt space", () => {
  assert.deepEqual(
    scalePixelBoxToSamInput(
      { x: 100, y: 50, width: 300, height: 200 },
      [500, 1000],
      [512, 1024],
    ),
    [102.4, 51.2, 409.6, 256],
  );
});

test("SAM box prompts clamp original edges to the image and reject empty regions", () => {
  assert.deepEqual(
    scalePixelBoxToSamInput(
      { x: -50, y: 25, width: 60, height: 100 },
      [100, 100],
      [1024, 1024],
    ),
    [0, 256, 102.4, 1024],
  );
  assert.throws(
    () => scalePixelBoxToSamInput({ x: 20, y: 20, width: 0, height: 5 }, [100, 100], [1024, 1024]),
    /non-zero region/,
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
