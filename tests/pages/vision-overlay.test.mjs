import test from "node:test";
import assert from "node:assert/strict";

import { imagePointFromClient } from "../../site/vision-overlay.js";

test("SAM points use the displayed portrait image rather than its letterboxed frame", () => {
  const image = { left: 360, top: 20, width: 280, height: 560 };
  assert.deepEqual(imagePointFromClient(430, 300, image), { x: 0.25, y: 0.5 });
  assert.equal(imagePointFromClient(100, 300, image), null);
});

test("SAM points follow resized images and reject vertical margins", () => {
  const image = { left: 10, top: 150, width: 400, height: 100 };
  assert.deepEqual(imagePointFromClient(310, 175, image), { x: 0.75, y: 0.25 });
  assert.equal(imagePointFromClient(310, 149, image), null);
  assert.equal(imagePointFromClient(310, 251, image), null);
  assert.equal(imagePointFromClient(411, 175, image), null);
  assert.equal(imagePointFromClient(10, 150, { ...image, width: 0 }), null);
});

test("SAM points preserve foreground/background edge coordinates without clamping padding", () => {
  const image = { left: 10, top: 20, width: 100, height: 200 };
  assert.deepEqual(imagePointFromClient(10, 20, image), { x: 0, y: 0 });
  assert.deepEqual(imagePointFromClient(110, 220, image), { x: 1, y: 1 });
  assert.equal(imagePointFromClient(Number.NaN, 20, image), null);
});
