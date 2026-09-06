import test from "node:test";
import assert from "node:assert/strict";

import {
  fitDimensions,
  formatBytes,
  histogramQuantile,
  meanLumaFromHistogram,
  rgbaToRgb24,
  rgbToHex,
  sampleTimes,
} from "../../site/analysis.js";

test("fitDimensions leaves small inputs untouched and bounds large inputs", () => {
  assert.deepEqual(fitDimensions(640, 480, 640 * 480), { width: 640, height: 480, scale: 1 });
  const bounded = fitDimensions(4000, 3000, 750000);
  assert.ok(bounded.width * bounded.height <= 755000);
  assert.ok(bounded.scale < 1);
});

test("rgbaToRgb24 drops alpha without reordering channels", () => {
  assert.deepEqual(Array.from(rgbaToRgb24(Uint8Array.from([1, 2, 3, 4, 5, 6, 7, 8]))), [1, 2, 3, 5, 6, 7]);
});

test("histogram helpers produce stable luma summaries", () => {
  const histogram = [0, 1, 0, 1];
  assert.equal(meanLumaFromHistogram(histogram), 160);
  assert.ok(histogramQuantile(histogram, 0.5) > 64);
});

test("presentation helpers stay deterministic", () => {
  assert.equal(rgbToHex({ red: 255, green: 128, blue: 0 }), "#ff8000");
  assert.equal(formatBytes(1536), "1.50 KB");
  assert.deepEqual(sampleTimes(8, 3).map((value) => Number(value.toFixed(2))), [0.05, 4, 7.95]);
});
