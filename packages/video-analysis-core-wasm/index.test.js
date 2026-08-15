import { beforeAll, expect, test } from "bun:test";

import init, {
  analyzeVideoFrame,
  frameTimecode,
  parseFrameTimecode,
  scenesFromCutFrames,
} from "./index.js";

beforeAll(async () => {
  await init();
});

test("analyzes video frame pixels through packaged wasm bindings", () => {
  const data = new Uint8Array([
    255, 0, 0,
    0, 255, 0,
    0, 0, 255,
    255, 255, 255,
  ]);

  const analysis = analyzeVideoFrame(data, 2, 2, "rgb24", 12, 24, 1, 3);

  expect(analysis.pixelCount).toBe(4);
  expect(analysis.timecode).toBe("00:00:00.500");
  expect(analysis.topLeft).toEqual({ r: 255, g: 0, b: 0 });
  expect(analysis.meanRgb.r).toBeCloseTo(127.5);
});

test("converts and parses timecodes and creates scenes", () => {
  expect(frameTimecode(48, 24, 1, 2)).toEqual({
    frameIndex: 48,
    seconds: 2,
    timecode: "00:00:02.00",
  });
  expect(parseFrameTimecode("00:00:02.00", 24, 1, 2).frameIndex).toBe(48);
  expect(scenesFromCutFrames([24, 48], 72, 24, 1)).toEqual([
    { startFrame: 0, endFrame: 24, startSeconds: 0, endSeconds: 1 },
    { startFrame: 24, endFrame: 48, startSeconds: 1, endSeconds: 2 },
    { startFrame: 48, endFrame: 72, startSeconds: 2, endSeconds: 3 },
  ]);
});
