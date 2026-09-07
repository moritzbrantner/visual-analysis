import("./vision-ui.js").catch((error) => {
  console.warn("Learned vision UI failed to initialize", error);
});

export const IMAGE_ANALYSIS_MAX_PIXELS = 786432;
export const VIDEO_FRAME_MAX_PIXELS = 196608;
export const VIDEO_SAMPLE_COUNT = 9;

export function fitDimensions(width, height, maxPixels) {
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    throw new Error("width and height must be positive finite numbers");
  }
  if (!Number.isFinite(maxPixels) || maxPixels <= 0) {
    throw new Error("maxPixels must be a positive finite number");
  }
  const pixels = width * height;
  if (pixels <= maxPixels) {
    return { width: Math.round(width), height: Math.round(height), scale: 1 };
  }
  const scale = Math.sqrt(maxPixels / pixels);
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
    scale,
  };
}

export function rgbaToRgb24(rgba) {
  const rgb = new Uint8Array(Math.floor(rgba.length / 4) * 3);
  let target = 0;
  for (let source = 0; source + 3 < rgba.length; source += 4) {
    rgb[target++] = rgba[source];
    rgb[target++] = rgba[source + 1];
    rgb[target++] = rgba[source + 2];
  }
  return rgb;
}

export function rgbToHex({ red, green, blue }) {
  const values = [red, green, blue].map((value) =>
    Math.max(0, Math.min(255, Math.round(Number(value) || 0))).toString(16).padStart(2, "0"),
  );
  return `#${values.join("")}`;
}

export function meanLumaFromHistogram(histogram) {
  if (!Array.isArray(histogram) || histogram.length === 0) return 0;
  const total = histogram.reduce((sum, value) => sum + Number(value || 0), 0);
  if (total <= 0) return 0;
  const binWidth = 256 / histogram.length;
  const weighted = histogram.reduce(
    (sum, count, index) => sum + Number(count || 0) * (index * binWidth + binWidth / 2),
    0,
  );
  return weighted / total;
}

export function histogramQuantile(histogram, quantile) {
  if (!Array.isArray(histogram) || histogram.length === 0) return 0;
  const total = histogram.reduce((sum, value) => sum + Number(value || 0), 0);
  if (total <= 0) return 0;
  const target = Math.max(0, Math.min(1, quantile)) * total;
  let seen = 0;
  for (let index = 0; index < histogram.length; index += 1) {
    seen += Number(histogram[index] || 0);
    if (seen >= target) {
      return ((index + 0.5) * 256) / histogram.length;
    }
  }
  return 255;
}

export function histogramSpread(histogram) {
  return Math.max(0, histogramQuantile(histogram, 0.9) - histogramQuantile(histogram, 0.1));
}

export function buildFindings({ histogram, width, height, analyzedWidth, analyzedHeight, mediaKind }) {
  const findings = [];
  const meanLuma = meanLumaFromHistogram(histogram);
  const spread = histogramSpread(histogram);

  if (meanLuma < 72) {
    findings.push("The analyzed pixels are predominantly dark.");
  } else if (meanLuma > 188) {
    findings.push("The analyzed pixels are predominantly bright.");
  } else {
    findings.push("Overall brightness is centered in the mid-range.");
  }

  if (spread < 70) {
    findings.push("Luma is concentrated in a relatively narrow range, suggesting lower tonal contrast.");
  } else if (spread > 165) {
    findings.push("The image spans a broad luma range, with strong dark-to-bright separation.");
  } else {
    findings.push("The luma distribution has a moderate tonal spread.");
  }

  if (mediaKind === "image" && (analyzedWidth !== width || analyzedHeight !== height)) {
    findings.push(
      `The browser downscaled the analysis buffer to ${analyzedWidth}×${analyzedHeight} to keep WASM work bounded; the original preview remains unchanged.`,
    );
  }

  return findings;
}

export function formatBytes(bytes) {
  const value = Number(bytes) || 0;
  if (value < 1024) return `${value} B`;
  const units = ["KB", "MB", "GB"];
  let scaled = value / 1024;
  for (const unit of units) {
    if (scaled < 1024 || unit === units.at(-1)) {
      return `${scaled.toFixed(scaled >= 100 ? 0 : scaled >= 10 ? 1 : 2)} ${unit}`;
    }
    scaled /= 1024;
  }
  return `${value} B`;
}

export function sampleTimes(duration, count = VIDEO_SAMPLE_COUNT) {
  if (!Number.isFinite(duration) || duration <= 0) return [0];
  const actualCount = Math.max(1, Math.min(16, Math.floor(count)));
  if (actualCount === 1) return [duration / 2];
  const edge = Math.min(0.05, duration * 0.01);
  const start = edge;
  const end = Math.max(start, duration - edge);
  return Array.from({ length: actualCount }, (_, index) =>
    start + ((end - start) * index) / (actualCount - 1),
  );
}
