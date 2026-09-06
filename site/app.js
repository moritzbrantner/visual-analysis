import {
  IMAGE_ANALYSIS_MAX_PIXELS,
  VIDEO_FRAME_MAX_PIXELS,
  VIDEO_SAMPLE_COUNT,
  buildFindings,
  fitDimensions,
  formatBytes,
  histogramQuantile,
  histogramSpread,
  meanLumaFromHistogram,
  rgbaToRgb24,
  rgbToHex,
  sampleTimes,
} from "./analysis.js";

const elements = Object.fromEntries(
  [
    "input-panel", "drop-zone", "file-input", "choose-file", "input-error", "loading-panel", "loading-title",
    "loading-detail", "progress-decode", "progress-analyze", "progress-report", "report", "report-title",
    "export-json", "choose-another", "runtime-summary", "summary-kind", "summary-dimensions", "summary-analyzed",
    "summary-color", "summary-luma", "summary-hash", "coverage-badge", "findings", "file-meta", "preview-image",
    "preview-video", "preview-note", "histogram", "histogram-readout", "timeline-section", "timeline-nav",
    "timeline", "timeline-badge", "technical-metrics", "raw-json",
  ].map((id) => [id, document.getElementById(id)]),
);

let wasmPromise;
let currentReport = null;
let currentObjectUrl = null;

function setError(message = "") {
  elements["input-error"].hidden = !message;
  elements["input-error"].textContent = message;
}

function setProgress(step, detail) {
  const order = ["decode", "analyze", "report"];
  const activeIndex = order.indexOf(step);
  for (const [index, name] of order.entries()) {
    const item = elements[`progress-${name}`];
    item.classList.toggle("is-done", index < activeIndex);
    item.classList.toggle("is-active", index === activeIndex);
  }
  elements["loading-detail"].textContent = detail;
}

async function loadWasm() {
  wasmPromise ??= Promise.all([
    import("./wasm/image-analysis-core/index.js"),
    import("./wasm/image-analysis-processing/index.js"),
  ]).then(async ([core, processing]) => {
    await Promise.all([core.init(), processing.init()]);
    elements["runtime-summary"].textContent = "Rust/WASM image core and processing runtimes are ready.";
    return { core, processing };
  });
  return wasmPromise;
}

function surfaceValue(response) {
  return response?.value?.result ?? response?.value ?? response;
}

async function runCore(operation, input) {
  const { core } = await loadWasm();
  return surfaceValue(await core.runOperation({ operation, input }));
}

async function runProcessing(operation, input) {
  const { processing } = await loadWasm();
  return surfaceValue(await processing.runOperation({ operation, input }));
}

function payloadFromCanvas(canvas) {
  const context = canvas.getContext("2d", { willReadFrequently: true });
  const rgba = context.getImageData(0, 0, canvas.width, canvas.height).data;
  const rgb = rgbaToRgb24(rgba);
  return {
    width: canvas.width,
    height: canvas.height,
    pixelFormat: "rgb24",
    stride: null,
    data: Array.from(rgb),
  };
}

function canvasFromSource(source, sourceWidth, sourceHeight, maxPixels) {
  const fitted = fitDimensions(sourceWidth, sourceHeight, maxPixels);
  const canvas = document.createElement("canvas");
  canvas.width = fitted.width;
  canvas.height = fitted.height;
  const context = canvas.getContext("2d", { alpha: false, willReadFrequently: true });
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(source, 0, 0, fitted.width, fitted.height);
  return { canvas, fitted };
}

async function analyzePayload(payload) {
  const hashSize = Math.max(1, Math.min(8, payload.width, payload.height));
  const [summary, histogramResponse, maskTensor, perceptualHash] = await Promise.all([
    runCore("image.core.summary", { image: payload }),
    runCore("image.core.lumaHistogram", { image: payload, bins: 32 }),
    runCore("image.core.maskTensorSummary", { image: payload, previewLimit: 32 }),
    runProcessing("image.processing.hash", { image: payload, hashSize }),
  ]);
  const histogram = histogramResponse.histogram ?? [];
  return {
    summary,
    histogram,
    histogramBins: histogramResponse.bins ?? histogram.length,
    maskTensor,
    perceptualHash: { ...perceptualHash, hash: String(perceptualHash.hash) },
    meanLuma: meanLumaFromHistogram(histogram),
  };
}

async function decodeImage(file) {
  if (typeof createImageBitmap === "function") {
    return createImageBitmap(file);
  }
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.decoding = "async";
    image.src = url;
    await image.decode();
    return image;
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function analyzeImageFile(file) {
  setProgress("decode", "Decoding the image locally in your browser.");
  const image = await decodeImage(file);
  const sourceWidth = image.width ?? image.naturalWidth;
  const sourceHeight = image.height ?? image.naturalHeight;
  const { canvas, fitted } = canvasFromSource(image, sourceWidth, sourceHeight, IMAGE_ANALYSIS_MAX_PIXELS);
  if (typeof image.close === "function") image.close();

  setProgress("analyze", "Running image summary, luma histogram and perceptual hash in Rust/WASM.");
  const payload = payloadFromCanvas(canvas);
  const analysis = await analyzePayload(payload);

  const url = URL.createObjectURL(file);
  setPreviewUrl("image", url);
  return {
    schemaVersion: 1,
    media: {
      kind: "image",
      name: file.name,
      mimeType: file.type || "unknown",
      sizeBytes: file.size,
      width: sourceWidth,
      height: sourceHeight,
    },
    coverage: {
      mode: fitted.scale === 1 ? "full decoded image" : "bounded downscaled image",
      analyzedWidth: canvas.width,
      analyzedHeight: canvas.height,
      analyzedPixels: canvas.width * canvas.height,
      scale: fitted.scale,
    },
    analysis,
  };
}

function waitForEvent(target, successEvent, errorEvent = "error") {
  return new Promise((resolve, reject) => {
    const onSuccess = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error(`Failed while waiting for ${successEvent}.`));
    };
    const cleanup = () => {
      target.removeEventListener(successEvent, onSuccess);
      target.removeEventListener(errorEvent, onError);
    };
    target.addEventListener(successEvent, onSuccess, { once: true });
    target.addEventListener(errorEvent, onError, { once: true });
  });
}

async function seekVideo(video, time) {
  if (Math.abs(video.currentTime - time) < 0.001) return;
  video.currentTime = time;
  await waitForEvent(video, "seeked");
}

async function analyzeVideoFile(file) {
  setProgress("decode", "Loading video metadata and preparing browser frame sampling.");
  const url = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.muted = true;
  video.preload = "auto";
  video.src = url;
  video.load();
  await waitForEvent(video, "loadedmetadata");

  const duration = video.duration;
  if (!Number.isFinite(duration) || duration <= 0) {
    URL.revokeObjectURL(url);
    throw new Error("This browser could not determine a usable video duration.");
  }
  const width = video.videoWidth;
  const height = video.videoHeight;
  const times = sampleTimes(duration, VIDEO_SAMPLE_COUNT);
  const samples = [];
  let representativePayload = null;
  let representativeAnalysis = null;

  setProgress("analyze", `Sampling ${times.length} frames and running Rust/WASM luma analysis.`);
  for (let index = 0; index < times.length; index += 1) {
    const time = times[index];
    await seekVideo(video, time);
    const { canvas } = canvasFromSource(video, width, height, VIDEO_FRAME_MAX_PIXELS);
    const payload = payloadFromCanvas(canvas);
    const [summary, histogramResponse] = await Promise.all([
      runCore("image.core.summary", { image: payload }),
      runCore("image.core.lumaHistogram", { image: payload, bins: 32 }),
    ]);
    const histogram = histogramResponse.histogram ?? [];
    const sample = {
      timeSeconds: time,
      meanLuma: meanLumaFromHistogram(histogram),
      meanRgb: summary.meanRgb,
    };
    samples.push(sample);
    if (index === Math.floor(times.length / 2)) {
      representativePayload = payload;
      representativeAnalysis = { summary, histogram, histogramBins: histogramResponse.bins ?? 32 };
    }
  }

  if (!representativePayload || !representativeAnalysis) {
    throw new Error("No representative video frame was produced.");
  }
  const hashSize = Math.max(1, Math.min(8, representativePayload.width, representativePayload.height));
  const [maskTensor, perceptualHash] = await Promise.all([
    runCore("image.core.maskTensorSummary", { image: representativePayload, previewLimit: 32 }),
    runProcessing("image.processing.hash", { image: representativePayload, hashSize }),
  ]);
  representativeAnalysis.maskTensor = maskTensor;
  representativeAnalysis.perceptualHash = { ...perceptualHash, hash: String(perceptualHash.hash) };
  representativeAnalysis.meanLuma = meanLumaFromHistogram(representativeAnalysis.histogram);

  setPreviewUrl("video", url);
  return {
    schemaVersion: 1,
    media: {
      kind: "video",
      name: file.name,
      mimeType: file.type || "unknown",
      sizeBytes: file.size,
      width,
      height,
      durationSeconds: duration,
    },
    coverage: {
      mode: "bounded evenly-spaced frame sampling",
      analyzedWidth: representativePayload.width,
      analyzedHeight: representativePayload.height,
      analyzedPixelsPerFrame: representativePayload.width * representativePayload.height,
      sampleCount: samples.length,
      sampleTimesSeconds: times,
    },
    analysis: representativeAnalysis,
    timeline: samples,
  };
}

function setPreviewUrl(kind, url) {
  if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl);
  currentObjectUrl = url;
  const image = elements["preview-image"];
  const video = elements["preview-video"];
  if (kind === "image") {
    video.pause();
    video.removeAttribute("src");
    video.hidden = true;
    image.src = url;
    image.hidden = false;
  } else {
    image.removeAttribute("src");
    image.hidden = true;
    video.src = url;
    video.hidden = false;
  }
}

function generatedExample(kind) {
  const canvas = document.createElement("canvas");
  canvas.width = 720;
  canvas.height = 420;
  const context = canvas.getContext("2d", { alpha: false });

  if (kind === "checkerboard") {
    const size = 42;
    for (let y = 0; y < canvas.height; y += size) {
      for (let x = 0; x < canvas.width; x += size) {
        const light = ((x / size) + (y / size)) % 2 === 0;
        context.fillStyle = light ? "#f3efe4" : "#17201d";
        context.fillRect(x, y, size, size);
      }
    }
  } else {
    const gradient = context.createLinearGradient(0, 0, canvas.width, canvas.height);
    gradient.addColorStop(0, "#10233d");
    gradient.addColorStop(0.45, "#28b7a0");
    gradient.addColorStop(1, "#f3c55a");
    context.fillStyle = gradient;
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "rgba(255,255,255,.78)";
    context.beginPath();
    context.arc(530, 150, 85, 0, Math.PI * 2);
    context.fill();
    context.fillStyle = "rgba(12,30,28,.78)";
    context.fillRect(90, 235, 240, 95);
  }
  return canvas;
}

async function analyzeExample(kind) {
  setProgress("decode", "Generating the example entirely in this browser.");
  const canvas = generatedExample(kind);
  const dataUrl = canvas.toDataURL("image/png");
  const response = await fetch(dataUrl);
  const blob = await response.blob();
  const file = new File([blob], `${kind}-example.png`, { type: "image/png" });
  return analyzeImageFile(file);
}

function renderFindings(report) {
  const findings = buildFindings({
    histogram: report.analysis.histogram,
    width: report.media.width,
    height: report.media.height,
    analyzedWidth: report.coverage.analyzedWidth,
    analyzedHeight: report.coverage.analyzedHeight,
    mediaKind: report.media.kind,
  });
  if (report.media.kind === "video") {
    const values = report.timeline.map((sample) => sample.meanLuma);
    const range = Math.max(...values) - Math.min(...values);
    findings.push(
      range > 90
        ? "Sampled frames show large brightness changes across the video."
        : range < 25
          ? "Sampled frames are relatively stable in overall brightness."
          : "Sampled frames show moderate brightness variation over time.",
    );
  }
  elements.findings.replaceChildren(
    ...findings.map((text) => {
      const node = document.createElement("div");
      node.className = "finding";
      node.textContent = text;
      return node;
    }),
  );
}

function drawHistogram(canvas, histogram) {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(320, canvas.clientWidth);
  const height = Math.max(180, canvas.clientHeight);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);
  const max = Math.max(1, ...histogram);
  const gap = 2;
  const barWidth = width / histogram.length;
  context.fillStyle = "#0f766e";
  histogram.forEach((value, index) => {
    const barHeight = (value / max) * (height - 34);
    context.fillRect(index * barWidth + gap / 2, height - 22 - barHeight, Math.max(1, barWidth - gap), barHeight);
  });
  context.fillStyle = "#737a73";
  context.font = "12px system-ui";
  context.fillText("dark", 6, height - 5);
  const label = "bright";
  context.fillText(label, width - context.measureText(label).width - 6, height - 5);
}

function drawTimeline(canvas, samples) {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(320, canvas.clientWidth);
  const height = Math.max(180, canvas.clientHeight);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);
  const pad = 22;
  context.strokeStyle = "#d7ddd8";
  context.beginPath();
  context.moveTo(pad, height - pad);
  context.lineTo(width - pad, height - pad);
  context.stroke();
  context.strokeStyle = "#0f766e";
  context.lineWidth = 2.5;
  context.beginPath();
  samples.forEach((sample, index) => {
    const x = pad + (index / Math.max(1, samples.length - 1)) * (width - pad * 2);
    const y = height - pad - (sample.meanLuma / 255) * (height - pad * 2);
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.stroke();
  context.fillStyle = "#0f766e";
  samples.forEach((sample, index) => {
    const x = pad + (index / Math.max(1, samples.length - 1)) * (width - pad * 2);
    const y = height - pad - (sample.meanLuma / 255) * (height - pad * 2);
    context.beginPath();
    context.arc(x, y, 3.5, 0, Math.PI * 2);
    context.fill();
  });
}

function renderTechnical(report) {
  const metrics = [
    ["Runtime", "browser decode + image-analysis Rust/WASM"],
    ["Coverage", report.coverage.mode],
    ["Core operations", "image.core.summary · image.core.lumaHistogram · image.core.maskTensorSummary"],
    ["Processing operation", "image.processing.hash"],
    ["Analyzed buffer", `${report.coverage.analyzedWidth}×${report.coverage.analyzedHeight} rgb24`],
  ];
  if (report.media.kind === "video") {
    metrics.push(["Sample count", String(report.coverage.sampleCount)]);
  }
  elements["technical-metrics"].replaceChildren(
    ...metrics.map(([label, value]) => {
      const wrapper = document.createElement("div");
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = label;
      dd.textContent = value;
      wrapper.append(dt, dd);
      return wrapper;
    }),
  );
  elements["raw-json"].textContent = JSON.stringify(report, null, 2);
}

function renderReport(report) {
  currentReport = report;
  const meanRgb = report.analysis.summary.meanRgb;
  const hash = report.analysis.perceptualHash.hash;
  const meanLuma = report.analysis.meanLuma;
  const mediaKind = report.media.kind;
  const duration = mediaKind === "video" ? ` · ${report.media.durationSeconds.toFixed(1)} s` : "";

  elements["report-title"].textContent = report.media.name;
  elements["summary-kind"].textContent = mediaKind === "video" ? "Video" : "Image";
  elements["summary-dimensions"].textContent = `${report.media.width}×${report.media.height}`;
  elements["summary-analyzed"].textContent = mediaKind === "video"
    ? `${report.coverage.sampleCount} frames`
    : `${report.coverage.analyzedWidth}×${report.coverage.analyzedHeight}`;
  elements["summary-color"].textContent = rgbToHex(meanRgb);
  elements["summary-color"].style.color = rgbToHex(meanRgb);
  elements["summary-luma"].textContent = `${meanLuma.toFixed(1)} / 255`;
  elements["summary-hash"].textContent = String(hash);
  elements["file-meta"].textContent = `${report.media.mimeType} · ${formatBytes(report.media.sizeBytes)}${duration}`;
  elements["coverage-badge"].textContent = mediaKind === "video" ? `${report.coverage.sampleCount} sampled frames` : report.coverage.mode;
  elements["preview-note"].textContent = mediaKind === "video"
    ? "Playback uses the original local object URL; analysis uses bounded sampled frames."
    : report.coverage.scale === 1
      ? "The full decoded image fits inside the bounded analysis budget."
      : `The original preview is shown here; analysis used a ${report.coverage.analyzedWidth}×${report.coverage.analyzedHeight} downscaled buffer.`;

  renderFindings(report);
  drawHistogram(elements.histogram, report.analysis.histogram);
  const spread = histogramSpread(report.analysis.histogram);
  elements["histogram-readout"].replaceChildren(
    ...[
      `10th percentile ≈ ${histogramQuantile(report.analysis.histogram, .1).toFixed(0)}`,
      `median ≈ ${histogramQuantile(report.analysis.histogram, .5).toFixed(0)}`,
      `90th percentile ≈ ${histogramQuantile(report.analysis.histogram, .9).toFixed(0)}`,
      `spread ≈ ${spread.toFixed(0)}`,
    ].map((text) => {
      const span = document.createElement("span");
      span.textContent = text;
      return span;
    }),
  );

  const hasTimeline = mediaKind === "video" && report.timeline?.length;
  elements["timeline-section"].hidden = !hasTimeline;
  elements["timeline-nav"].hidden = !hasTimeline;
  if (hasTimeline) {
    elements["timeline-badge"].textContent = `${report.timeline.length} evenly spaced frames`;
    drawTimeline(elements.timeline, report.timeline);
  }

  renderTechnical(report);
  elements["loading-panel"].hidden = true;
  elements["input-panel"].hidden = true;
  elements.report.hidden = false;
  requestAnimationFrame(() => elements["report-title"].focus());
}

async function analyzeFile(file) {
  if (!file) return;
  setError();
  elements.report.hidden = true;
  elements["loading-panel"].hidden = false;
  elements["loading-title"].textContent = `Analyzing ${file.name}…`;
  try {
    const kind = file.type.startsWith("video/") ? "video" : file.type.startsWith("image/") ? "image" : null;
    if (!kind) throw new Error("Choose a browser-decodable image or video file.");
    const report = kind === "video" ? await analyzeVideoFile(file) : await analyzeImageFile(file);
    setProgress("report", "Preparing the visual report and technical export.");
    renderReport(report);
  } catch (error) {
    elements["loading-panel"].hidden = true;
    elements["input-panel"].hidden = false;
    setError(error instanceof Error ? error.message : String(error));
  }
}

async function runExample(kind) {
  setError();
  elements.report.hidden = true;
  elements["loading-panel"].hidden = false;
  elements["loading-title"].textContent = `Analyzing ${kind} example…`;
  try {
    const report = await analyzeExample(kind);
    setProgress("report", "Preparing the visual report and technical export.");
    renderReport(report);
  } catch (error) {
    elements["loading-panel"].hidden = true;
    elements["input-panel"].hidden = false;
    setError(error instanceof Error ? error.message : String(error));
  }
}

function reset() {
  elements.report.hidden = true;
  elements["loading-panel"].hidden = true;
  elements["input-panel"].hidden = false;
  elements["file-input"].value = "";
  setError();
  currentReport = null;
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function exportReport() {
  if (!currentReport) return;
  const blob = new Blob([JSON.stringify(currentReport, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  const base = currentReport.media.name.replace(/\.[^.]+$/, "") || "visual-analysis";
  anchor.href = url;
  anchor.download = `${base}.visual-analysis.json`;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

elements["choose-file"].addEventListener("click", () => elements["file-input"].click());
elements["file-input"].addEventListener("change", () => void analyzeFile(elements["file-input"].files?.[0]));
elements["choose-another"].addEventListener("click", reset);
elements["export-json"].addEventListener("click", exportReport);
for (const card of document.querySelectorAll("[data-example]")) {
  card.addEventListener("click", () => void runExample(card.dataset.example));
}
for (const eventName of ["dragenter", "dragover"]) {
  elements["drop-zone"].addEventListener(eventName, (event) => {
    event.preventDefault();
    elements["drop-zone"].classList.add("is-dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  elements["drop-zone"].addEventListener(eventName, (event) => {
    event.preventDefault();
    elements["drop-zone"].classList.remove("is-dragging");
  });
}
elements["drop-zone"].addEventListener("drop", (event) => void analyzeFile(event.dataTransfer?.files?.[0]));
window.addEventListener("resize", () => {
  if (!currentReport || elements.report.hidden) return;
  drawHistogram(elements.histogram, currentReport.analysis.histogram);
  if (currentReport.timeline?.length) drawTimeline(elements.timeline, currentReport.timeline);
});
