import {
  OPEN_VOCAB_MODEL_ID,
  SAM_MODEL_ID,
  browserVisionCapabilities,
  detectOpenVocabulary,
  prepareSamImage,
  segmentSamPoint,
} from "./vision-models.js";

function ensureVisionUi() {
  if (!document.querySelector('link[href="./vision.css"]')) {
    const stylesheet = document.createElement("link");
    stylesheet.rel = "stylesheet";
    stylesheet.href = "./vision.css";
    document.head.append(stylesheet);
  }

  const previewSection = document.getElementById("preview-section");
  const previewFrame = previewSection?.querySelector(".preview-frame");
  if (!previewSection || !previewFrame) {
    throw new Error("Visual Inspector preview surface is unavailable.");
  }

  let overlay = document.getElementById("vision-overlay");
  if (!overlay) {
    overlay = document.createElement("canvas");
    overlay.id = "vision-overlay";
    overlay.hidden = true;
    overlay.setAttribute("aria-label", "Learned vision overlay");
    previewFrame.append(overlay);
  }

  let section = document.getElementById("learned-vision-section");
  if (!section) {
    section = document.createElement("section");
    section.id = "learned-vision-section";
    section.className = "panel report-section";
    section.hidden = true;
    section.innerHTML = `
      <div class="section-heading">
        <div>
          <div class="eyebrow">opt-in learned vision</div>
          <h2>Object detection &amp; segmentation</h2>
        </div>
        <span id="vision-runtime-capability" class="badge">checking runtime</span>
      </div>
      <div class="vision-controls">
        <label class="vision-field">
          <span>Concepts to detect</span>
          <input id="vision-concepts" value="person, car, dog, cat" autocomplete="off" />
        </label>
        <button id="detect-concepts" class="button button-primary" type="button">Detect concepts</button>
        <button id="prepare-sam" class="button button-secondary" type="button">Prepare SAM</button>
        <button id="clear-vision-overlay" class="button button-secondary" type="button">Clear overlay</button>
      </div>
      <p id="learned-vision-status" class="vision-notice" aria-live="polite">
        Learned vision is available for still images.
      </p>
      <p class="vision-notice">
        Model weights are fetched from Hugging Face only after you choose an action. The selected image remains a local browser blob URL and is not uploaded by this inspector. Native DETR remains the repository's accepted baseline; these browser adapters are explicitly separate execution backends over the same detection/segmentation concepts.
      </p>
      <div id="vision-results" class="vision-results"></div>
    `;
    previewSection.insertAdjacentElement("afterend", section);
  }

  const reportNav = document.querySelector(".report-nav");
  if (reportNav && !reportNav.querySelector('a[href="#learned-vision-section"]')) {
    const link = document.createElement("a");
    link.href = "#learned-vision-section";
    link.textContent = "Vision";
    const capabilities = reportNav.querySelector('a[href="#capabilities"]');
    reportNav.insertBefore(link, capabilities ?? null);
  }

  for (const card of document.querySelectorAll(".capability-card")) {
    const heading = card.querySelector("h3");
    if (heading?.textContent !== "Detection & segmentation") continue;
    const badge = card.querySelector(".capability-status");
    if (badge) {
      badge.textContent = "Native + opt-in browser";
      badge.classList.add("capability-status-live");
    }
    const description = card.querySelector("p");
    if (description) {
      description.textContent =
        "Native DETR/YuNet remain accepted model backends. The inspector can additionally run text-conditioned object detection and interactive SAM locally after explicit opt-in.";
    }
  }

  return { overlay, section };
}

const ui = ensureVisionUi();
const previewImage = document.getElementById("preview-image");
const section = ui.section;
const status = document.getElementById("learned-vision-status");
const concepts = document.getElementById("vision-concepts");
const detectButton = document.getElementById("detect-concepts");
const samButton = document.getElementById("prepare-sam");
const clearButton = document.getElementById("clear-vision-overlay");
const overlay = ui.overlay;
const results = document.getElementById("vision-results");
const capability = document.getElementById("vision-runtime-capability");

let imageUrl = "";
let samSession = null;
let samReady = false;
let running = false;

function setStatus(message, kind = "") {
  status.textContent = message;
  status.dataset.kind = kind;
}

function setBusy(value) {
  running = value;
  detectButton.disabled = value || !imageUrl;
  samButton.disabled = value || !imageUrl || !browserVisionCapabilities().webgpu;
  clearButton.disabled = value || !imageUrl;
}

function clearResults() {
  results.replaceChildren();
}

function clearOverlay() {
  const context = overlay.getContext("2d");
  context.clearRect(0, 0, overlay.width, overlay.height);
  clearResults();
}

function configureOverlay(width, height) {
  overlay.width = Math.max(1, Math.round(width));
  overlay.height = Math.max(1, Math.round(height));
  overlay.hidden = false;
}

function currentPreviewUrl() {
  return !previewImage.hidden && previewImage.currentSrc
    ? previewImage.currentSrc
    : !previewImage.hidden
      ? previewImage.src
      : "";
}

function resetForImage() {
  const nextUrl = currentPreviewUrl();
  if (nextUrl === imageUrl) return;
  imageUrl = nextUrl;
  samSession = null;
  samReady = false;
  clearOverlay();

  section.hidden = !imageUrl;
  overlay.hidden = !imageUrl;
  overlay.classList.remove("is-sam-ready");
  if (!imageUrl) {
    setStatus("Learned vision is available for still images.");
  } else {
    const capabilities = browserVisionCapabilities();
    setStatus(
      capabilities.webgpu
        ? "Ready. Model weights download only after you choose a learned-vision action."
        : "Text-conditioned detection can use the browser CPU/WASM backend; interactive SAM requires WebGPU.",
    );
    configureOverlay(previewImage.naturalWidth || 1, previewImage.naturalHeight || 1);
  }
  setBusy(false);
}

function appendResult(title, detail) {
  const item = document.createElement("div");
  item.className = "vision-result";
  const strong = document.createElement("strong");
  strong.textContent = title;
  const span = document.createElement("span");
  span.textContent = detail;
  item.append(strong, span);
  results.append(item);
}

function drawDetections(detections) {
  configureOverlay(previewImage.naturalWidth, previewImage.naturalHeight);
  const context = overlay.getContext("2d");
  context.clearRect(0, 0, overlay.width, overlay.height);
  context.lineWidth = Math.max(2, Math.round(Math.min(overlay.width, overlay.height) / 220));
  context.font = `${Math.max(13, Math.round(Math.min(overlay.width, overlay.height) / 35))}px system-ui`;
  context.textBaseline = "bottom";

  for (const detection of detections) {
    const { x, y, width, height } = detection.region;
    context.strokeRect(x, y, width, height);
    const label = `${detection.label} ${(detection.score * 100).toFixed(0)}%`;
    const metrics = context.measureText(label);
    const labelHeight = Math.max(18, Number.parseInt(context.font, 10) + 6);
    const labelY = Math.max(labelHeight, y);
    context.fillRect(x, labelY - labelHeight, metrics.width + 10, labelHeight);
    context.save();
    context.globalCompositeOperation = "difference";
    context.fillText(label, x + 5, labelY - 3);
    context.restore();
  }
}

function drawMask(segment) {
  configureOverlay(segment.width, segment.height);
  const context = overlay.getContext("2d");
  const imageData = context.createImageData(segment.width, segment.height);
  for (let index = 0; index < segment.data.length; index += 1) {
    if (segment.data[index] === 0) continue;
    const offset = index * 4;
    imageData.data[offset] = 255;
    imageData.data[offset + 1] = 255;
    imageData.data[offset + 2] = 255;
    imageData.data[offset + 3] = 112;
  }
  context.putImageData(imageData, 0, 0);
}

function parseConcepts() {
  return concepts.value
    .split(",")
    .map((label) => label.trim())
    .filter(Boolean);
}

async function runConceptDetection() {
  if (!imageUrl || running) return;
  setBusy(true);
  clearResults();
  setStatus(`Loading ${OPEN_VOCAB_MODEL_ID} and detecting requested concepts locally…`);
  try {
    const detections = await detectOpenVocabulary(imageUrl, parseConcepts(), {
      threshold: 0.08,
      topK: 20,
    });
    drawDetections(detections);
    if (detections.length === 0) {
      appendResult("No matches", "No requested concept cleared the confidence threshold.");
    } else {
      for (const detection of detections) {
        const { x, y, width, height } = detection.region;
        appendResult(
          detection.label,
          `${(detection.score * 100).toFixed(1)}% · x ${x}, y ${y}, ${width}×${height}`,
        );
      }
    }
    setStatus(
      `Open-vocabulary detection completed locally with ${detections.length} result${detections.length === 1 ? "" : "s"}.`,
      "success",
    );
  } catch (error) {
    setStatus(error instanceof Error ? error.message : String(error), "error");
  } finally {
    setBusy(false);
  }
}

async function prepareSam() {
  if (!imageUrl || running) return;
  setBusy(true);
  clearResults();
  setStatus(`Loading ${SAM_MODEL_ID} and computing the image embedding locally…`);
  try {
    samSession = await prepareSamImage(imageUrl);
    samReady = true;
    overlay.classList.add("is-sam-ready");
    setStatus(
      "SAM is ready. Left-click the image for a foreground point; right-click for a background point.",
      "success",
    );
    appendResult("SAM ready", "The image embedding stays in browser memory; each point reuses it.");
  } catch (error) {
    samSession = null;
    samReady = false;
    overlay.classList.remove("is-sam-ready");
    setStatus(error instanceof Error ? error.message : String(error), "error");
  } finally {
    setBusy(false);
  }
}

async function segmentAtPointer(event) {
  if (!samReady || !samSession || running) return;
  if (event.button !== 0 && event.button !== 2) return;
  event.preventDefault();

  const bounds = overlay.getBoundingClientRect();
  if (bounds.width <= 0 || bounds.height <= 0) return;
  const point = {
    x: (event.clientX - bounds.left) / bounds.width,
    y: (event.clientY - bounds.top) / bounds.height,
    label: event.button === 2 ? 0 : 1,
  };

  setBusy(true);
  setStatus("Decoding a SAM mask from the cached image embedding…");
  try {
    const segment = await segmentSamPoint(samSession, point);
    drawMask(segment);
    clearResults();
    appendResult(
      "SAM mask",
      `${segment.score == null ? "unscored" : `${(segment.score * 100).toFixed(1)}%`} · ${segment.activePixels.toLocaleString()} pixels · ${segment.region ? `${segment.region.width}×${segment.region.height} bounds` : "empty mask"}`,
    );
    setStatus("SAM mask decoded locally. Click another point to refine the selection.", "success");
  } catch (error) {
    setStatus(error instanceof Error ? error.message : String(error), "error");
  } finally {
    setBusy(false);
  }
}

const capabilities = browserVisionCapabilities();
capability.textContent = capabilities.webgpu
  ? "WebGPU detected · SAM + open-vocabulary detection available"
  : "No WebGPU · SAM disabled; open-vocabulary detection may fall back to browser WASM";

detectButton.addEventListener("click", runConceptDetection);
samButton.addEventListener("click", prepareSam);
clearButton.addEventListener("click", () => {
  clearOverlay();
  setStatus(samReady ? "SAM is still ready; click the image to segment again." : "Overlay cleared.");
});
overlay.addEventListener("mousedown", segmentAtPointer);
overlay.addEventListener("contextmenu", (event) => event.preventDefault());
previewImage.addEventListener("load", () => {
  resetForImage();
  if (imageUrl && previewImage.naturalWidth > 0 && previewImage.naturalHeight > 0) {
    configureOverlay(previewImage.naturalWidth, previewImage.naturalHeight);
  }
});

const observer = new MutationObserver(() => resetForImage());
observer.observe(previewImage, { attributes: true, attributeFilter: ["src", "hidden"] });
resetForImage();
