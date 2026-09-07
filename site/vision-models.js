const TRANSFORMERS_MODULE_URL =
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.0";

export const SAM_MODEL_ID = "Xenova/slimsam-77-uniform";
export const OPEN_VOCAB_MODEL_ID = "Xenova/owlvit-base-patch32";

let transformersPromise;
let samRuntimePromise;
let openVocabularyDetectorPromise;

function loadTransformers() {
  transformersPromise ??= import(TRANSFORMERS_MODULE_URL);
  return transformersPromise;
}

export function browserVisionCapabilities() {
  return {
    webgpu: typeof navigator !== "undefined" && Boolean(navigator.gpu),
    samModel: SAM_MODEL_ID,
    openVocabularyModel: OPEN_VOCAB_MODEL_ID,
    transformersModule: TRANSFORMERS_MODULE_URL,
  };
}

async function loadSamRuntime() {
  const capabilities = browserVisionCapabilities();
  if (!capabilities.webgpu) {
    throw new Error(
      "Interactive SAM segmentation requires WebGPU in this browser. Native DETR remains available through the CLI/server runtime.",
    );
  }

  samRuntimePromise ??= loadTransformers().then(async ({ SamModel, AutoProcessor, RawImage, Tensor }) => {
    const [model, processor] = await Promise.all([
      SamModel.from_pretrained(SAM_MODEL_ID, { dtype: "fp16", device: "webgpu" }),
      AutoProcessor.from_pretrained(SAM_MODEL_ID),
    ]);
    return { model, processor, RawImage, Tensor };
  });
  return samRuntimePromise;
}

export async function prepareSamImage(imageUrl) {
  if (!imageUrl) throw new Error("An image URL is required for SAM segmentation.");
  const runtime = await loadSamRuntime();
  const image = await runtime.RawImage.fromURL(imageUrl);
  const processed = await runtime.processor(image);
  const embeddings = await runtime.model.get_image_embeddings(processed);
  return { ...runtime, image, processed, embeddings };
}

function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value) || 0));
}

function regionFromMask(data, width, height) {
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  let activePixels = 0;

  for (let index = 0; index < data.length; index += 1) {
    if (data[index] === 0) continue;
    activePixels += 1;
    const x = index % width;
    const y = Math.floor(index / width);
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }

  return {
    activePixels,
    region:
      activePixels === 0
        ? null
        : {
            x: minX,
            y: minY,
            width: maxX - minX + 1,
            height: maxY - minY + 1,
          },
  };
}

export async function segmentSamPoint(session, point) {
  if (!session?.embeddings || !session?.processed) {
    throw new Error("SAM image embeddings are not ready.");
  }

  const normalizedX = clamp01(point?.x);
  const normalizedY = clamp01(point?.y);
  const label = point?.label === 0 ? 0 : 1;
  const reshaped = session.processed.reshaped_input_sizes[0];
  const inputPoints = new session.Tensor(
    "float32",
    [normalizedX * reshaped[1], normalizedY * reshaped[0]],
    [1, 1, 1, 2],
  );
  const inputLabels = new session.Tensor("int64", [BigInt(label)], [1, 1, 1]);

  const { pred_masks: predMasks, iou_scores: iouScores } = await session.model({
    ...session.embeddings,
    input_points: inputPoints,
    input_labels: inputLabels,
  });
  const masks = await session.processor.post_process_masks(
    predMasks,
    session.processed.original_sizes,
    session.processed.reshaped_input_sizes,
  );
  const mask = session.RawImage.fromTensor(masks[0][0]);
  const scores = Array.from(iouScores.data, Number);

  let bestIndex = 0;
  for (let index = 1; index < scores.length; index += 1) {
    if (scores[index] > scores[bestIndex]) bestIndex = index;
  }

  const data = new Uint8Array(mask.width * mask.height);
  const maskCount = scores.length;
  for (let index = 0; index < data.length; index += 1) {
    data[index] = mask.data[maskCount * index + bestIndex] === 1 ? 255 : 0;
  }
  const summary = regionFromMask(data, mask.width, mask.height);

  return {
    backend: "transformers.js-sam-webgpu",
    modelId: SAM_MODEL_ID,
    score: scores[bestIndex] ?? null,
    width: mask.width,
    height: mask.height,
    data,
    activePixels: summary.activePixels,
    region: summary.region,
    prompt: { x: normalizedX, y: normalizedY, label },
  };
}

export async function detectOpenVocabulary(imageUrl, labels, options = {}) {
  if (!imageUrl) throw new Error("An image URL is required for object detection.");
  const candidateLabels = Array.from(
    new Set((labels ?? []).map((label) => String(label).trim()).filter(Boolean)),
  );
  if (candidateLabels.length === 0) {
    throw new Error("Enter at least one concept to detect.");
  }

  const threshold = Number.isFinite(options.threshold) ? options.threshold : 0.08;
  const topK = Number.isInteger(options.topK) ? Math.max(1, options.topK) : 20;
  const { pipeline } = await loadTransformers();
  openVocabularyDetectorPromise ??= pipeline(
    "zero-shot-object-detection",
    OPEN_VOCAB_MODEL_ID,
    browserVisionCapabilities().webgpu ? { device: "webgpu" } : {},
  );
  const detector = await openVocabularyDetectorPromise;
  const detections = await detector(imageUrl, candidateLabels, {
    threshold,
    top_k: topK,
  });

  return detections.map((detection) => {
    const xmin = Math.max(0, Math.round(detection.box?.xmin ?? 0));
    const ymin = Math.max(0, Math.round(detection.box?.ymin ?? 0));
    const xmax = Math.max(xmin, Math.round(detection.box?.xmax ?? xmin));
    const ymax = Math.max(ymin, Math.round(detection.box?.ymax ?? ymin));
    return {
      label: String(detection.label ?? "object"),
      score: Number(detection.score ?? 0),
      region: {
        x: xmin,
        y: ymin,
        width: xmax - xmin,
        height: ymax - ymin,
      },
      attributes: {
        backend: "transformers.js-zero-shot-object-detection",
        modelId: OPEN_VOCAB_MODEL_ID,
        promptKind: "text",
      },
    };
  });
}
