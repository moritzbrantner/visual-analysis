import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/visual-app-ui/package-surface";
import * as wasm from "@moritzbrantner/image-analysis-ocr-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "image-analysis-ocr",
  title: "Image Analysis OCR",
  description: "OCR model presets, rich text outputs, and image/video backend contracts for video-analysis.",
  domain: "image",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/image-analysis-ocr",
    standaloneRoute: "",
  },
  defaultOperation: "image.ocr.recognize",
  featuredOperations: ["image.ocr.recognize", "image.ocr.toTextDocument", "image.ocr.models", "image.ocr.presets", "describe"],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Recognize OCR text and convert OCR output into text document contracts.",
      operations: ["image.ocr.recognize", "image.ocr.toTextDocument"],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect OCR models, presets, request options, imported document summaries, and package metadata.",
      operations: ["image.ocr.models", "image.ocr.presets", "image.ocr.requestSummary", "image.ocr.documentSummary", "describe"],
    },
  ],
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
