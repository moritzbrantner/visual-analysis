import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/visual-app-ui/package-surface";
import * as wasm from "@moritzbrantner/image-analysis-segmentation-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "image-analysis-segmentation",
  title: "Image Analysis Segmentation",
  description: "Image segmentation prompts, masks, and segment contracts for video-analysis.",
  domain: "image",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/image-analysis-segmentation",
    standaloneRoute: "",
  },
  defaultOperation: "image.segmentation.maskSummary",
  featuredOperations: [
    "image.segmentation.maskSummary",
    "image.segmentation.promptSummary",
    "image.segmentation.model",
    "describe",
  ],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Summarize imported segmentation masks.",
      operations: ["image.segmentation.maskSummary"],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect segmentation model metadata, prompts, and package metadata.",
      operations: ["image.segmentation.model", "image.segmentation.promptSummary", "describe"],
    },
  ],
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
