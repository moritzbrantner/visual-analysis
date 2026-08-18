import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/visual-app-ui/package-surface";
import * as wasm from "@moritzbrantner/image-analysis-detection-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "image-analysis-detection",
  title: "Image Analysis Detection",
  description: "Canonical image detection types and mask-proposal adapters for video-analysis.",
  domain: "image",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/image-analysis-detection",
    standaloneRoute: "",
  },
  defaultOperation: "image.detection.colorBlob",
  featuredOperations: ["image.detection.colorBlob", "image.detection.models", "image.detection.boxSummary", "describe"],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run deterministic image detection workflows.",
      operations: ["image.detection.colorBlob"],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect detection catalogs, imported boxes, and package metadata.",
      operations: ["image.detection.models", "image.detection.boxSummary", "describe"],
    },
  ],
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
