import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/visual-app-ui/package-surface";
import * as wasm from "@moritzbrantner/image-analysis-processing-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "image-analysis-processing",
  title: "Image Analysis Processing",
  description: "CPU image processing primitives for video-analysis.",
  domain: "image",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/image-analysis-processing",
    standaloneRoute: "",
  },
  defaultOperation: "image.processing.apply",
  featuredOperations: [
    "image.processing.apply",
    "image.processing.pipeline",
    "image.processing.composite",
    "image.processing.hash",
    "describe",
  ],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run deterministic CPU image processing workflows.",
      operations: [
        "image.processing.apply",
        "image.processing.pipeline",
        "image.processing.composite",
        "image.processing.hash",
      ],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect package metadata and operation support.",
      operations: ["describe"],
    },
  ],
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
