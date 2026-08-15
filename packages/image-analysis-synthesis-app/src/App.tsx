import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/visual-app-ui/package-surface";
import * as wasm from "@moritzbrantner/image-analysis-synthesis-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "image-analysis-synthesis",
  title: "Image Analysis Synthesis",
  description: "Deterministic image synthesis helpers for video-analysis.",
  domain: "image",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/image-analysis-synthesis",
    standaloneRoute: "",
  },
  defaultOperation: "image.synthesis.solid",
  featuredOperations: ["image.synthesis.solid", "image.synthesis.gradient", "image.synthesis.histogram", "describe"],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run deterministic image synthesis workflows.",
      operations: ["image.synthesis.solid", "image.synthesis.gradient", "image.synthesis.histogram"],
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
