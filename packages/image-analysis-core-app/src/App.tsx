import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/visual-app-ui/package-surface";
import * as wasm from "@moritzbrantner/image-analysis-core-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "image-analysis-core",
  title: "Image Analysis Core",
  description: "Shared image views, pixel formats, and image statistics for video-analysis.",
  domain: "image",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/image-analysis-core",
    standaloneRoute: "",
  },
  defaultOperation: "image.core.summary",
  featuredOperations: ["image.core.summary", "image.core.lumaHistogram", "image.core.maskTensorSummary", "describe"],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run image summary and tensor analysis workflows.",
      operations: ["image.core.summary", "image.core.lumaHistogram", "image.core.maskTensorSummary"],
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
