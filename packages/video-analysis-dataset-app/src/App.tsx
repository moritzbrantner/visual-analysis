import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/visual-app-ui/package-surface";
import * as wasm from "@moritzbrantner/video-analysis-dataset-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "video-analysis-dataset",
  title: "Video Analysis Dataset",
  description: "Serializable retained analysis records for video-analysis.",
  domain: "video",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/video-analysis-dataset",
    standaloneRoute: "",
  },
  defaultOperation: "video.dataset.summary",
  featuredOperations: ["video.dataset.summary", "video.dataset.recordsByKind", "describe"],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run the main package workflow.",
      operations: ["video.dataset.summary", "video.dataset.recordsByKind"],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect inputs, plans, metadata, and diagnostic helpers.",
      operations: ["describe"],
    },
  ],
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
