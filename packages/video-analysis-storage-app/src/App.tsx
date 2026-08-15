import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/visual-app-ui/package-surface";
import * as wasm from "@moritzbrantner/video-analysis-storage-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "video-analysis-storage",
  title: "Video Analysis Storage",
  description: "Dataset persistence for video-analysis.",
  domain: "video",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/video-analysis-storage",
    standaloneRoute: "",
  },
  defaultOperation: "video.storage.manifestPlan",
  featuredOperations: ["video.storage.manifestPlan", "video.storage.jsonlPreview", "describe"],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run the main package workflow.",
      operations: ["video.storage.manifestPlan"],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect inputs, plans, metadata, and diagnostic helpers.",
      operations: ["describe", "video.storage.jsonlPreview"],
    },
  ],
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
