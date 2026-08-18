import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/visual-app-ui/package-surface";
import * as wasm from "@moritzbrantner/video-analysis-features-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "video-analysis-features",
  title: "Video Analysis Features",
  description: "Feature extraction over retained video-analysis datasets.",
  domain: "video",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/video-analysis-features",
    standaloneRoute: "",
  },
  defaultOperation: "video.features.extract",
  featuredOperations: ["video.features.extract", "video.features.aggregate", "video.features.timelineSummary", "describe"],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run the main package workflow.",
      operations: ["video.features.extract", "video.features.aggregate"],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect inputs, plans, metadata, and diagnostic helpers.",
      operations: ["describe", "video.features.timelineSummary"],
    },
  ],
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
