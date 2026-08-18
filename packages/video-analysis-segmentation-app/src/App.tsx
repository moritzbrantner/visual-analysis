import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/visual-app-ui/package-surface";
import * as wasm from "@moritzbrantner/video-analysis-segmentation-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "video-analysis-segmentation",
  title: "Video Analysis Segmentation",
  description: "Video segmentation primitives and SAM 2 defaults for video-analysis.",
  domain: "video",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/video-analysis-segmentation",
    standaloneRoute: "",
  },
  defaultOperation: "video.segmentation.maskSummary",
  featuredOperations: ["video.segmentation.maskSummary", "video.segmentation.promptPlan", "video.segmentation.trackPlan", "describe"],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run the main package workflow.",
      operations: ["video.segmentation.maskSummary"],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect inputs, plans, metadata, and diagnostic helpers.",
      operations: ["describe", "video.segmentation.promptPlan", "video.segmentation.trackPlan"],
    },
  ],
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
