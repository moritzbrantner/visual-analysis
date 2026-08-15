import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/visual-app-ui/package-surface";
import * as wasm from "@moritzbrantner/video-analysis-tracking-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "video-analysis-tracking",
  title: "Video Analysis Tracking",
  description: "IoU-based object tracking for video-analysis.",
  domain: "video",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/video-analysis-tracking",
    standaloneRoute: "",
  },
  defaultOperation: "video.tracking.trackSummary",
  featuredOperations: ["video.tracking.trackSummary", "video.tracking.smoothPath", "video.tracking.assignmentPlan", "describe"],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run the main package workflow.",
      operations: ["video.tracking.trackSummary", "video.tracking.smoothPath"],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect inputs, plans, metadata, and diagnostic helpers.",
      operations: ["describe", "video.tracking.assignmentPlan"],
    },
  ],
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
