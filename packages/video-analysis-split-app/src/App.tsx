import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/visual-app-ui/package-surface";
import * as wasm from "@moritzbrantner/video-analysis-split-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "video-analysis-split",
  title: "Video Analysis Split",
  description: "FFmpeg-backed scene splitting utilities for video-analysis.",
  domain: "video",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/video-analysis-split",
    standaloneRoute: "",
  },
  defaultOperation: "video.split.scenePlan",
  featuredOperations: ["video.split.scenePlan", "video.split.segmentManifest", "video.split.namingPlan", "describe"],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run the main package workflow.",
      operations: ["video.split.scenePlan"],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect inputs, plans, metadata, and diagnostic helpers.",
      operations: ["describe", "video.split.segmentManifest", "video.split.namingPlan"],
    },
  ],
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
