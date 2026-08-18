import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/visual-app-ui/package-surface";
import * as wasm from "@moritzbrantner/video-analysis-synthesis-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "video-analysis-synthesis",
  title: "Video Analysis Synthesis",
  description: "Deterministic storyboard and frame synthesis for video-analysis.",
  domain: "video",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/video-analysis-synthesis",
    standaloneRoute: "",
  },
  defaultOperation: "video.synthesis.framePlan",
  featuredOperations: ["video.synthesis.framePlan", "video.synthesis.overlayPlan", "video.synthesis.renderSummary", "describe"],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run the main package workflow.",
      operations: ["video.synthesis.framePlan"],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect inputs, plans, metadata, and diagnostic helpers.",
      operations: ["describe", "video.synthesis.overlayPlan", "video.synthesis.renderSummary"],
    },
  ],
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
