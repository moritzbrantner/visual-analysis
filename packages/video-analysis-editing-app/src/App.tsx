import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/visual-app-ui/package-surface";
import * as wasm from "@moritzbrantner/video-analysis-editing-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "video-analysis-editing",
  title: "Video Analysis Editing",
  description: "CPU video frame editing primitives for video-analysis.",
  domain: "video",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/video-analysis-editing",
    standaloneRoute: "",
  },
  defaultOperation: "video.editing.cutPlan",
  featuredOperations: [
    "video.editing.cutPlan",
    "video.editing.frameApply",
    "video.editing.concatPlan",
    "video.editing.subtitlePlan",
    "describe",
  ],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run the main package workflow.",
      operations: ["video.editing.cutPlan", "video.editing.frameApply"],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect inputs, plans, metadata, and diagnostic helpers.",
      operations: ["describe", "video.editing.concatPlan", "video.editing.subtitlePlan"],
    },
  ],
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
