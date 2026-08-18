import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/visual-app-ui/package-surface";
import * as wasm from "@moritzbrantner/video-analysis-recognition-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "video-analysis-recognition",
  title: "Video Analysis Recognition",
  description: "Reference-embedding recognition helpers for video-analysis.",
  domain: "video",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/video-analysis-recognition",
    standaloneRoute: "",
  },
  defaultOperation: "video.recognition.labelPlan",
  featuredOperations: ["video.recognition.labelPlan", "video.recognition.confidence", "video.recognition.trackSummary", "describe"],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run the main package workflow.",
      operations: ["video.recognition.labelPlan", "video.recognition.confidence"],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect inputs, plans, metadata, and diagnostic helpers.",
      operations: ["describe", "video.recognition.trackSummary"],
    },
  ],
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
