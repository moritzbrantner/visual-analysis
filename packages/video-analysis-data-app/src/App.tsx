import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/visual-app-ui/package-surface";
import * as wasm from "@moritzbrantner/video-analysis-data-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "video-analysis-data",
  title: "Video Analysis Data",
  description: "Normalized stream records and online aggregation for video-analysis.",
  domain: "video",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/video-analysis-data",
    standaloneRoute: "",
  },
  defaultOperation: "video.data.recordSummary",
  featuredOperations: ["video.data.recordSummary", "video.data.eventTimeline", "video.data.joinPlan", "describe"],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run the main package workflow.",
      operations: ["video.data.recordSummary", "video.data.eventTimeline"],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect inputs, plans, metadata, and diagnostic helpers.",
      operations: ["describe", "video.data.joinPlan"],
    },
  ],
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
