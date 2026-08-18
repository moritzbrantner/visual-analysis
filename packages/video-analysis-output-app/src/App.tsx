import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/visual-app-ui/package-surface";
import * as wasm from "@moritzbrantner/video-analysis-output-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "video-analysis-output",
  title: "Video Analysis Output",
  description: "CSV, HTML, JSON, EDL, FCP, OTIO, and qpfile report helpers for video-analysis.",
  domain: "video",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/video-analysis-output",
    standaloneRoute: "",
  },
  defaultOperation: "video.output.reportSummary",
  featuredOperations: [
    "video.output.reportSummary",
    "video.output.csvPlan",
    "video.output.htmlPlan",
    "video.output.editListPlan",
    "describe",
  ],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run the main package workflow.",
      operations: ["video.output.reportSummary"],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect inputs, plans, metadata, and diagnostic helpers.",
      operations: ["describe", "video.output.csvPlan", "video.output.htmlPlan", "video.output.editListPlan"],
    },
  ],
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
