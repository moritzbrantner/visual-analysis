import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/visual-app-ui/package-surface";
import * as wasm from "@moritzbrantner/video-analysis-ingest-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "video-analysis-ingest",
  title: "Video Analysis Ingest",
  description: "Media ingest traits and source adapters for video-analysis.",
  domain: "video",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/video-analysis-ingest",
    standaloneRoute: "",
  },
  defaultOperation: "video.ingest.sourcePlan",
  featuredOperations: ["video.ingest.sourcePlan", "video.ingest.manifest", "video.ingest.validate", "describe"],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run the main package workflow.",
      operations: ["video.ingest.sourcePlan"],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect inputs, plans, metadata, and diagnostic helpers.",
      operations: ["describe", "video.ingest.manifest", "video.ingest.validate"],
    },
  ],
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
