import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/visual-app-ui/package-surface";
import * as wasm from "@moritzbrantner/video-analysis-detectors-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "video-analysis-detectors",
  title: "Video Analysis Detectors",
  description: "Scene detection algorithms and detector adapters for video-analysis.",
  domain: "video",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/video-analysis-detectors",
    standaloneRoute: "",
  },
  defaultOperation: "video.detectors.registry",
  featuredOperations: ["video.detectors.registry", "video.detectors.flashFilter", "video.detectors.compositePlan", "describe"],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run the main package workflow.",
      operations: ["video.detectors.registry", "video.detectors.flashFilter"],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect inputs, plans, metadata, and diagnostic helpers.",
      operations: ["describe", "video.detectors.compositePlan"],
    },
  ],
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
