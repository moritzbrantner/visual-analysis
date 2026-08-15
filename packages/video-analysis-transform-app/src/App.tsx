import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/visual-app-ui/package-surface";
import * as wasm from "@moritzbrantner/video-analysis-transform-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "video-analysis-transform",
  title: "Video Analysis Transform",
  description: "Filtering, joins, grouping, and resampling for video-analysis datasets.",
  domain: "video",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/video-analysis-transform",
    standaloneRoute: "",
  },
  defaultOperation: "video.transform.filter",
  featuredOperations: [
    "video.transform.filter",
    "video.transform.window",
    "video.transform.groupScenes",
    "video.transform.resampleFeatures",
    "describe",
  ],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run the main package workflow.",
      operations: [
        "video.transform.filter",
        "video.transform.window",
        "video.transform.groupScenes",
        "video.transform.resampleFeatures",
      ],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect inputs, plans, metadata, and diagnostic helpers.",
      operations: ["describe"],
    },
  ],
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
