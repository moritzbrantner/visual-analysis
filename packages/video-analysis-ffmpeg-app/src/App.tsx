import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/visual-app-ui/package-surface";
import * as wasm from "@moritzbrantner/video-analysis-ffmpeg-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "video-analysis-ffmpeg",
  title: "Video Analysis Ffmpeg",
  description: "FFmpeg-backed media ingest for video-analysis.",
  domain: "video",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/video-analysis-ffmpeg",
    standaloneRoute: "",
  },
  defaultOperation: "video.ffmpeg.probePlan",
  featuredOperations: ["video.ffmpeg.probePlan", "video.ffmpeg.extractFramesPlan", "video.ffmpeg.filterGraphPlan", "describe"],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run the main package workflow.",
      operations: ["video.ffmpeg.probePlan"],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect inputs, plans, metadata, and diagnostic helpers.",
      operations: ["describe", "video.ffmpeg.extractFramesPlan", "video.ffmpeg.filterGraphPlan"],
    },
  ],
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
