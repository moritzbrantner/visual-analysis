import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/visual-app-ui/package-surface";
import * as wasm from "@moritzbrantner/video-analysis-core-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "video-analysis-core",
  title: "Video Analysis Core",
  description: "Core media, timing, detection, and analyzer contracts for video-analysis.",
  domain: "video",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/video-analysis-core",
    standaloneRoute: "",
  },
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
