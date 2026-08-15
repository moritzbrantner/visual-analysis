import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/visual-app-ui/package-surface";
import * as wasm from "@moritzbrantner/image-analysis-io-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "image-analysis-io",
  title: "Image Analysis IO",
  description: "Still-image PNG/JPEG/WebP loading and saving for video-analysis.",
  domain: "image",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/image-analysis-io",
    standaloneRoute: "",
  },
  defaultOperation: "image.io.plan",
  featuredOperations: ["image.io.plan", "image.io.inferFormat", "image.io.supportedFormats", "describe"],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Plan image read and write operations.",
      operations: ["image.io.plan"],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect supported formats, inferred formats, and package metadata.",
      operations: ["image.io.supportedFormats", "image.io.inferFormat", "describe"],
    },
  ],
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
