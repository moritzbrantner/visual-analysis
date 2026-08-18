import { PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/visual-app-ui/package-surface";
import * as wasm from "@moritzbrantner/image-analysis-embeddings-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "image-analysis-embeddings",
  title: "Image Analysis Embeddings",
  description: "Aggregate image task schemas, catalogs, and backend contracts for video-analysis.",
  domain: "image",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/image-analysis-embeddings",
    standaloneRoute: "",
  },
  defaultOperation: "image.embeddings.validate",
  featuredOperations: ["image.embeddings.validate", "image.embeddings.models", "image.embeddings.schema", "describe"],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Validate imported image and face embedding vectors.",
      operations: ["image.embeddings.validate"],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect embedding catalogs, schemas, and package metadata.",
      operations: ["image.embeddings.models", "image.embeddings.schema", "describe"],
    },
  ],
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
