import {
  PackageSurfaceWorkbench,
  type PackageAppConfig,
} from "@moritzbrantner/visual-app-ui/package-surface";
import * as wasm from "@moritzbrantner/vision-core-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "vision-core",
  title: "Vision Core",
  description: "Shared visual detection, keypoint, embedding, and identity-match contracts.",
  domain: "vision",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/vision-core",
    standaloneRoute: "",
  },
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run the main package workflow.",
      operations: [
        "vision.validateDetection",
        "vision.validateEmbedding",
        "vision.validateIdentityMatch",
        "vision.convertDetectionSummary",
      ],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect inputs, plans, metadata, and diagnostic helpers.",
      operations: ["describe"],
    },
  ],
  defaultOperation: "vision.convertDetectionSummary",
  featuredOperations: [
    "vision.convertDetectionSummary",
    "vision.validateDetection",
    "vision.validateEmbedding",
    "vision.validateIdentityMatch",
    "describe",
  ],
  presets: [
    {
      id: "face-detection-summary",
      label: "Face detection summary",
      operation: "vision.convertDetectionSummary",
      description: "Summarize one localized face detection.",
      input: {
        detections: [
          {
            id: "face-1",
            kind: "face",
            label: "face",
            score: 0.92,
            region: { x: 12, y: 20, width: 80, height: 96 },
            keypoints: [],
            attributes: {},
          },
        ],
      },
    },
  ],
  benchmarkScenarios: [],
  resultTabs: [],
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
