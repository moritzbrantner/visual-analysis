import type { Meta, StoryObj } from "@storybook/react-vite";

import { PackageSurfaceWorkbench } from "./index";
import type { PackageAppConfig, PackageSurface, SurfaceRequest, SurfaceResponse } from "./types";

const operations = [
  {
    id: "analysis.summary",
    name: "Analysis summary",
    description: "Create a compact deterministic analysis summary.",
    inputSchema: { type: "object" },
    outputSchema: { type: "object" },
    exampleRequest: { text: "Scene one introduces the package surface workbench." },
    wasmSupported: true,
    serverSupported: true,
  },
  {
    id: "describe",
    name: "Describe package",
    description: "Inspect package metadata.",
    inputSchema: { type: "object" },
    outputSchema: { type: "object" },
    exampleRequest: { includeOperations: true },
    wasmSupported: true,
    serverSupported: true,
  },
];

const surface: PackageSurface = {
  library: "storybook-surface",
  version: "0.1.0",
  capabilities: { runtime: "storybook" },
  operations,
};

function responseFor(request: SurfaceRequest): SurfaceResponse {
  return {
    operation: request.operation,
    value: {
      title: request.operation === "describe" ? "Storybook surface" : "Analysis summary",
      message: "The story runtime returned a deterministic package-surface response.",
      summary: {
        operationCount: operations.length,
        inputFields: Object.keys((request.input ?? {}) as Record<string, unknown>).length,
        wasmReady: true,
      },
      rows: [
        { id: "scene-1", label: "Opening", score: 0.91 },
        { id: "scene-2", label: "Analysis", score: 0.84 },
      ],
    },
    diagnostics: [],
    artifacts: [],
  };
}

const baseConfig: PackageAppConfig = {
  library: "storybook-surface",
  title: "Storybook Package Surface",
  description: "A deterministic package surface used to exercise the reusable workbench UI.",
  domain: "video",
  defaultRuntime: "client-wasm",
  defaultOperation: "analysis.summary",
  featuredOperations: ["analysis.summary", "describe"],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      operations: ["analysis.summary"],
    },
    {
      id: "debug",
      label: "Debug",
      operations: ["describe"],
    },
  ],
  presets: [
    {
      id: "short-summary",
      label: "Short Summary",
      operation: "analysis.summary",
      input: { text: "A package surface renders deterministic results." },
    },
  ],
  wasm: {
    init: async () => ({ ready: true }),
    packageSurface: async () => surface,
    runOperation: async (request) => responseFor(request),
  },
  server: {
    scopedRoute: "/api/rust/packages/storybook-surface",
    standaloneRoute: "",
  },
  resultTabs: [
    {
      id: "rows",
      label: "Rows",
      select: (response) => (response.value as { rows?: unknown[] }).rows ?? [],
    },
  ],
};

const meta = {
  title: "Package Surface/PackageSurfaceWorkbench",
  component: PackageSurfaceWorkbench,
} satisfies Meta<typeof PackageSurfaceWorkbench>;

export default meta;
type Story = StoryObj<typeof meta>;

export const ReadyWasmRuntime: Story = {
  args: {
    config: baseConfig,
  },
};

export const ServerUnavailableFallback: Story = {
  args: {
    config: {
      ...baseConfig,
      defaultRuntime: "overview-server",
      title: "Server Fallback Surface",
    },
  },
};

export const BenchmarkTabVisible: Story = {
  args: {
    config: {
      ...baseConfig,
      benchmarkScenarios: [
        {
          id: "summary",
          label: "Summary",
          operation: "analysis.summary",
          input: { text: "Benchmark the story operation." },
          iterations: 5,
          warmupIterations: 1,
          outputCountPath: ["rows"],
        },
      ],
    },
  },
};
