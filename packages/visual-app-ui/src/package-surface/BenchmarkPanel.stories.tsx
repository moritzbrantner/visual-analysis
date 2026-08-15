import type { Meta, StoryObj } from "@storybook/react-vite";

import { BenchmarkPanel } from "./BenchmarkPanel";
import type { PackageAppConfig, SurfaceResponse } from "./types";

const config: PackageAppConfig = {
  library: "storybook-benchmarks",
  title: "Storybook Benchmarks",
  description: "Package benchmark scenarios for Storybook coverage.",
  domain: "text",
  defaultRuntime: "client-wasm",
  wasm: {
    init: async () => ({ ready: true }),
    packageSurface: async () => ({
      library: "storybook-benchmarks",
      version: "0.1.0",
      capabilities: {},
      operations: [],
    }),
    runOperation: async (): Promise<SurfaceResponse> => ({
      operation: "bench.echo",
      value: { tokens: ["rust", "wasm", "surface"], summary: { tokenCount: 3 } },
      diagnostics: [],
      artifacts: [],
    }),
  },
  server: {
    scopedRoute: "/api/rust/packages/storybook-benchmarks",
    standaloneRoute: "",
  },
  benchmarkScenarios: [
    {
      id: "echo",
      label: "Echo",
      operation: "bench.echo",
      input: { text: "Rust WASM surface" },
      iterations: 10,
      warmupIterations: 1,
      outputCountPath: ["tokens"],
    },
  ],
};

const meta = {
  title: "Package Surface/BenchmarkPanel",
  component: BenchmarkPanel,
} satisfies Meta<typeof BenchmarkPanel>;

export default meta;
type Story = StoryObj<typeof meta>;

export const ConfiguredScenarios: Story = {
  args: {
    config,
    runtimeMode: "client-wasm",
  },
};

export const EmptyScenarioState: Story = {
  args: {
    config: {
      ...config,
      benchmarkScenarios: [],
    },
    runtimeMode: "client-wasm",
  },
};
