import type { Meta, StoryObj } from "@storybook/react-vite";

import { ResultViewer } from "./ResultViewer";
import type { SurfaceResponse } from "./types";

const baseResponse: SurfaceResponse = {
  operation: "analysis.summary",
  value: {
    title: "Surface summary",
    message: "The result viewer summarizes scalar fields and nested response content.",
    summary: {
      scenes: 4,
      durationSeconds: 18.4,
      confidence: 0.92,
      cached: false,
    },
    records: [
      { id: "scene-1", label: "Opening" },
      { id: "scene-2", label: "Cutaway" },
    ],
  },
  diagnostics: [],
  artifacts: [],
};

const meta = {
  title: "Package Surface/ResultViewer",
  component: ResultViewer,
} satisfies Meta<typeof ResultViewer>;

export default meta;
type Story = StoryObj<typeof meta>;

export const SummaryResult: Story = {
  args: {
    response: baseResponse,
  },
};

export const DiagnosticsResult: Story = {
  args: {
    response: {
      ...baseResponse,
      diagnostics: [
        { level: "warning", message: "The server model was unavailable; deterministic fallback was used." },
        { level: "info", message: "Two low-confidence observations were omitted from the summary." },
      ],
    },
  },
};

export const ArtifactResult: Story = {
  args: {
    response: {
      ...baseResponse,
      artifacts: [
        { kind: "json", path: "reports/summary.json", bytes: 1840 },
        { kind: "image", path: "frames/scene-1.jpg", width: 1280, height: 720 },
      ],
    },
  },
};
