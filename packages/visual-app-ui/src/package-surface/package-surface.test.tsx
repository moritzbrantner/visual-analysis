import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { createTextResultTabs, landscapeContractForOperation, ModelSelector, PackageSurfaceWorkbench, ResultViewer } from "./index";
import type { PackageAppConfig, SurfaceResponse } from "./types";

const operationResponse: SurfaceResponse = {
  operation: "demo.run",
  value: {
    ok: true,
    count: 1,
    title: "Demo result",
    message: "Demo operation completed.",
    summary: { count: 1 },
  },
  diagnostics: [{ code: "demo", message: "diagnostic" }],
  artifacts: [{ id: "artifact-1" }],
};

const scenarioOperations = [
  {
    id: "demo.run",
    name: "Run demo",
    description: "Runs the main workflow.",
    inputSchema: {},
    outputSchema: {},
    exampleRequest: { text: "workflow example" },
    wasmSupported: true,
    serverSupported: true,
  },
  {
    id: "demo.inspect",
    name: "Inspect JSON",
    description: "Inspects advanced JSON.",
    inputSchema: {},
    outputSchema: {},
    exampleRequest: { text: "inspect example" },
    wasmSupported: true,
    serverSupported: true,
  },
];

const curatedOperations = [
  {
    id: "describe",
    name: "Describe",
    description: "Describes the package.",
    curation: { role: "debug" as const, primary: false, sortOrder: 900 },
    inputSchema: {},
    outputSchema: {},
    exampleRequest: { text: "debug example" },
    wasmSupported: true,
    serverSupported: true,
  },
  {
    id: "demo.support",
    name: "Support demo",
    description: "Runs support planning.",
    curation: { role: "support" as const, primary: false, sortOrder: 500 },
    inputSchema: {},
    outputSchema: {},
    exampleRequest: { text: "support example" },
    wasmSupported: true,
    serverSupported: true,
  },
  {
    id: "demo.secondary",
    name: "Secondary workflow",
    description: "Runs a secondary workflow.",
    curation: { role: "workflow" as const, primary: false, sortOrder: 20 },
    inputSchema: {},
    outputSchema: {},
    exampleRequest: { text: "secondary example" },
    wasmSupported: true,
    serverSupported: true,
  },
  {
    id: "demo.primary",
    name: "Primary workflow",
    description: "Runs the primary workflow.",
    curation: { role: "workflow" as const, primary: true, sortOrder: 10 },
    inputSchema: {},
    outputSchema: {},
    exampleRequest: { text: "primary example" },
    wasmSupported: true,
    serverSupported: true,
  },
];

function config(overrides: Partial<PackageAppConfig> = {}): PackageAppConfig {
  return {
    library: "demo-package",
    title: "Demo Package",
    description: "Demo package workbench.",
    domain: "text",
    wasm: {
      init: vi.fn(async () => undefined),
      packageSurface: vi.fn(() => ({
        library: "demo-package",
        version: "0.1.0",
        capabilities: {},
        operations: [
          {
            id: "demo.run",
            name: "Run demo",
            description: "Runs the demo operation.",
            inputSchema: {},
            outputSchema: {},
            exampleRequest: { text: "hello" },
            wasmSupported: true,
            serverSupported: true,
          },
        ],
      })),
      runOperation: vi.fn(async () => operationResponse),
    },
    server: {
      scopedRoute: "/api/rust/packages/demo-package",
      standaloneRoute: "",
    },
    ...overrides,
  };
}

function scenarioConfig(overrides: Partial<PackageAppConfig> = {}): PackageAppConfig {
  return config({
    wasm: {
      init: vi.fn(async () => undefined),
      packageSurface: vi.fn(() => ({
        library: "demo-package",
        version: "0.1.0",
        capabilities: {},
        operations: scenarioOperations,
      })),
      runOperation: vi.fn(async () => operationResponse),
    },
    operationGroups: [
      {
        id: "workflow",
        label: "Workflow",
        operations: ["demo.run"],
      },
      {
        id: "debug",
        label: "Debug",
        operations: ["demo.inspect"],
      },
    ],
    defaultPresetId: "demo-preset",
    presets: [
      {
        id: "demo-preset",
        label: "Curated demo",
        operation: "demo.run",
        description: "Run curated demo input.",
        input: { text: "curated input" },
      },
      {
        id: "demo-alt",
        label: "Alternate demo",
        operation: "demo.run",
        description: "Run alternate demo input.",
        input: { text: "alternate input" },
      },
    ],
    ...overrides,
  });
}

beforeEach(() => {
  localStorage.clear();
  window.history.replaceState({}, "", "/");
  vi.restoreAllMocks();
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/health")) {
        return jsonResponse({ ok: true, package: "demo-package-server", library: "demo-package" });
      }
      if (url.endsWith("/api/package")) {
        return jsonResponse({
          library: "demo-package",
          version: "0.1.0",
          operations: [
            {
              id: "demo.run",
              name: "Run demo",
              description: "Runs the demo operation.",
              exampleRequest: { text: "server", includeNearDuplicates: true },
              wasmSupported: true,
              serverSupported: true,
            },
          ],
        });
      }
      if (url.endsWith("/api/models")) {
        return jsonResponse([
          {
            id: "large-model",
            label: "Large model",
            task: "demo",
            runtime: "onnx",
            supported: false,
            fallback: "small-model",
            note: "Requires optional runtime.",
          },
        ]);
      }
      if (url.endsWith("/api/run")) {
        return jsonResponse(operationResponse);
      }
      if (url.endsWith(".webm") || url.endsWith(".mp4")) {
        const type = url.endsWith(".mp4") ? "video/mp4" : "video/webm";
        return new Response(new Blob(["sample video"], { type }), {
          status: 200,
          headers: { "content-type": type },
        });
      }
      return new Response("not found", { status: 404 });
    }),
  );
});

afterEach(() => cleanup());

describe("PackageSurfaceWorkbench", () => {
  test("loads operations and model fallback metadata", async () => {
    render(<PackageSurfaceWorkbench config={config()} />);

    expect(await screen.findByRole("heading", { name: "Demo Package" })).toBeTruthy();
    expect(await screen.findByRole("combobox", { name: "Operation" })).toBeTruthy();
    expect(await screen.findByText("Large model", { exact: false })).toBeTruthy();
    expect((await screen.findAllByText("Fallback")).length).toBeGreaterThan(0);
  });

  test("parses and renders optional curated landscape metadata", async () => {
    render(
      <PackageSurfaceWorkbench
        config={config({
          wasm: {
            init: vi.fn(async () => undefined),
            packageSurface: vi.fn(() => ({
              library: "demo-package",
              version: "0.1.0",
              capabilities: {},
              operations: [
                {
                  id: "demo.run",
                  name: "Run demo",
                  description: "Runs the demo operation.",
                  inputSchema: {
                    type: "object",
                    xLandscape: {
                      function: {
                        id: "demo.curated",
                        owner: "demo-package",
                        stability: "stable",
                        inputs: [
                          {
                            name: "document",
                            typeRef: {
                              id: "text.document",
                              owner: "moritzbrantner-text-core",
                              rustType: "text_core::TextDocumentContract",
                            },
                            required: true,
                            cardinality: "one",
                          },
                          {
                            name: "config",
                            typeRef: {
                              id: "audio.transcriptionConfig",
                              owner: "moritzbrantner-audio-analysis-transcription",
                            },
                            required: true,
                            cardinality: "one",
                          },
                        ],
                        outputs: [
                          {
                            name: "segments",
                            typeRef: {
                              id: "text.segment",
                              owner: "moritzbrantner-text-core",
                              rustType: "text_core::TextSegmentContract",
                            },
                            required: true,
                            cardinality: "many",
                          },
                          {
                            name: "document",
                            typeRef: {
                              id: "text.document",
                              owner: "moritzbrantner-text-core",
                            },
                            required: true,
                            cardinality: "one",
                          },
                        ],
                      },
                    },
                  },
                  outputSchema: {},
                  exampleRequest: { text: "hello" },
                  wasmSupported: true,
                  serverSupported: true,
                },
              ],
            })),
            runOperation: vi.fn(async () => operationResponse),
          },
        })}
      />,
    );

    expect(await screen.findByText("Curated I/O")).toBeTruthy();
    expect(screen.getByText("demo.curated")).toBeTruthy();
    expect(screen.getByText("stable")).toBeTruthy();
    expect(screen.getAllByText("document: text.document").length).toBe(2);
    expect(screen.getByText("config: audio.transcriptionConfig")).toBeTruthy();
    expect(screen.getByText("segments: text.segment")).toBeTruthy();
  });

  test("omits curated landscape section when metadata is absent", async () => {
    render(<PackageSurfaceWorkbench config={config()} />);

    expect(await screen.findByRole("heading", { name: "Demo Package" })).toBeTruthy();
    expect(screen.queryByText("Curated I/O")).toBeNull();
  });

  test("ignores malformed curated landscape metadata", async () => {
    render(
      <PackageSurfaceWorkbench
        config={config({
          wasm: {
            init: vi.fn(async () => undefined),
            packageSurface: vi.fn(() => ({
              library: "demo-package",
              version: "0.1.0",
              capabilities: {},
              operations: [
                {
                  id: "demo.run",
                  name: "Run demo",
                  description: "Runs the demo operation.",
                  inputSchema: {
                    type: "object",
                    xLandscape: {
                      function: {
                        inputs: [{ name: "document" }],
                        outputs: [{ name: "segments" }],
                      },
                    },
                  },
                  outputSchema: {},
                  exampleRequest: { text: "hello" },
                  wasmSupported: true,
                  serverSupported: true,
                },
              ],
            })),
            runOperation: vi.fn(async () => operationResponse),
          },
        })}
      />,
    );

    expect(await screen.findByRole("heading", { name: "Demo Package" })).toBeTruthy();
    expect(screen.queryByText("Curated I/O")).toBeNull();
  });

  test("edits operation input through form fields", async () => {
    const runOperation = vi.fn(async () => operationResponse);
    render(
      <PackageSurfaceWorkbench
        config={config({
          wasm: {
            init: vi.fn(async () => undefined),
            packageSurface: vi.fn(() => ({
              library: "demo-package",
              version: "0.1.0",
              capabilities: {},
              operations: [
                {
                  id: "demo.run",
                  name: "Run demo",
                  description: "Runs the demo operation.",
                  inputSchema: {},
                  outputSchema: {},
                  exampleRequest: { text: "hello", includeNearDuplicates: true },
                  wasmSupported: true,
                  serverSupported: true,
                },
              ],
            })),
            runOperation,
          },
        })}
      />,
    );

    const textInput = await screen.findByDisplayValue(/hello|server/);
    fireEvent.change(textInput, { target: { value: "updated text" } });
    const toggle = screen.getByRole("switch", { name: "Include Near Duplicates" });
    expect(toggle.getAttribute("aria-checked")).toBe("true");
    fireEvent.click(toggle);
    expect(toggle.getAttribute("aria-checked")).toBe("false");
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    await waitFor(() => {
      expect(runOperation).toHaveBeenCalledWith({
        operation: "demo.run",
        input: { text: "updated text", includeNearDuplicates: false },
      });
    });
  });

  test("focused input fields hide internal request fields but preserve them in the payload", async () => {
    const runOperation = vi.fn(async () => operationResponse);
    const focusedExample = {
      text: "visible text",
      includeNearDuplicates: true,
      embedding: { mode: "hashed", dimensions: 128 },
    };
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("/health")) {
          return jsonResponse({ ok: true, package: "demo-package-server", library: "demo-package" });
        }
        if (url.endsWith("/api/package")) {
          return jsonResponse({
            library: "demo-package",
            version: "0.1.0",
            operations: [
              {
                id: "demo.run",
                name: "Run demo",
                description: "Runs the demo operation.",
                exampleRequest: focusedExample,
                wasmSupported: true,
                serverSupported: true,
              },
            ],
          });
        }
        if (url.endsWith("/api/models")) {
          return jsonResponse([]);
        }
        if (url.endsWith("/api/run")) {
          return jsonResponse(operationResponse);
        }
        return new Response("not found", { status: 404 });
      }),
    );

    render(
      <PackageSurfaceWorkbench
        config={config({
          workbench: {
            inputChrome: "compact",
            showLandscapeContract: false,
            inputFields: {
              "demo.run": ["text"],
            },
          },
          wasm: {
            init: vi.fn(async () => undefined),
            packageSurface: vi.fn(() => ({
              library: "demo-package",
              version: "0.1.0",
              capabilities: {},
              operations: [
                {
                  id: "demo.run",
                  name: "Run demo",
                  description: "Runs the demo operation.",
                  inputSchema: {},
                  outputSchema: {},
                  exampleRequest: focusedExample,
                  wasmSupported: true,
                  serverSupported: true,
                },
              ],
            })),
            runOperation,
          },
        })}
      />,
    );

    const textInput = await screen.findByDisplayValue(/visible text|server/);
    fireEvent.change(textInput, { target: { value: "updated visible text" } });
    expect(screen.queryByRole("switch", { name: "Include Near Duplicates" })).toBeNull();
    expect(screen.queryByLabelText("Embedding")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    await waitFor(() => {
      expect(runOperation).toHaveBeenCalledWith({
        operation: "demo.run",
        input: {
          text: "updated visible text",
          includeNearDuplicates: true,
          embedding: { mode: "hashed", dimensions: 128 },
        },
      });
    });
  });

  test("focused layout can suppress metadata side panels", async () => {
    render(
      <PackageSurfaceWorkbench
        config={config({
          workbench: {
            layout: "focused",
            sidePanels: {
              runtime: false,
              models: false,
              files: false,
              support: false,
            },
          },
        })}
      />,
    );

    expect(await screen.findByRole("heading", { name: "Demo Package" })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: "Runtime" })).toBeNull();
    expect(screen.queryByRole("heading", { name: "Models" })).toBeNull();
    expect(screen.queryByRole("heading", { name: "Support" })).toBeNull();
  });

  test("runs the selected operation", async () => {
    render(<PackageSurfaceWorkbench config={config()} />);

    await screen.findByRole("combobox", { name: "Operation" });
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    await waitFor(() => expect(screen.getByText("Demo result")).toBeTruthy());
    expect(screen.getByText("Demo operation completed.")).toBeTruthy();
    expect(screen.getAllByText("Count").length).toBeGreaterThan(0);
    expect(screen.getByText("1 diagnostics")).toBeTruthy();
  });

  test("renders one Scenario combobox when presets are configured", async () => {
    render(<PackageSurfaceWorkbench config={scenarioConfig()} />);

    expect(await screen.findByRole("combobox", { name: "Scenario" })).toBeTruthy();
    expect(screen.queryByRole("combobox", { name: "Operation" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Curated demo" })).toBeNull();
    expect(screen.getByRole("option", { name: "Curated demo" })).toBeTruthy();
    expect(screen.getByRole("option", { name: "Alternate demo" })).toBeTruthy();
    expect(screen.getByRole("option", { name: "Inspect JSON" })).toBeTruthy();
  });

  test("loads the configured default preset on initial mount", async () => {
    localStorage.setItem("package-workbench:demo-package:demo.run", JSON.stringify({ text: "stale draft" }, null, 2));

    render(<PackageSurfaceWorkbench config={scenarioConfig()} />);

    const scenario = (await screen.findByRole("combobox", { name: "Scenario" })) as HTMLSelectElement;
    expect(scenario.value).toBe("preset:demo-preset");
    expect(await screen.findByDisplayValue("curated input")).toBeTruthy();
  });

  test("selecting a preset updates operation, input, URL params, and run payload", async () => {
    localStorage.setItem("package-workbench:demo-package:demo.run", JSON.stringify({ text: "stale draft" }, null, 2));
    const runOperation = vi.fn(async () => operationResponse);
    render(
      <PackageSurfaceWorkbench
        config={scenarioConfig({
          wasm: {
            init: vi.fn(async () => undefined),
            packageSurface: vi.fn(() => ({
              library: "demo-package",
              version: "0.1.0",
              capabilities: {},
              operations: scenarioOperations,
            })),
            runOperation,
          },
        })}
      />,
    );

    const scenario = (await screen.findByRole("combobox", { name: "Scenario" })) as HTMLSelectElement;
    fireEvent.change(scenario, { target: { value: "preset:demo-alt" } });

    expect(await screen.findByDisplayValue("alternate input")).toBeTruthy();
    expect(window.location.search).toContain("operation=demo.run");
    expect(window.location.search).toContain("preset=demo-alt");
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    await waitFor(() => {
      expect(runOperation).toHaveBeenCalledWith({
        operation: "demo.run",
        input: { text: "alternate input" },
      });
    });
  });

  test("debug raw operations remain selectable through the scenario dropdown", async () => {
    render(<PackageSurfaceWorkbench config={scenarioConfig()} />);

    const scenario = (await screen.findByRole("combobox", { name: "Scenario" })) as HTMLSelectElement;
    fireEvent.change(scenario, { target: { value: "operation:demo.inspect" } });

    expect(await screen.findByDisplayValue("inspect example")).toBeTruthy();
    expect(window.location.search).toContain("operation=demo.inspect");
    expect(window.location.search).not.toContain("preset=");
  });

  test("preset query param restores the selected scenario", async () => {
    localStorage.setItem("package-workbench:demo-package:demo.run", JSON.stringify({ text: "stale draft" }, null, 2));
    window.history.replaceState({}, "", "/?operation=demo.run&preset=demo-alt");

    render(<PackageSurfaceWorkbench config={scenarioConfig()} />);

    const scenario = (await screen.findByRole("combobox", { name: "Scenario" })) as HTMLSelectElement;
    expect(scenario.value).toBe("preset:demo-alt");
    expect(await screen.findByDisplayValue("alternate input")).toBeTruthy();
  });

  test("groups operations under category tabs", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("/health")) {
          return jsonResponse({ ok: true, package: "demo-package-server", library: "demo-package" });
        }
        if (url.endsWith("/api/package")) {
          return jsonResponse({
            library: "demo-package",
            version: "0.1.0",
            operations: [
              {
                id: "demo.run",
                name: "Run demo",
                description: "Runs the main workflow.",
                exampleRequest: { mode: "run" },
                wasmSupported: true,
                serverSupported: true,
              },
              {
                id: "demo.inspect",
                name: "Inspect JSON",
                description: "Inspects advanced JSON.",
                exampleRequest: { mode: "inspect" },
                wasmSupported: true,
                serverSupported: true,
              },
            ],
          });
        }
        if (url.endsWith("/api/models")) {
          return jsonResponse([]);
        }
        return new Response("not found", { status: 404 });
      }),
    );

    render(
      <PackageSurfaceWorkbench
        config={config({
          wasm: undefined,
          defaultOperation: "demo.run",
          operationGroups: [
            {
              id: "workflow",
              label: "Workflow",
              operations: ["demo.run"],
            },
            {
              id: "advanced",
              label: "Debug",
              operations: ["demo.inspect"],
            },
          ],
        })}
      />,
    );

    expect(await screen.findByRole("tab", { name: "Workflow" })).toBeTruthy();
    expect(screen.getByRole("tab", { name: "Debug" })).toBeTruthy();
    expect(screen.getByRole("option", { name: "Run demo" })).toBeTruthy();
    expect(screen.queryByRole("option", { name: "Inspect JSON" })).toBeNull();

    fireEvent.click(screen.getByRole("tab", { name: "Debug" }));

    expect(await screen.findByRole("option", { name: "Inspect JSON" })).toBeTruthy();
    expect(screen.queryByRole("option", { name: "Run demo" })).toBeNull();
    expect((await screen.findByDisplayValue(/inspect/)) as HTMLTextAreaElement).toBeTruthy();
  });

  test("derives operation groups and primary selection from Rust curation", async () => {
    render(
      <PackageSurfaceWorkbench
        config={config({
          wasm: {
            init: vi.fn(async () => undefined),
            packageSurface: vi.fn(() => ({
              library: "demo-package",
              version: "0.1.0",
              capabilities: {},
              operations: curatedOperations,
            })),
            runOperation: vi.fn(async () => operationResponse),
          },
        })}
      />,
    );

    expect(await screen.findByRole("tab", { name: "Workflow" })).toBeTruthy();
    expect(screen.getByRole("tab", { name: "Support" })).toBeTruthy();
    expect(screen.getByRole("tab", { name: "Debug" })).toBeTruthy();
    expect((screen.getByRole("combobox", { name: "Operation" }) as HTMLSelectElement).value).toBe("demo.primary");
    expect(screen.getByRole("option", { name: "Primary workflow" })).toBeTruthy();
    expect(screen.getByRole("option", { name: "Secondary workflow" })).toBeTruthy();

    fireEvent.click(screen.getByRole("tab", { name: "Support" }));
    expect(await screen.findByRole("option", { name: "Support demo" })).toBeTruthy();
    expect(screen.queryByRole("option", { name: "Primary workflow" })).toBeNull();
  });

  test("uses featured operations as a presentation ordering override", async () => {
    render(
      <PackageSurfaceWorkbench
        config={config({
          featuredOperations: ["demo.secondary", "demo.primary"],
          wasm: {
            init: vi.fn(async () => undefined),
            packageSurface: vi.fn(() => ({
              library: "demo-package",
              version: "0.1.0",
              capabilities: {},
              operations: [curatedOperations[3], curatedOperations[2]],
            })),
            runOperation: vi.fn(async () => operationResponse),
          },
        })}
      />,
    );

    const operation = (await screen.findByRole("combobox", { name: "Operation" })) as HTMLSelectElement;
    expect([...operation.options].map((option) => option.textContent)).toEqual([
      "Secondary workflow",
      "Primary workflow",
    ]);
  });

  test("falls back to overview server when WASM initialization fails", async () => {
    const runOperation = vi.fn(async () => operationResponse);
    const packageConfig = config({
      wasm: {
        init: vi.fn(async () => {
          throw new Error("missing generated wasm");
        }),
        packageSurface: vi.fn(),
        runOperation,
      },
    });

    render(<PackageSurfaceWorkbench config={packageConfig} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Overview Server" }).className).toContain("bg-zinc-950");
    });
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    await waitFor(() => expect(screen.getByText("Demo result")).toBeTruthy());
    expect(runOperation).not.toHaveBeenCalled();
    expect(fetch).toHaveBeenCalledWith(
      "http://127.0.0.1:3000/api/rust/packages/demo-package/api/run",
      expect.objectContaining({ method: "POST" }),
    );
  });

  test("defaults to overview server when configured", async () => {
    render(<PackageSurfaceWorkbench config={config({ defaultRuntime: "overview-server" })} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Overview Server" }).className).toContain("bg-zinc-950");
    });
  });

  test("moves server-only operations away from client WASM", async () => {
    render(
      <PackageSurfaceWorkbench
        config={config({
          defaultOperation: "demo.serverOnly",
          wasm: {
            init: vi.fn(async () => undefined),
            packageSurface: vi.fn(() => ({
              library: "demo-package",
              version: "0.1.0",
              capabilities: {},
              operations: [
                {
                  id: "demo.serverOnly",
                  name: "Server only",
                  description: "Runs on the server.",
                  inputSchema: {},
                  outputSchema: {},
                  exampleRequest: { text: "server" },
                  wasmSupported: false,
                  serverSupported: true,
                },
              ],
            })),
            runOperation: vi.fn(async () => operationResponse),
          },
        })}
      />,
    );

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Overview Server" }).className).toContain("bg-zinc-950");
    });
    expect((screen.getByRole("button", { name: "Client WASM" }) as HTMLButtonElement).disabled).toBe(true);
  });

  test("loads bundled video samples into the request form", async () => {
    render(<PackageSurfaceWorkbench config={config({ domain: "video" })} />);

    await screen.findByDisplayValue(/hello|server/);
    expect(await screen.findByRole("button", { name: "Test Pattern" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Color Bars" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Moving Box" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "COLMAP Test Video" })).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Test Pattern" }));

    const editor = (await screen.findByDisplayValue(/data:video\/webm/)) as HTMLTextAreaElement;
    expect(editor.value).toContain("data:video/webm");
  });

  test("preserves a loaded sample when the second runtime surface resolves", async () => {
    const baselineFetch = fetch;
    let resolveServerSurface: (response: Response) => void = () => undefined;
    const serverSurface = new Promise<Response>((resolve) => {
      resolveServerSurface = resolve;
    });
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
        if (String(input).endsWith("/api/package")) {
          return serverSurface;
        }
        return baselineFetch(input, init);
      }),
    );

    render(<PackageSurfaceWorkbench config={scenarioConfig({ domain: "video" })} />);

    await screen.findByDisplayValue("curated input");
    fireEvent.click(screen.getByRole("button", { name: "Test Pattern" }));
    await screen.findByDisplayValue(/data:video\/webm/);

    resolveServerSurface(
      jsonResponse({
        library: "demo-package",
        version: "0.1.0",
        operations: scenarioOperations,
      }),
    );
    await waitFor(() => {
      expect(screen.getByText("demo-package-server")).toBeTruthy();
    });

    expect((screen.getByDisplayValue(/data:video\/webm/) as HTMLTextAreaElement).value).toContain("data:video/webm");
  });

  test("loads COLMAP sample patches and preview data into the request form", async () => {
    render(<PackageSurfaceWorkbench config={config({ domain: "video" })} />);

    await screen.findByDisplayValue(/hello|server/);
    fireEvent.click(screen.getByRole("button", { name: "COLMAP Test Video" }));

    await waitFor(() => {
      const editors = screen.getAllByDisplayValue(/test-video\.mp4/) as HTMLTextAreaElement[];
      expect(editors.some((editor) => editor.value.includes("prototypes/web/video-analysis-web/public/samples/video/test-video.mp4"))).toBe(true);
    });
    expect(
      (screen.getAllByDisplayValue(/\/samples\/video\/test-video\.mp4/) as HTMLTextAreaElement[]).some((editor) =>
        editor.value.includes("/samples/video/test-video.mp4"),
      ),
    ).toBe(true);
    expect((screen.getByDisplayValue(/\.external-test-tools\/colmap-runs\/test-video/) as HTMLTextAreaElement).value).toContain(
      ".external-test-tools/colmap-runs/test-video",
    );
    expect(((await screen.findByDisplayValue(/data:video\/mp4/)) as HTMLTextAreaElement).value).toContain("data:video/mp4");
  });

  test("shows setup guidance when the optional COLMAP sample is missing", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("/health")) {
          return jsonResponse({ ok: true, package: "demo-package-server", library: "demo-package" });
        }
        if (url.endsWith("/api/package")) {
          return jsonResponse({
            library: "demo-package",
            version: "0.1.0",
            operations: [
              {
                id: "demo.run",
                name: "Run demo",
                description: "Runs the demo operation.",
                exampleRequest: { text: "server" },
                wasmSupported: true,
                serverSupported: true,
              },
            ],
          });
        }
        if (url.endsWith("/api/models")) {
          return jsonResponse([]);
        }
        if (url.endsWith(".mp4")) {
          return new Response("missing", { status: 404 });
        }
        return new Response("not found", { status: 404 });
      }),
    );
    render(<PackageSurfaceWorkbench config={config({ domain: "video" })} />);

    await screen.findByDisplayValue(/hello|server/);
    fireEvent.click(screen.getByRole("button", { name: "COLMAP Test Video" }));

    expect(await screen.findByText(/bun run setup:colmap-video/)).toBeTruthy();
    await waitFor(() => {
      const editors = screen.getAllByDisplayValue(/test-video\.mp4/) as HTMLTextAreaElement[];
      expect(editors.some((editor) => editor.value.includes("test-video.mp4"))).toBe(true);
    });
  });

  test("disables Run when no runtime is available", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("unavailable", { status: 503 })),
    );
    render(
      <PackageSurfaceWorkbench
        config={config({
          wasm: {
            init: vi.fn(async () => {
              throw new Error("missing generated wasm");
            }),
            packageSurface: vi.fn(),
            runOperation: vi.fn(),
          },
        })}
      />,
    );

    expect(await screen.findByText("No runnable runtime is available for this package.")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Run" }) as HTMLButtonElement).disabled).toBe(true);
  });
});

describe("landscapeContractForOperation", () => {
  test("normalizes xLandscape when present and returns null when absent", () => {
    expect(
      landscapeContractForOperation({
        id: "demo.empty",
        name: "Empty",
        inputSchema: {},
        outputSchema: {},
        exampleRequest: {},
        wasmSupported: true,
        serverSupported: true,
      }),
    ).toBeNull();

    const landscape = landscapeContractForOperation({
      id: "demo.run",
      name: "Run",
      inputSchema: {
        xLandscape: {
          function: {
            id: "demo.curated",
            owner: "demo-package",
            inputs: [
              {
                name: "request",
                type_ref: { id: "runtime.surfaceRequest", owner: "moritzbrantner-runtime-core" },
                required: true,
                cardinality: "one",
              },
            ],
            outputs: [],
            stability: "stable",
          },
        },
      },
      outputSchema: {},
      exampleRequest: {},
      wasmSupported: true,
      serverSupported: true,
    });

    expect(landscape?.function.id).toBe("demo.curated");
    expect(landscape?.function.inputs[0]?.typeRef.id).toBe("runtime.surfaceRequest");
  });
});

describe("createTextResultTabs", () => {
  const tabs = createTextResultTabs({
    library: "text-demo",
    primaryOperations: {
      "text.demo": {
        title: "Demo text operation",
        summaryFields: ["count", "score"],
        listFields: ["predictions", "keywords", "results", "segments", "tokens"],
        objectFields: ["model", "metadata"],
        explanation: () => "The text operation scored the sample input and exposed focused result sections.",
      },
    },
  });

  test("renders title, message, and scalar summary cards", () => {
    render(
      <ResultViewer
        response={{
          operation: "text.demo",
          value: {
            title: "Configured result",
            message: "Completed the text run.",
            summary: { count: 3, score: 0.75 },
          },
          diagnostics: [],
          artifacts: [],
        }}
        resultTabs={tabs}
      />,
    );

    expect(screen.getByText("Configured result")).toBeTruthy();
    expect(screen.getByText("Completed the text run.")).toBeTruthy();
    expect(screen.getByText("Count")).toBeTruthy();
    expect(screen.getByText("0.750")).toBeTruthy();
    expect(screen.getByText("The text operation scored the sample input and exposed focused result sections.")).toBeTruthy();
  });

  test("renders configured list fields and keeps the raw JSON tab available", () => {
    render(
      <ResultViewer
        response={{
          operation: "text.demo",
          value: {
            operation: "text.demo",
            title: "Lists",
            message: "Lists returned.",
            summary: { count: 2 },
            predictions: [{ label: "positive", score: 0.9 }],
            keywords: [{ term: "rust", score: 0.8 }],
            results: [{ id: "doc-1", score: 0.7 }],
            segments: [{ text: "Hello", startSeconds: 1 }],
            tokens: [{ text: "Hello" }],
          },
          diagnostics: [],
          artifacts: [],
        }}
        resultTabs={tabs}
      />,
    );

    expect(screen.getByRole("heading", { name: "Predictions" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Keywords" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Results" })).toBeTruthy();
    expect(screen.getByText("positive")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: /JSON/ }));

    expect(screen.getByText(/"operation": "text.demo"/)).toBeTruthy();
  });

  test("handles missing configured fields without crashing", () => {
    render(
      <ResultViewer
        response={{
          operation: "text.demo",
          value: {
            title: "Sparse",
            message: "Sparse response.",
            summary: {},
          },
          diagnostics: [],
          artifacts: [],
        }}
        resultTabs={tabs}
      />,
    );

    expect(screen.getByText("Sparse")).toBeTruthy();
    expect(screen.getByText("Sparse response.")).toBeTruthy();
  });
});

describe("ModelSelector", () => {
  test("displays reference-only fallback messaging and metadata", () => {
    render(
      <ModelSelector
        models={[
          {
            id: "reference-model",
            label: "Reference model",
            task: "classification",
            runtime: "onnx",
            supported: false,
            loadable: false,
            fallback: "lexical_fallback",
            requiredFeature: "onnx",
            requiredSetup: "Download model weights",
            smokeOperation: "classification.classify",
            source: "overview-server",
          },
        ]}
        selectedModel="reference-model"
        onSelectModel={vi.fn()}
      />,
    );

    expect(screen.getByText("Catalog metadata only; this page will use the fallback or deterministic operation.")).toBeTruthy();
    expect(screen.getByText("overview-server")).toBeTruthy();
    expect(screen.getByText("lexical_fallback")).toBeTruthy();
    expect(screen.getByText("Download model weights")).toBeTruthy();
  });
});

function jsonResponse(value: unknown): Response {
  return new Response(JSON.stringify(value), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}
