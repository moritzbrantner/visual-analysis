# image-analysis-classification-app

React, TypeScript, TailwindCSS, Bun, and oxfmt frontend for `image-analysis-classification`.

Run the server:

```bash
cargo run -p image-analysis-classification-server -- --addr 127.0.0.1:3000
```

Run the app:

```bash
bun run --cwd packages/image-analysis-classification-app dev
```

The default app operation is `image.classification.classify`, a server-only
local ViT ONNX workflow that may materialize `Xenova/vit-base-patch16-224`
under `.model-runtime`. Imported label workflows remain available for client
and WASM use.
