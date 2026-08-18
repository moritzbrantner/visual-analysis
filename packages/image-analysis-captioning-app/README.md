# image-analysis-captioning-app

React, TypeScript, TailwindCSS, Bun, and oxfmt frontend for `image-analysis-captioning`.

Run the server:

```bash
cargo run -p image-analysis-captioning-server -- --addr 127.0.0.1:3000
```

Run the app:

```bash
bun run --cwd packages/image-analysis-captioning-app dev
```

The default app operation is `image.captioning.caption`, a server-only local
ViT-GPT2 ONNX workflow that may materialize
`Xenova/vit-gpt2-image-captioning` under `.model-runtime`. Imported caption
workflows remain available for client and WASM use.
