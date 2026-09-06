import { access, readFile, stat } from "node:fs/promises";
import { join, resolve } from "node:path";

const root = resolve(process.argv[2] ?? "_site");
const requiredFiles = [
  "index.html",
  "styles.css",
  "analysis.js",
  "app.js",
  ".nojekyll",
  "wasm/image-analysis-core/index.js",
  "wasm/image-analysis-processing/index.js",
];

for (const relative of requiredFiles) {
  const path = join(root, relative);
  await access(path);
  const info = await stat(path);
  if (!info.isFile()) throw new Error(`${relative} is not a file`);
}

for (const wasmPackage of ["image-analysis-core", "image-analysis-processing"]) {
  const pkgDir = join(root, "wasm", wasmPackage, "pkg");
  await access(pkgDir);
}

const html = await readFile(join(root, "index.html"), "utf8");
for (const requiredText of [
  "Visual Inspector",
  "Private by default".replace("Private", "Local"),
  "image.core.lumaHistogram",
  "image.processing.hash",
  "script type=\"module\" src=\"./app.js\"",
]) {
  if (!html.includes(requiredText)) throw new Error(`index.html missing: ${requiredText}`);
}

const app = await readFile(join(root, "app.js"), "utf8");
for (const requiredText of [
  "./wasm/image-analysis-core/index.js",
  "./wasm/image-analysis-processing/index.js",
  "VIDEO_SAMPLE_COUNT",
]) {
  if (!app.includes(requiredText)) throw new Error(`app.js missing: ${requiredText}`);
}

console.log(`Pages artifact looks complete: ${root}`);
