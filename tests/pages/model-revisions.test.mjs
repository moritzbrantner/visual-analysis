import test from "node:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

import { BROWSER_VISION_MODELS } from "../../site/vision-models.js";

// Issue #71: every browser model load must name an exact, committed Hugging Face
// revision so the Pages site's model behaviour cannot change upstream without
// a commit in this repository.

const SITE_DIR = fileURLToPath(new URL("../../site/", import.meta.url));
const COMMIT_SHA = /^[0-9a-f]{40}$/;

function stripComments(source) {
  return source
    .replace(/\/\*[\s\S]*?\*\//g, (match) => match.replace(/[^\n]/g, " "))
    .replace(/(^|[^:"'`\\])\/\/[^\n]*/g, (match, prefix) => prefix);
}

// Returns the text between the parentheses of a call starting at `openIndex`.
function callArguments(source, openIndex) {
  let depth = 0;
  for (let index = openIndex; index < source.length; index += 1) {
    const char = source[index];
    if (char === "(") depth += 1;
    else if (char === ")") {
      depth -= 1;
      if (depth === 0) return source.slice(openIndex + 1, index);
    }
  }
  throw new Error("Unbalanced parentheses while scanning a model load call.");
}

function findModelLoads() {
  const loads = [];
  for (const name of readdirSync(SITE_DIR).filter((file) => file.endsWith(".js"))) {
    const source = stripComments(readFileSync(path.join(SITE_DIR, name), "utf8"));
    const pattern = /(?<!function\s+)\b(from_pretrained|pipeline)\s*\(/g;
    for (const match of source.matchAll(pattern)) {
      const openIndex = match.index + match[0].length - 1;
      const line = source.slice(0, match.index).split("\n").length;
      loads.push({
        file: name,
        line,
        kind: match[1],
        args: callArguments(source, openIndex),
      });
    }
  }
  return loads;
}

test("every exported browser vision model records an exact commit revision", () => {
  assert.ok(BROWSER_VISION_MODELS.length > 0, "expected at least one browser model");
  for (const model of BROWSER_VISION_MODELS) {
    assert.match(
      String(model.revision ?? ""),
      COMMIT_SHA,
      `${model.id} must pin an exact 40-hex Hugging Face commit revision`,
    );
  }
});

test("the model scan finds the known transformers.js loads, all in vision-models.js", () => {
  const loads = findModelLoads();
  // Guard against the scanner silently matching nothing: today's loads are
  // SamModel.from_pretrained, AutoProcessor.from_pretrained and pipeline().
  assert.ok(
    loads.filter(({ kind }) => kind === "from_pretrained").length >= 2,
    "expected the SAM model and processor from_pretrained loads",
  );
  assert.ok(
    loads.some(({ kind }) => kind === "pipeline"),
    "expected the zero-shot detection pipeline load",
  );
  for (const load of loads) {
    assert.equal(
      load.file,
      "vision-models.js",
      `model load at site/${load.file}:${load.line} must go through vision-models.js`,
    );
  }
});

test("every from_pretrained and pipeline load passes a pinned revision", () => {
  const unpinned = [];
  for (const load of findModelLoads()) {
    const where = `site/${load.file}:${load.line} ${load.kind}(${load.args.replace(/\s+/g, " ").trim()})`;
    if (!/\brevision\b/.test(load.args)) {
      unpinned.push(where);
      continue;
    }
    // A literal revision must itself be an exact commit SHA; a reference
    // (constant or model-entry field) is covered by the export check above.
    for (const literal of load.args.matchAll(/\brevision\s*:\s*(["'`])([^"'`]*)\1/g)) {
      assert.match(literal[2], COMMIT_SHA, `${where}: revision literal must be a 40-hex commit SHA`);
    }
  }
  assert.deepEqual(unpinned, [], `model loads without a revision:\n${unpinned.join("\n")}`);
});

test("pinned revision literals in site sources are exact commit SHAs", () => {
  const source = stripComments(readFileSync(path.join(SITE_DIR, "vision-models.js"), "utf8"));
  for (const literal of source.matchAll(/\brevision\s*:\s*(["'`])([^"'`]*)\1/g)) {
    assert.match(literal[2], COMMIT_SHA, `revision "${literal[2]}" must be a 40-hex commit SHA`);
  }
  for (const branchy of source.matchAll(/\brevision\s*:\s*(["'`])(main|master|HEAD)\1/g)) {
    assert.fail(`revision "${branchy[2]}" is a moving ref, not a commit`);
  }
});
