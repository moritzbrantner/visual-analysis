// Verify that a Pages artifact was built from a clean checkout of exactly the
// given repository checkout and its declared source pins (#69).
//
// Usage: node scripts/check-pages-provenance.mjs <site-dir> [checkout-dir]
// The checkout defaults to the current directory. Fails unless the artifact's
// build-provenance.json names the checkout's HEAD commit, the coding-tooling
// pin in scripts/coding-tooling.rev and the single moenarch-foundation pin of
// .coding-tooling.source-deps.json.
import { execFileSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import { join, resolve } from "node:path";

const site = resolve(process.argv[2] ?? "_site");
const checkout = resolve(process.argv[3] ?? ".");
const commitId = /^[0-9a-f]{40}$/;

function fail(message) {
  console.error(`check-pages-provenance: ${message}`);
  process.exit(1);
}

let record;
try {
  record = JSON.parse(await readFile(join(site, "build-provenance.json"), "utf8"));
} catch (error) {
  fail(`cannot read ${join(site, "build-provenance.json")}: ${error.message}`);
}

const head = execFileSync("git", ["-C", checkout, "rev-parse", "HEAD"], { encoding: "utf8" }).trim();
const toolingPin = (await readFile(join(checkout, "scripts/coding-tooling.rev"), "utf8")).trim();
const manifest = JSON.parse(await readFile(join(checkout, ".coding-tooling.source-deps.json"), "utf8"));
const foundationPins = [...new Set(manifest.cargo.patches.map((patch) => patch.rev))];
if (foundationPins.length !== 1) fail(`expected one Foundation pin, found ${foundationPins.join(", ")}`);

const expected = {
  schema_version: 1,
  repository: "moritzbrantner/visual-analysis",
  commit: head,
  built_from: "clean-checkout",
  "source_inputs.coding-tooling": toolingPin,
  "source_inputs.moenarch-foundation": foundationPins[0],
};
const actual = {
  schema_version: record.schema_version,
  repository: record.repository,
  commit: record.commit,
  built_from: record.built_from,
  "source_inputs.coding-tooling": record.source_inputs?.["coding-tooling"],
  "source_inputs.moenarch-foundation": record.source_inputs?.["moenarch-foundation"],
};
for (const [key, value] of Object.entries(expected)) {
  if (actual[key] !== value) fail(`${key} is ${JSON.stringify(actual[key])}, expected ${JSON.stringify(value)}`);
}
for (const key of ["commit", "source_inputs.coding-tooling", "source_inputs.moenarch-foundation"]) {
  if (!commitId.test(actual[key])) fail(`${key} is not a full commit id`);
}
const extra = Object.keys(record.source_inputs ?? {}).filter(
  (name) => !["coding-tooling", "moenarch-foundation"].includes(name),
);
if (extra.length > 0) fail(`undeclared source inputs: ${extra.join(", ")}`);

console.log(
  `Pages artifact built from a clean checkout of ${head} with coding-tooling ${toolingPin} and moenarch-foundation ${foundationPins[0]}`,
);
