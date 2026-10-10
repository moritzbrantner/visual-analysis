// Issue #67: GitHub Pages must build from the repository alone, using declared
// dependencies, and its browser acceptance must run as a required check that
// is never skipped. These are static checks over the committed workflows.
import test from "node:test";
import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";

const workflowDir = new URL("../../.github/workflows/", import.meta.url);

async function workflow(name) {
  return readFile(new URL(name, workflowDir), "utf8");
}

function steps(source) {
  // Returns every `run:` body and `uses:` reference (with its `with:` block).
  return source.split(/\n\s*- (?=name:|uses:|run:)/).slice(1);
}

const SIBLING_CHECKOUT_PATTERNS = [
  [/git\s+(?:-C\s+\S+\s+)?(?:init|clone|fetch)\b[^\n]*(?:\.\.\/|https?:\/\/)/, "clones or fetches another repository"],
  [/(?:^|[\s"'=(])\.\.\//m, "reads or writes a path outside the checkout (../)"],
  [/uses:\s*actions\/checkout@[^\n]*\n(?:\s+#[^\n]*\n)*\s+with:[\s\S]*?\brepository:/, "checks out a second repository"],
  [/CODING_TOOLING_DIR|source-deps\s+activate/, "activates sibling source workspaces"],
];

function siblingCheckoutViolations(source) {
  const violations = [];
  for (const step of steps(source)) {
    for (const [pattern, reason] of SIBLING_CHECKOUT_PATTERNS) {
      if (pattern.test(step)) violations.push(`${reason}: ${step.split("\n")[0].trim()}`);
    }
  }
  return violations;
}

test("the Pages workflow builds from its own checkout without undeclared sibling repositories", async () => {
  const source = await workflow("pages.yml");
  assert.deepEqual(siblingCheckoutViolations(source), []);
});

test("Pages acceptance runs on every pull request under a stable, never-skipped check name", async () => {
  const names = await readdir(workflowDir);
  const candidates = [];
  for (const name of names.filter((entry) => /\.ya?ml$/.test(entry))) {
    const source = await workflow(name);
    if (/^\s+name:\s*["']?Pages acceptance["']?\s*$/m.test(source)) candidates.push([name, source]);
  }
  assert.equal(candidates.length, 1, "exactly one workflow defines the `Pages acceptance` job");
  const [name, source] = candidates[0];

  const on = source.match(/^on:\n((?:[ \t]+.*\n|\n)*)/m)?.[1] ?? "";
  assert.match(on, /^ {2}pull_request:/m, `${name} must run on pull_request`);
  assert.doesNotMatch(on, /\bpaths(?:-ignore)?:/, `${name} must not be path-filtered`);
  const pullRequest = on.match(/^ {2}pull_request:.*\n((?: {4,}.*\n)*)/m)?.[1] ?? "";
  assert.doesNotMatch(pullRequest, /branches(?:-ignore)?:/, `${name} must run for pull requests into any branch`);

  const jobs = source.slice(source.search(/^jobs:/m));
  assert.doesNotMatch(jobs, /^ {4}if:/m, "the Pages acceptance job must not be conditional");
  assert.doesNotMatch(jobs, /continue-on-error:\s*true/, "acceptance failures must fail the check");
  assert.match(jobs, /bash scripts\/check-pages-clean-checkout\.sh/, "the job must build from a clean checkout");
  assert.match(jobs, /tests\/browser\/pages_workbench\.py/, "the job must run the browser acceptance suite");
  assert.match(jobs, /node --test tests\/pages\/\*\.test\.mjs tests\/pages-acceptance\/\*\.test\.mjs/, "the job must run the Pages and Pages-acceptance node tests");
  assert.deepEqual(siblingCheckoutViolations(source), []);
});
