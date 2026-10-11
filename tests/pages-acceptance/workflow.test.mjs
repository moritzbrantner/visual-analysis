// Issue #67: the Pages browser acceptance must run as a required check that
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

// Issue #69: pages.yml must deploy the artifact built from a clean clone that
// fetches only the declared pins (scripts/check-pages-clean-checkout.sh), never
// one built in the workflow's own checkout against prepared siblings.
test("Pages deploys only the clean-checkout artifact with verified provenance", async () => {
  const source = await workflow("pages.yml");
  assert.deepEqual(siblingCheckoutViolations(source), [], "pages.yml must not prepare sibling source workspaces itself");
  const jobs = source.slice(source.search(/^jobs:/m));
  assert.doesNotMatch(jobs, /run:[^\n]*(?:bootstrap-source-deps|scripts\/check-pages\.sh)/, "pages.yml must not build in its own checkout");
  const build = source.slice(source.search(/^ {2}build:/m), source.search(/^ {2}deploy:/m));
  const clean = build.search(/bash scripts\/check-pages-clean-checkout\.sh _site\n/);
  const provenance = build.search(/node scripts\/check-pages-provenance\.mjs _site \.\n/);
  const upload = build.search(/uses:\s*actions\/upload-pages-artifact@/);
  assert.ok(clean >= 0, "the build job must build _site from a clean checkout");
  assert.ok(provenance > clean, "the build job must verify the artifact's provenance against this commit");
  assert.ok(upload > provenance, "the uploaded artifact must be the verified one");
  assert.match(build.slice(upload), /with:\s*\n\s*path:\s*_site\s*\n/, "the uploaded path must be the clean-checkout _site");
  assert.match(source, /^ {2}deploy:[\s\S]*needs:\s*build/m, "deploy must use the build job's artifact");
});

test("Pages acceptance tests the artifact the deployed clean-checkout path builds", async () => {
  const source = await workflow("pages-acceptance.yml");
  const inPlace = source.indexOf("bash scripts/check-pages.sh _site");
  const clean = source.search(/bash scripts\/check-pages-clean-checkout\.sh _site\n/);
  const provenance = source.search(/node scripts\/check-pages-provenance\.mjs _site \.\n/);
  const browser = source.indexOf("tests/browser/pages_workbench.py");
  assert.ok(inPlace >= 0 && clean > inPlace, "rebuild _site through the clean-checkout path after the in-place check");
  assert.ok(provenance > clean, "verify the rebuilt artifact's provenance against this commit");
  assert.ok(browser > provenance, "the browser suite must run against the clean-checkout artifact");
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
  assert.match(jobs, /bash scripts\/check-pages\.sh _site/, "the job must build and verify the Pages artifact");
  assert.match(jobs, /tests\/browser\/pages_workbench\.py/, "the job must run the browser acceptance suite");
  assert.match(jobs, /node --test tests\/pages\/\*\.test\.mjs tests\/pages-acceptance\/\*\.test\.mjs/, "the job must run the Pages and Pages-acceptance node tests");
  assert.match(jobs, /PAGES_ARTIFACT_DIR:\s*_site/, "the browser suite must test the artifact the job built");
});
