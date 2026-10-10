"""Browser acceptance for the built GitHub Pages Visual Inspector artifact.

Issue #67: the Pages artifact must work from its own files alone. This suite
serves a *built* artifact directory (``PAGES_ARTIFACT_DIR``, default
``_site``) from a local static server under a project-style sub-path, blocks
every request that does not target that server, and drives representative
image and video workbench journeys with the committed fixtures in
``tests/fixtures/pages``. It never starts a backend and never downloads models.

Run after building the artifact, e.g.::

    bash scripts/check-pages-clean-checkout.sh _site
    python3 tests/browser/pages_workbench.py

The suite runs sequentially in one browser (a single worker).
"""
from __future__ import annotations

import functools
import hashlib
import http.server
import json
import os
import re
import threading
import unittest
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "pages"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())
ARTIFACT = Path(os.environ.get("PAGES_ARTIFACT_DIR", ROOT / "_site")).resolve()
BASE_PATH = "/visual-analysis/"  # GitHub Pages serves project sites below the repository name.
TIMEOUT_MS = 30_000

IMAGE = FIXTURES / MANIFEST["image"]["file"]
VIDEO = FIXTURES / MANIFEST["video"]["file"]

# Browser-model (learned vision) controls. They run on Pages in a capable
# browser by loading a declared external runtime (transformers.js pinned to an
# exact version on jsDelivr) and model weights from Hugging Face. These are the
# only off-origin requests the artifact may make, and only after the user
# chooses such a control. This suite still blocks them: it must then show an
# explicit, visible failed-to-load/unavailable state that names the feature.
# Note: the transformers.js runtime is version-pinned (@3.5.0 in
# site/vision-models.js), but the Hugging Face model weights are fetched without
# a pinned revision -- tracked as a follow-up, not asserted here.
BROWSER_MODEL_CONTROLS = {
    "detect-concepts": re.compile(r"detect", re.IGNORECASE),
    "prepare-sam": re.compile(r"\bSAM\b|segment", re.IGNORECASE),
    "refine-detections": re.compile(r"\bSAM\b|segment|refine", re.IGNORECASE),
}
DECLARED_BROWSER_MODEL_HOSTS = ("cdn.jsdelivr.net", "huggingface.co", "hf.co")
LOAD_FAILURE_TEXT = re.compile(
    r"unavailable|not available|(?:could not|couldn't|cannot|can't|failed to|unable to) (?:be )?load",
    re.IGNORECASE,
)
# Capabilities that need native, ONNX or server backends cannot run on Pages.
NATIVE_BACKEND_TEXT = re.compile(r"native|onnx|server|\bDETR\b|yunet|cli\b", re.IGNORECASE)
UNAVAILABLE_TEXT = re.compile(r"unavailable|not available", re.IGNORECASE)

# Regression golden for the WASM perceptual hash (image.processing.hash,
# hashSize 8) of the committed image fixture and of a uniform video frame.
IMAGE_HASH = "9144009100440091"
UNIFORM_FRAME_HASH = "8000000000000000"


def expected_image_report() -> dict:
    """Derive the image expectations from the fixture's pixel definition.

    Luma is the documented BT.601 integer luma of image-analysis-core; the page
    shows a 32-bin histogram and bin-centre statistics (site/analysis.js).
    """
    regions = MANIFEST["image"]["regions"]
    total = sum(region["pixels"] for region in regions)
    histogram = [0] * 32
    for region in regions:
        red, green, blue = region["rgb"]
        luma = round(0.299 * red + 0.587 * green + 0.114 * blue)
        histogram[min(luma * 32, 255 * 32) // 256] += region["pixels"]
    mean = [sum(region["rgb"][channel] * region["pixels"] for region in regions) / total for channel in range(3)]
    centres = [index * 8 + 4 for index in range(32)]
    mean_luma = sum(count * centre for count, centre in zip(histogram, centres)) / total

    def quantile(q: float) -> float:
        seen = 0
        for index, count in enumerate(histogram):
            seen += count
            if seen >= q * total:
                return (index + 0.5) * 8
        return 255

    return {
        "histogram": histogram,
        "meanRgb": {"red": mean[0], "green": mean[1], "blue": mean[2]},
        "hex": "#" + "".join(f"{max(0, min(255, int(value + 0.5))):02x}" for value in mean),
        "meanLuma": mean_luma,
        "p10": quantile(0.1),
        "p50": quantile(0.5),
        "p90": quantile(0.9),
    }


def expected_video_timeline() -> list[float]:
    """Mean luma (bin centre) the page must report for each of its 9 samples."""
    video = MANIFEST["video"]
    duration = video["durationSeconds"]
    edge = min(0.05, duration * 0.01)
    times = [edge + (duration - 2 * edge) * index / 8 for index in range(9)]
    boundaries, greys, start = [], [], 0.0
    for segment in video["segments"]:
        start += segment["frames"] / video["fps"]
        boundaries.append(start)
        greys.append(segment["grey"])
    values = []
    for time in times:
        grey = next(grey for boundary, grey in zip(boundaries, greys) if time < boundary)
        values.append((grey // 8) * 8 + 4)
    return values


def format_bytes(size: int) -> str:
    """The page's documented byte formatting for files below 100 KB."""
    assert size < 100 * 1024, "Pages fixtures must stay tiny"
    if size < 1024:
        return f"{size} B"
    scaled = size / 1024
    return f"{scaled:.{1 if scaled >= 10 else 2}f} KB"


class _Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".wasm": "application/wasm",
        ".json": "application/json",
    }

    def translate_path(self, path):
        parsed = urlsplit(path).path
        if not parsed.startswith(BASE_PATH):
            return str(ARTIFACT / "__outside_pages_base_path__")
        return super().translate_path("/" + parsed[len(BASE_PATH):])

    def log_message(self, *_args):
        pass


class PagesWorkbench(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ARTIFACT / "index.html").is_file():
            raise RuntimeError(f"Pages artifact not found at {ARTIFACT}; build it first (PAGES_ARTIFACT_DIR).")
        cls.server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), functools.partial(_Handler, directory=str(ARTIFACT))
        )
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.origin = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.url = cls.origin + BASE_PATH
        cls.playwright = sync_playwright().start()
        executable = os.environ.get("CHROMIUM_EXECUTABLE") or None
        cls.browser = cls.playwright.chromium.launch(
            executable_path=executable,
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--force-color-profile=srgb"],
        )

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.context = self.browser.new_context(
            viewport={"width": 1200, "height": 1000}, accept_downloads=True, service_workers="block"
        )
        self.blocked: list[str] = []  # undeclared off-origin requests
        self.declared_external: list[str] = []  # declared browser-model runtime/weights (blocked)
        self.responses: list = []
        self.page_errors: list[str] = []
        self.context.route("**/*", self._route)
        self.page = self.context.new_page()
        self.page.set_default_timeout(TIMEOUT_MS)
        self.page.on("pageerror", lambda error: self.page_errors.append(str(error)))
        self.page.on("response", lambda response: self.responses.append(response))

    def tearDown(self):
        self.context.close()
        self.assertEqual(self.blocked, [], "the Pages artifact requested resources outside its own origin")
        self.assertEqual(self.page_errors, [], "uncaught page errors")

    def _route(self, route):
        url = route.request.url
        if url.startswith(self.origin + "/") or url.startswith(("data:", "blob:")):
            route.continue_()
        else:
            host = urlsplit(url).hostname or ""
            declared = any(host == item or host.endswith("." + item) for item in DECLARED_BROWSER_MODEL_HOSTS)
            (self.declared_external if declared else self.blocked).append(url)
            route.abort()

    # -- helpers ---------------------------------------------------------

    def open(self):
        self.page.goto(self.url)
        self.page.wait_for_selector("#file-input", state="attached")

    def analyze(self, fixture: Path):
        self.page.locator("#file-input").set_input_files(str(fixture))
        self.page.wait_for_function(
            """() => !document.getElementById('report').hidden
              || !document.getElementById('input-error').hidden"""
        )
        error = self.page.locator("#input-error")
        if error.is_visible():
            self.fail(f"analysis of {fixture.name} failed in the page: {error.inner_text()}")

    def text(self, selector: str) -> str:
        return self.page.locator(selector).inner_text().strip()

    def raw_report(self) -> dict:
        return json.loads(self.page.locator("#raw-json").text_content())

    # -- fixtures --------------------------------------------------------

    def test_fixtures_match_manifest(self):
        for entry in (MANIFEST["image"], MANIFEST["video"]):
            digest = hashlib.sha256((FIXTURES / entry["file"]).read_bytes()).hexdigest()
            self.assertEqual(digest, entry["sha256"], entry["file"])

    # -- image journey ---------------------------------------------------

    def test_image_workbench_reports_fixture_statistics_via_wasm(self):
        expected = expected_image_report()
        self.open()
        self.analyze(IMAGE)

        # WASM runtime is loaded from the artifact itself.
        self.assertEqual(
            self.text("#runtime-summary"), "Rust/WASM image core and processing runtimes are ready."
        )
        wasm = [response for response in self.responses if urlsplit(response.url).path.endswith(".wasm")]
        self.assertEqual(len(wasm), 2, [response.url for response in wasm])
        for response in wasm:
            self.assertTrue(response.url.startswith(self.url + "wasm/"), response.url)
            self.assertEqual(response.status, 200, response.url)
            self.assertEqual(response.headers.get("content-type"), "application/wasm")

        self.assertEqual(self.text("#report-title"), IMAGE.name)
        self.assertEqual(self.text("#summary-kind"), "Image")
        self.assertEqual(self.text("#summary-dimensions"), "16×16")
        self.assertEqual(self.text("#summary-analyzed"), "16×16")
        self.assertEqual(self.text("#summary-color"), expected["hex"])
        self.assertEqual(self.text("#summary-luma"), f"{expected['meanLuma']:.1f} / 255")
        self.assertEqual(self.text("#summary-hash"), IMAGE_HASH)
        self.assertEqual(self.text("#file-meta"), f"image/png · {format_bytes(IMAGE.stat().st_size)}")
        self.assertEqual(self.text("#coverage-badge"), "full decoded image")
        self.assertEqual(
            self.page.locator("#histogram-readout span").all_inner_texts(),
            [
                f"10th percentile ≈ {expected['p10']:.0f}",
                f"median ≈ {expected['p50']:.0f}",
                f"90th percentile ≈ {expected['p90']:.0f}",
                f"spread ≈ {expected['p90'] - expected['p10']:.0f}",
            ],
        )
        findings = self.page.locator("#findings .finding").all_inner_texts()
        self.assertEqual(
            findings,
            [
                "Overall brightness is centered in the mid-range.",
                "The image spans a broad luma range, with strong dark-to-bright separation.",
            ],
        )
        self.assertTrue(self.page.locator("#timeline-section").is_hidden())
        self.assertTrue(self.page.locator("#preview-image").is_visible())
        self.assertEqual(self.page.locator("#preview-image").evaluate("img => img.naturalWidth"), 16)

        report = self.raw_report()
        self.assertEqual(report["media"]["kind"], "image")
        self.assertEqual(report["analysis"]["histogram"], expected["histogram"])
        self.assertEqual(report["analysis"]["summary"]["meanRgb"], expected["meanRgb"])
        self.assertEqual(report["analysis"]["summary"]["width"], 16)
        self.assertEqual(report["analysis"]["perceptualHash"], {"hash": IMAGE_HASH, "hashSize": 8})

        # The JSON export is the same report the page rendered.
        with self.page.expect_download() as download_info:
            self.page.locator("#export-json").click()
        download = download_info.value
        self.assertEqual(download.suggested_filename, "workbench-quadrants.visual-analysis.json")
        self.assertEqual(json.loads(Path(download.path()).read_text()), report)

        # Returning to the input panel clears the report.
        self.page.locator("#choose-another").click()
        self.page.wait_for_selector("#input-panel:not([hidden])")
        self.assertTrue(self.page.locator("#report").is_hidden())

    # -- video journey ---------------------------------------------------

    def test_video_workbench_samples_fixture_frames(self):
        video = MANIFEST["video"]
        timeline = expected_video_timeline()
        self.open()
        self.analyze(VIDEO)

        self.assertEqual(self.text("#summary-kind"), "Video")
        self.assertEqual(self.text("#summary-dimensions"), f"{video['width']}×{video['height']}")
        self.assertEqual(self.text("#summary-analyzed"), "9 frames")
        self.assertEqual(self.text("#coverage-badge"), "9 sampled frames")
        self.assertEqual(
            self.text("#file-meta"),
            f"video/webm · {format_bytes(VIDEO.stat().st_size)} · {video['durationSeconds']:.1f} s",
        )
        self.assertTrue(self.page.locator("#timeline-section").is_visible())
        self.assertEqual(self.text("#timeline-badge"), "9 evenly spaced frames")
        self.assertTrue(self.page.locator("#preview-video").is_visible())

        report = self.raw_report()
        self.assertEqual([sample["meanLuma"] for sample in report["timeline"]], timeline)
        self.assertAlmostEqual(report["media"]["durationSeconds"], video["durationSeconds"], places=2)
        middle = timeline[4]
        self.assertEqual(self.text("#summary-luma"), f"{middle:.1f} / 255")
        for channel, value in report["analysis"]["summary"]["meanRgb"].items():
            self.assertLessEqual(abs(value - (middle - 0.5)), 4, channel)
        self.assertEqual(report["analysis"]["perceptualHash"], {"hash": UNIFORM_FRAME_HASH, "hashSize": 8})
        self.assertIn(
            "Sampled frames show large brightness changes across the video.",
            self.page.locator("#findings .finding").all_inner_texts(),
        )

    # -- model-backed features ------------------------------------------

    def test_native_backend_capabilities_have_no_enabled_controls_on_pages(self):
        """Capabilities needing native/ONNX/server backends cannot run on Pages.

        Where the page mentions them it must not offer an enabled control for
        them; any control naming such a backend must be disabled and explained.
        """
        self.open()
        self.analyze(IMAGE)
        self.page.wait_for_timeout(250)
        offending = self.page.evaluate(
            """() => {
              const enabled = node => !node.disabled && node.getClientRects().length > 0;
              const out = [];
              for (const card of document.querySelectorAll('#capabilities .capability-card')) {
                const status = card.querySelector('.capability-status');
                if (status?.textContent.trim() === 'Runs here') continue;
                for (const node of card.querySelectorAll('button, input, select, textarea')) {
                  if (enabled(node)) out.push(card.querySelector('h3')?.textContent + ': ' + (node.id || node.textContent));
                }
              }
              return out;
            }"""
        )
        self.assertEqual(offending, [], "capabilities that do not run on Pages expose enabled controls")
        for control in self.page.locator("button").all():
            label = control.inner_text()
            if NATIVE_BACKEND_TEXT.search(label) and control.is_visible():
                self.assertTrue(control.is_disabled(), f"native-backend control {label!r} is enabled on Pages")
                self.assertTrue(
                    self.page.get_by_text(UNAVAILABLE_TEXT).count() > 0,
                    f"native-backend control {label!r} lacks an explicit unavailable explanation",
                )

    def test_browser_model_controls_report_runtime_load_failure_explicitly(self):
        """With the declared external runtime/weights unreachable, each enabled
        browser-model control must end in a visible state that names the
        feature and says it is unavailable / failed to load -- not a silent
        no-op, a raw technical error or an unhandled exception."""
        problems = []
        for control_id, feature in BROWSER_MODEL_CONTROLS.items():
            self.open()
            self.analyze(IMAGE)
            self.page.wait_for_timeout(250)
            control = self.page.locator(f"#{control_id}")
            if control.count() == 0 or not control.is_visible():
                continue
            section = self.page.locator("#learned-vision-section")
            if control.is_disabled():
                # Disabled up front (e.g. no WebGPU, or no detections yet to
                # refine): fine, as long as SAM/detection unavailability is
                # explained when the precondition is the runtime itself.
                if control_id == "refine-detections":
                    continue
                visible = [t for t in section.get_by_text(feature).all_inner_texts() if t.strip()]
                if not any(LOAD_FAILURE_TEXT.search(t) or re.search(r"disabled|requires", t, re.I) for t in visible):
                    problems.append(f"#{control_id} is disabled without a visible explanation naming the feature")
                continue
            control.click()
            try:
                self.page.wait_for_function(
                    """() => {
                      const status = document.getElementById('learned-vision-status');
                      const busy = Array.from(document.querySelectorAll('#learned-vision-section button'))
                        .some(button => button.disabled && button.id !== 'refine-detections');
                      return status && !/^Loading|^Decoding|^Refining/.test(status.textContent.trim()) && !busy;
                    }""",
                    timeout=15_000,
                )
            except Exception:  # noqa: BLE001 - reported through the state assertions below
                pass
            self.page.wait_for_timeout(250)
            status = self.page.locator("#learned-vision-status")
            status_text = status.inner_text().strip() if status.count() and status.is_visible() else ""
            candidates = [status_text] + [
                t for t in section.get_by_text(LOAD_FAILURE_TEXT).all_inner_texts() if t.strip()
            ]
            if not any(LOAD_FAILURE_TEXT.search(t) and feature.search(t) for t in candidates if t):
                problems.append(
                    f"#{control_id}: no visible unavailable/failed-to-load message naming the feature "
                    f"(status: {status_text!r})"
                )
        self.assertEqual(problems, [])

    # -- every enabled control does something ---------------------------

    def _enabled_controls(self) -> list[str]:
        return self.page.evaluate(
            """() => Array.from(document.querySelectorAll('button, summary'))
                .filter(node => !node.disabled && node.getClientRects().length > 0
                  && getComputedStyle(node).visibility !== 'hidden')
                .map(node => node.id ? '#' + node.id
                  : node.dataset.example ? `[data-example="${node.dataset.example}"]`
                  : node.closest('details[id]') ? `#${node.closest('details[id]').id} > summary`
                  : null)"""
        )

    def _assert_click_has_effect(self, selector: str):
        events = {"download": 0, "filechooser": 0}
        self.page.on("download", lambda _event: events.__setitem__("download", events["download"] + 1))
        self.page.on("filechooser", lambda _event: events.__setitem__("filechooser", events["filechooser"] + 1))
        self.page.evaluate(
            """() => {
              window.__acceptanceMutations = 0;
              new MutationObserver(records => { window.__acceptanceMutations += records.length; })
                .observe(document.documentElement, {subtree: true, childList: true, attributes: true, characterData: true});
            }"""
        )
        blocked_before = len(self.blocked)
        self.page.locator(selector).click()
        try:
            self.page.wait_for_function("() => window.__acceptanceMutations > 0", timeout=3_000)
        except Exception:  # noqa: BLE001 - fall through to the event check below
            pass
        mutated = self.page.evaluate("() => window.__acceptanceMutations")
        self.assertTrue(
            mutated or events["download"] or events["filechooser"],
            f"enabled control {selector} had no observable effect",
        )
        self.page.wait_for_timeout(500)  # let asynchronous work (and any off-origin request) surface
        self.assertEqual(
            self.blocked[blocked_before:],
            [],
            f"enabled control {selector} depends on undeclared resources outside the artifact",
        )

    def test_every_enabled_control_has_an_effect(self):
        stages = {
            "landing": lambda: None,
            "image report": lambda: self.analyze(IMAGE),
        }
        for stage, prepare in stages.items():
            self.open()
            prepare()
            self.page.wait_for_timeout(250)
            selectors = self._enabled_controls()
            self.assertNotIn(None, selectors, f"unidentifiable enabled control on {stage}")
            for selector in selectors:
                with self.subTest(stage=stage, control=selector):
                    self.open()
                    prepare()
                    self.page.wait_for_timeout(250)
                    self._assert_click_has_effect(selector)
        self.blocked.clear()  # each offending control was already reported by its subtest


if __name__ == "__main__":
    if not (ARTIFACT / "index.html").is_file():
        raise SystemExit(f"Pages artifact not found at {ARTIFACT}; build it first (PAGES_ARTIFACT_DIR).")
    unittest.main(verbosity=2)
