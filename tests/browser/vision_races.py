"""Real-browser request-isolation regressions with deferred model adapters.

This exercises the shipped UI and DOM; it does not claim model accuracy or GPU
acceptance. No models, media, or network services are needed by these fixtures.
"""
from __future__ import annotations
import base64
import os
from pathlib import Path
import unittest
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
ADAPTER = r'''
export const SAM_MODEL_ID = 'fixture-sam';
export const OPEN_VOCAB_MODEL_ID = 'fixture-detector';
export function browserVisionCapabilities() { return {webgpu: true}; }
window.audit = {calls: [], disposed: []};
function defer(kind, payload) {
  return new Promise((resolve, reject) => window.audit.calls.push({kind, payload, resolve, reject}));
}
export const prepareSamImage = imageUrl => defer('prepare', {imageUrl});
export const detectOpenVocabulary = imageUrl => defer('detect', {imageUrl});
export const segmentSamBox = (session, region) => defer('box', {imageUrl:session.imageUrl, region});
export const segmentSamPoints = (session, points) => defer('points', {imageUrl:session.imageUrl, points});
export const disposeSamImage = session => window.audit.disposed.push(session.imageUrl);
window.audit.finish = (index, label = 'current image') => {
  const call = window.audit.calls[index];
  if (call.kind === 'prepare') call.resolve({imageUrl:call.payload.imageUrl});
  else if (call.kind === 'detect') call.resolve([0,1].map(i => ({label, score:0.95,
    region:{x:80+i*260,y:70,width:180,height:180}})));
  else call.resolve({width:64,height:64,data:new Uint8Array(4096).fill(255),
    activePixels:4096,score:0.98,region:{x:0,y:0,width:64,height:64}});
};
'''

def module_url(source: str) -> str:
    return "data:text/javascript;base64," + base64.b64encode(source.encode()).decode()

def fixture_html() -> str:
    # Bundle the unchanged production module graph in memory. This allows the
    # same zero-network test to run in restricted local browsers and hosted CI.
    site = ROOT / "site"
    overlay = module_url((site / "vision-overlay.js").read_text())
    ui = module_url((site / "vision-ui.js").read_text()
        .replace('"./vision-models.js"', '"' + module_url(ADAPTER) + '"')
        .replace('"./vision-overlay.js"', '"' + overlay + '"'))
    analysis = module_url((site / "analysis.js").read_text().replace('"./vision-ui.js"', '"' + ui + '"'))
    app = module_url((site / "app.js").read_text().replace('"./analysis.js"', '"' + analysis + '"'))
    html = (site / "index.html").read_text()
    html = html.replace('<link rel="stylesheet" href="./styles.css" />',
        '<style>' + (site / "styles.css").read_text() + (site / "vision.css").read_text() + '</style>'
        + '<link href="./vision.css" data-fixture-inline="true" />')
    return html.replace('src="./app.js"', 'src="' + app + '"')

class VisionRaces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.playwright = sync_playwright().start()
        executable = os.environ.get("CHROMIUM_EXECUTABLE")
        if not executable and Path("/usr/bin/chromium").exists():
            executable = "/usr/bin/chromium"
        cls.browser = cls.playwright.chromium.launch(executable_path=executable, headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"])

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()

    def setUp(self):
        self.page = self.browser.new_page(viewport={"width":1200,"height":1100})
        self.errors = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.route("**/*", lambda route: route.abort())
        self.page.set_content(fixture_html())
        self.page.wait_for_selector("#learned-vision-section",state="attached")
        self.page.evaluate(r'''() => {
          window.audit.images = {};
          for (const [name,color] of [['A','#395869'],['B','#32684b']]) {
            const canvas = document.createElement('canvas'); canvas.width=720; canvas.height=420;
            const c=canvas.getContext('2d'); c.fillStyle=color; c.fillRect(0,0,720,420);
            c.fillStyle='white'; c.font='bold 36px sans-serif'; c.fillText(`Image ${name}`,70,180);
            c.font='22px sans-serif'; c.fillText('Deterministic request-isolation fixture',70,235);
            window.audit.images[name] = canvas.toDataURL();
          }
          document.getElementById('report').hidden=false;
          document.getElementById('input-panel').hidden=true;
          document.getElementById('report-title').textContent='Request isolation regression';
        }''')
        self.select("A")

    def tearDown(self):
        self.assertEqual(self.errors, [])
        self.page.close()

    def select(self, name):
        self.page.evaluate("""async name => {
          const image=document.getElementById('preview-image');
          image.src=window.audit.images[name]; image.hidden=false; await image.decode();
        }""", name)
        self.page.wait_for_function("!document.getElementById('prepare-sam').disabled")

    def start(self, button, kind):
        count = self.page.evaluate("window.audit.calls.length")
        self.page.locator(f"#{button}").click()
        self.page.wait_for_function("count => window.audit.calls.length > count",arg=count)
        self.assertEqual(self.page.evaluate("i => window.audit.calls[i].kind", count),kind)
        return count

    def finish(self, index, label="current image"):
        self.page.evaluate("([index,label]) => window.audit.finish(index,label)",[index,label])
        self.page.evaluate("() => new Promise(resolve => setTimeout(resolve, 0))")

    def prepare(self):
        index=self.start("prepare-sam","prepare"); self.finish(index)
        self.page.wait_for_function("document.getElementById('vision-overlay').classList.contains('is-sam-ready')")

    def point(self):
        count=self.page.evaluate("window.audit.calls.length")
        self.page.locator("#vision-overlay").click(position={"x":120,"y":120})
        self.page.wait_for_function("count => window.audit.calls.length > count",arg=count)
        return count

    def test_old_detection_cannot_draw_or_unlock_a_new_request(self):
        old=self.start("detect-concepts","detect")
        self.select("B")
        new=self.start("detect-concepts","detect")
        self.finish(old,"WRONG IMAGE A")
        self.assertTrue(self.page.locator("#detect-concepts").is_disabled())
        self.assertNotIn("WRONG",self.page.locator("#vision-results").inner_text())
        self.finish(new,"Image B result")
        self.assertIn("Image B result",self.page.locator("#vision-results").inner_text())
        self.assertFalse(self.page.locator("#detect-concepts").is_disabled())
        output=Path(os.environ.get("VISUAL_EVIDENCE_DIR",ROOT/".artifacts/browser")); output.mkdir(parents=True,exist_ok=True)
        self.page.evaluate('window.scrollTo(0, document.getElementById("preview-section").offsetTop - 70)')
        self.page.screenshot(path=str(output/"vision-request-isolation.png"))

    def test_reselecting_same_url_does_not_revive_an_old_generation(self):
        old=self.start("detect-concepts","detect")
        self.select("B"); self.select("A")
        new=self.start("detect-concepts","detect")
        self.finish(old,"STALE A")
        self.assertNotIn("STALE",self.page.locator("#vision-results").inner_text())
        self.assertTrue(self.page.locator("#detect-concepts").is_disabled())
        self.finish(new,"Fresh A")
        self.assertIn("Fresh A",self.page.locator("#vision-results").inner_text())

    def test_stale_error_does_not_replace_current_status(self):
        old=self.start("detect-concepts","detect")
        self.select("B"); new=self.start("detect-concepts","detect")
        self.page.evaluate("index => window.audit.calls[index].reject(new Error('STALE FAILURE'))",old)
        self.assertNotIn("STALE",self.page.locator("#learned-vision-status").inner_text())
        self.assertTrue(self.page.locator("#detect-concepts").is_disabled())
        self.finish(new)

    def test_late_embedding_is_disposed_not_installed_on_new_image(self):
        old=self.start("prepare-sam","prepare")
        self.select("B"); new=self.start("prepare-sam","prepare")
        self.finish(old)
        self.assertNotIn("is-sam-ready",self.page.locator("#vision-overlay").get_attribute("class") or "")
        self.assertTrue(self.page.locator("#prepare-sam").is_disabled())
        self.assertEqual(self.page.evaluate("window.audit.disposed"),[self.page.evaluate("window.audit.images.A")])
        self.finish(new)
        point=self.point()
        self.assertEqual(self.page.evaluate("i => window.audit.calls[i].payload.imageUrl",point),self.page.evaluate("window.audit.images.B"))
        self.finish(point)

    def test_inflight_mask_keeps_lease_then_discards_old_points(self):
        self.prepare(); old=self.point()
        self.select("B")
        self.assertEqual(self.page.evaluate("window.audit.disposed.length"),0,"in-use tensors were disposed early")
        self.prepare(); self.finish(old)
        self.assertEqual(self.page.evaluate("window.audit.disposed"),[self.page.evaluate("window.audit.images.A")])
        new=self.point()
        self.assertEqual(self.page.evaluate("i => window.audit.calls[i].payload.points.length",new),1)
        self.finish(new)
        self.assertIn("1 prompt point",self.page.locator("#vision-results").inner_text())

    def test_old_box_refinement_stops_before_starting_another_box(self):
        self.prepare(); detection=self.start("detect-concepts","detect"); self.finish(detection)
        old=self.start("refine-detections","box")
        self.select("B"); self.prepare()
        count=self.page.evaluate("window.audit.calls.length")
        self.finish(old)
        self.assertEqual(self.page.evaluate("window.audit.calls.length"),count)
        self.assertEqual(self.page.locator("#vision-results strong").inner_text(),"SAM ready")

    def test_choose_another_invalidates_a_pending_operation(self):
        old=self.start("prepare-sam","prepare")
        self.page.locator("#choose-another").click()
        self.finish(old)
        self.assertTrue(self.page.locator("#preview-image").is_hidden())
        self.assertTrue(self.page.locator("#learned-vision-section").is_hidden())
        self.assertEqual(self.page.evaluate("window.audit.disposed.length"),1)

if __name__ == "__main__":
    unittest.main(verbosity=2)
