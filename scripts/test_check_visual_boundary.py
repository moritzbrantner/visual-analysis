#!/usr/bin/env python3
"""Tests for scripts/check-visual-boundary.py against the real repository and
synthetic copies that violate one boundary rule each."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("check-visual-boundary.py")
SPEC = importlib.util.spec_from_file_location("check_visual_boundary", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
boundary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = boundary
SPEC.loader.exec_module(boundary)

ROOT = boundary.ROOT


class VisualBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name) / "visual-analysis"
        self.root.mkdir()
        for relative in ("Cargo.toml", "package.json"):
            shutil.copy2(ROOT / relative, self.root / relative)
        shutil.copytree(ROOT / "docs/ownership", self.root / "docs/ownership")
        for manifest in sorted((ROOT / "crates").rglob("Cargo.toml")):
            target = self.root / manifest.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(manifest, target)
        for manifest in sorted((ROOT / "packages").glob("*/package.json")):
            target = self.root / manifest.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(manifest, target)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def manifest(self, family: str, crate: str) -> Path:
        return self.root / "crates" / family / crate / "Cargo.toml"

    def append_dependency(self, family: str, crate: str, line: str) -> None:
        path = self.manifest(family, crate)
        text = path.read_text()
        self.assertIn("[dependencies]\n", text)
        path.write_text(text.replace("[dependencies]\n", f"[dependencies]\n{line}\n", 1))

    def contract(self) -> dict:
        return json.loads((self.root / boundary.CONTRACT_RELATIVE).read_text())

    def write_contract(self, document: dict) -> None:
        (self.root / boundary.CONTRACT_RELATIVE).write_text(json.dumps(document, indent=2))

    def test_repository_satisfies_its_boundary(self) -> None:
        self.assertEqual(boundary.validate(ROOT), [])
        self.assertEqual(boundary.validate(self.root), [])

    def test_scene_crates_other_than_the_seam_are_rejected(self) -> None:
        self.append_dependency(
            "image", "image-analysis-core", 'scene-cli = { package = "scenedetect-cli", version = "0.1" }'
        )
        errors = boundary.validate(self.root)
        self.assertTrue(any("depends on scenedetect-cli" in e for e in errors), errors)

    def test_scene_seam_is_limited_to_the_declared_adapter_crates(self) -> None:
        self.append_dependency("image", "image-analysis-ocr", 'scenedetect-core = { workspace = true }')
        errors = boundary.validate(self.root)
        self.assertTrue(
            any("moenarch-image-analysis-ocr depends on scenedetect-core outside" in e for e in errors),
            errors,
        )

    def test_corpus_and_product_repositories_are_rejected(self) -> None:
        self.append_dependency("video", "video-analysis-storage", 'youtube-corpus = "0.1"')
        cargo = self.root / "Cargo.toml"
        cargo.write_text(
            cargo.read_text().replace(
                "[workspace.dependencies]\n",
                '[workspace.dependencies]\nmi = { package = "media-intelligence-core", version = "0.1" }\n',
                1,
            )
        )
        errors = boundary.validate(self.root)
        self.assertTrue(any("depends on youtube-corpus" in e for e in errors), errors)
        self.assertTrue(any("[workspace] depends on media-intelligence-core" in e for e in errors), errors)

    def test_target_specific_dependencies_are_checked(self) -> None:
        path = self.manifest("video", "video-analysis-output")
        path.write_text(
            path.read_text() + "\n[target.'cfg(unix)'.dependencies]\nvideo-to-3d-core = \"0.1\"\n"
        )
        errors = boundary.validate(self.root)
        self.assertTrue(any("depends on video-to-3d-core" in e for e in errors), errors)

    def test_in_tree_path_dependencies_outside_members_are_checked(self) -> None:
        helper = self.root / "tools" / "helper"
        helper.mkdir(parents=True)
        (helper / "Cargo.toml").write_text(
            '[package]\nname = "visual-helper"\nversion = "0.1.0"\nedition = "2021"\n\n'
            '[dependencies]\nyoutube-corpus = "0.1"\n'
        )
        self.append_dependency(
            "image", "image-analysis-io", 'visual-helper = { path = "../../../tools/helper" }'
        )
        errors = boundary.validate(self.root)
        self.assertTrue(any("visual-helper depends on youtube-corpus" in e for e in errors), errors)

    def test_bun_packages_cannot_depend_on_products(self) -> None:
        path = sorted((self.root / "packages").glob("*/package.json"))[0]
        document = json.loads(path.read_text())
        document.setdefault("dependencies", {})["@moritzbrantner/interactive-videos"] = "1.0.0"
        path.write_text(json.dumps(document))
        errors = boundary.validate(self.root)
        self.assertTrue(any("excluded package @moritzbrantner/interactive-videos" in e for e in errors), errors)

    def test_contract_edits_cannot_widen_the_scene_seam(self) -> None:
        document = self.contract()
        document["sceneSeam"]["adapterCrates"].append("moenarch-image-analysis-ocr")
        self.write_contract(document)
        self.append_dependency("image", "image-analysis-ocr", 'scenedetect-core = { workspace = true }')
        errors = boundary.validate(self.root)
        self.assertTrue(any("adapter crates must be exactly" in e for e in errors), errors)
        self.assertTrue(any("outside the declared scene adapter seam" in e for e in errors), errors)

    def test_excluded_authorities_cannot_be_dropped(self) -> None:
        document = self.contract()
        document["excludedAuthorities"] = [
            authority
            for authority in document["excludedAuthorities"]
            if authority["authority"] != "canonical-scene-boundary-algorithms"
        ]
        self.write_contract(document)
        errors = boundary.validate(self.root)
        self.assertTrue(any("must exclude" in e for e in errors), errors)


if __name__ == "__main__":
    unittest.main()
