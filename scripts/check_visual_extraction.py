#!/usr/bin/env python3
"""Structural-only verification for the visual-analysis bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "b8b29cf8db0b86ed1b133a18155adf24992f9483"
EXPECTED_EXTERNAL = {
    "audio-contracts": ("moenarch-audio-contracts", "=0.1.0"),
    "data-inversion-core": ("moenarch-data-inversion-core", "=0.1.1"),
    "math-geometry-2d": ("moenarch-math-geometry-2d", "=0.1.1"),
    "math-linear": ("moenarch-math-linear", "=0.1.1"),
    "media-core": ("moenarch-media-core", "=0.1.0"),
    "model-runtime": ("moenarch-model-runtime", "=0.1.1"),
    "numbers-core": ("moenarch-numbers-core", "=0.1.1"),
    "runtime-core": ("moenarch-runtime-core", "=0.2.1"),
    "runtime-onnx": ("moenarch-runtime-onnx", "=0.1.1"),
    "scenedetect-core": ("scenedetect-core", "=0.1.0"),
    "tensor-data": ("moenarch-tensor-data", "=0.1.1"),
    "text-core": ("moenarch-text-core", "=0.1.1"),
}
FORBIDDEN_DEPENDENCIES = {
    "three-d-processing-core",
    "video-analysis-posture",
    "video-analysis-test-support",
}
FORBIDDEN_PATH_PARTS = {
    "comfyui",
    "three-d",
    "video-analysis-gaussian-splatting",
    "video-analysis-mvs",
    "video-analysis-posture",
    "video-analysis-posture-io",
    "video-analysis-radiance-fields",
    "video-analysis-radiance-io",
    "video-analysis-radiance-pipeline",
    "video-analysis-reconstruction",
    "video-analysis-sfm",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def dependency_tables(document: dict):
    for section in ("dependencies", "dev-dependencies", "build-dependencies"):
        yield document.get(section, {})
    for target in document.get("target", {}).values():
        if isinstance(target, dict):
            for section in ("dependencies", "dev-dependencies", "build-dependencies"):
                yield target.get(section, {})


def validate() -> list[str]:
    errors: list[str] = []
    cargo_manifests = sorted((ROOT / "crates").rglob("Cargo.toml"))
    source_bun_manifests = sorted(
        path
        for path in (ROOT / "packages").glob("*/package.json")
        if path.parent.name != "visual-app-ui"
    )
    if len(cargo_manifests) != 108:
        errors.append(f"expected 108 Cargo manifests, found {len(cargo_manifests)}")
    if len(source_bun_manifests) != 54:
        errors.append(f"expected 54 source Bun manifests, found {len(source_bun_manifests)}")

    for path in [*cargo_manifests, *source_bun_manifests]:
        relative = path.relative_to(ROOT)
        if any(part in FORBIDDEN_PATH_PARTS for part in relative.parts):
            errors.append(f"excluded ownership path present: {relative}")

    root_manifest = tomllib.loads((ROOT / "Cargo.toml").read_text())
    workspace_dependencies = root_manifest["workspace"]["dependencies"]
    for key, (package, version) in EXPECTED_EXTERNAL.items():
        value = workspace_dependencies.get(key)
        if not isinstance(value, dict):
            errors.append(f"missing exact external dependency {key}")
            continue
        if value.get("package", key) != package or value.get("version") != version:
            errors.append(f"{key} must resolve as {package} {version}")
        if "path" in value or "git" in value:
            errors.append(f"{key} must be registry-only")

    for manifest in cargo_manifests:
        document = tomllib.loads(manifest.read_text())
        for dependencies in dependency_tables(document):
            for key, value in dependencies.items():
                if key in FORBIDDEN_DEPENDENCIES:
                    errors.append(
                        f"forbidden spatial/test-support edge in {manifest.relative_to(ROOT)}: {key}"
                    )
                if isinstance(value, dict):
                    if "git" in value:
                        errors.append(f"Git dependency in {manifest.relative_to(ROOT)}: {key}")
                    dependency_path = value.get("path")
                    if dependency_path:
                        resolved = (manifest.parent / dependency_path).resolve()
                        try:
                            resolved.relative_to(ROOT.resolve())
                        except ValueError:
                            errors.append(
                                f"cross-repository path dependency in {manifest.relative_to(ROOT)}: {key}"
                            )

    app_manifests = sorted((ROOT / "packages").glob("*-app/package.json"))
    if len(app_manifests) != 27:
        errors.append(f"expected 27 visual apps, found {len(app_manifests)}")
    for manifest in app_manifests:
        document = load_json(manifest)
        dependencies = document.get("dependencies", {})
        if dependencies.get("@moritzbrantner/visual-app-ui") != "workspace:*":
            errors.append(f"{manifest.relative_to(ROOT)} does not use visual-app-ui")
    stale_ui = []
    for path in (ROOT / "packages").rglob("*"):
        if path.is_file() and path.suffix in {".json", ".ts", ".tsx"}:
            if "@moritzbrantner/video-analysis-ui" in path.read_text(errors="ignore"):
                stale_ui.append(str(path.relative_to(ROOT)))
    if stale_ui:
        errors.append("rust-packages UI dependency remains: " + ", ".join(stale_ui))

    scene_text = "\n".join(
        path.read_text()
        for path in (
            ROOT / "crates/video/video-analysis-core/src/lib.rs",
            ROOT / "crates/video/video-analysis-detectors/src/lib.rs",
        )
    )
    for declaration in (
        r"struct\s+ContentScorer",
        r"struct\s+AdaptiveDetector\s*\{",
        r"struct\s+ThresholdDetector\s*\{",
        r"struct\s+HistogramDetector\s*\{",
        r"struct\s+HashDetector\s*\{",
    ):
        if re.search(declaration, scene_text):
            errors.append(f"canonical scene implementation copied: {declaration}")
    if "scenedetect_core::" not in scene_text:
        errors.append("canonical scenedetect-core adapter seam is absent")

    ownership = load_json(ROOT / "docs/repository-split/package-ownership.json")
    inventory = ownership.get("inventory", {})
    expected_inventory = {
        "families": 27,
        "cargo_source_packages": 108,
        "bun_source_packages": 54,
        "destination_authored_packages": 1,
    }
    if inventory != expected_inventory:
        errors.append(f"ownership inventory mismatch: {inventory!r}")
    if ownership.get("extraction_sha") != SOURCE_SHA:
        errors.append("ownership inventory does not freeze the actual extraction SHA")

    release = load_json(ROOT / "docs/repository-split/release-plan.json")
    if release.get("publication_authorized") is not False:
        errors.append("bootstrap release inventory must not authorize publication")
    if release.get("package_count") != 108:
        errors.append("release inventory must enumerate 108 Cargo packages")

    identity = load_json(ROOT / "docs/repository-split/source-byte-identity.json")
    if identity.get("source_commit") != SOURCE_SHA:
        errors.append("byte-identity inventory uses the wrong source SHA")
    identical = 0
    adapted = 0
    for record in identity.get("files", []):
        path = ROOT / record["path"]
        if not path.is_file():
            errors.append(f"inventoried source file is absent: {record['path']}")
            continue
        if record["status"] == "byte-identical":
            identical += 1
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != record["source_sha256"]:
                errors.append(f"byte identity changed: {record['path']}")
        elif record["status"] == "adapted":
            adapted += 1
        else:
            errors.append(f"unknown identity status for {record['path']}")
    if identical != identity.get("byte_identical_count"):
        errors.append("byte-identical count does not match file records")
    if adapted != identity.get("adapted_count"):
        errors.append("adapted count does not match file records")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.parse_args()
    errors = validate()
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("visual extraction structural checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
