#!/usr/bin/env python3
"""Validate visual-analysis's capability ownership boundary.

The machine-readable contract lives in docs/ownership/visual-analysis-boundary.json.

- No package depends on canonical scene-detection crates other than through the
  declared scenedetect-core adapter seam, and only the declared adapter crates
  depend on scenedetect-core.
- No package depends on downstream corpus or product repositories.
- The scene seam and its owner cannot be dropped from the contract.

Copied scene algorithms are rejected by scripts/check_visual_extraction.py.
"""

from __future__ import annotations

import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_RELATIVE = Path("docs/ownership/visual-analysis-boundary.json")
REPOSITORY = "moritzbrantner/visual-analysis"
SCENE_SEAM_PACKAGE = "scenedetect-core"
# Pinned here so that editing the contract this check protects cannot widen the seam.
PINNED_ADAPTER_CRATES = {"moenarch-video-analysis-core", "moenarch-video-analysis-detectors"}
REQUIRED_EXCLUDED = {
    "canonical-scene-boundary-algorithms",
    "media-corpus-persistence-and-product-workflows",
}
DEPENDENCY_SECTIONS = ("dependencies", "dev-dependencies", "build-dependencies")
BUN_SECTIONS = ("dependencies", "devDependencies", "optionalDependencies", "peerDependencies")


@dataclass(frozen=True)
class Dependency:
    crate: str
    package: str


def load_contract(root: Path) -> dict[str, Any]:
    return json.loads((root / CONTRACT_RELATIVE).read_text())


def contract_errors(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if contract.get("schemaVersion") != 1:
        errors.append("boundary contract must use schemaVersion 1")
    if contract.get("repository") != REPOSITORY:
        errors.append(f"boundary contract must name {REPOSITORY}")
    if contract.get("layer") != "capability":
        errors.append("visual-analysis must stay a capability-layer repository")
    excluded = {authority.get("authority") for authority in contract.get("excludedAuthorities", [])}
    missing = sorted(REQUIRED_EXCLUDED - excluded)
    if missing:
        errors.append(f"boundary contract must exclude: {missing}")
    owned = set(contract.get("ownedCapabilities", []))
    if owned & excluded:
        errors.append(f"capabilities both owned and excluded: {sorted(owned & excluded)}")
    seam = contract.get("sceneSeam", {})
    if seam.get("package") != SCENE_SEAM_PACKAGE or seam.get("ownerRepository") != "moritzbrantner/scenedetect-rs":
        errors.append("the scene seam must be scenedetect-core, owned by scenedetect-rs")
    if set(seam.get("adapterCrates", [])) != PINNED_ADAPTER_CRATES:
        errors.append(
            "scene seam adapter crates must be exactly moenarch-video-analysis-core and moenarch-video-analysis-detectors; "
            "widening the seam needs an ADR and a change to this check"
        )
    return errors


def forbidden_rules(contract: dict[str, Any]) -> tuple[set[str], tuple[str, ...]]:
    names: set[str] = set()
    prefixes: list[str] = []
    for authority in contract.get("excludedAuthorities", []):
        names.update(authority.get("forbiddenPackages", []))
        prefixes.extend(authority.get("forbiddenPackagePrefixes", []))
    return names, tuple(prefixes)


def package_manifests(root: Path, workspace: dict[str, Any]) -> list[Path]:
    """The root package, every listed member, and every in-tree package reached
    through a path dependency (workspace `exclude` is deliberately ignored)."""
    resolved_root = root.resolve()
    settings = workspace.get("workspace", {})
    workspace_dependencies = settings.get("dependencies", {})

    def admissible(directory: Path) -> bool:
        directory = directory.resolve()
        inside = directory == resolved_root or resolved_root in directory.parents
        return inside and (directory / "Cargo.toml").is_file()

    pending: list[Path] = []
    if "package" in workspace:
        pending.append(root)
    for pattern in settings.get("members", []):
        pending.extend(member for member in root.glob(pattern) if (member / "Cargo.toml").is_file())
    pending.extend(
        root / spec["path"]
        for spec in workspace_dependencies.values()
        if isinstance(spec, dict) and "path" in spec and admissible(root / spec["path"])
    )
    manifests: set[Path] = set()
    while pending:
        directory = pending.pop().resolve()
        manifest = directory / "Cargo.toml"
        if manifest in manifests:
            continue
        manifests.add(manifest)
        for key, spec in dependency_specs(tomllib.loads(manifest.read_text())):
            if not isinstance(spec, dict):
                continue
            if spec.get("workspace"):
                base = workspace_dependencies.get(key)
                if not (isinstance(base, dict) and "path" in base):
                    continue
                candidate = root / base["path"]
            elif "path" in spec:
                candidate = directory / spec["path"]
            else:
                continue
            if admissible(candidate):
                pending.append(candidate)
    return sorted(manifests)


def dependency_specs(manifest: dict[str, Any]):
    tables = [manifest.get(section, {}) for section in DEPENDENCY_SECTIONS]
    for target in manifest.get("target", {}).values():
        tables.extend(target.get(section, {}) for section in DEPENDENCY_SECTIONS)
    for table in tables:
        yield from table.items()


def crate_dependencies(root: Path) -> list[Dependency]:
    workspace = tomllib.loads((root / "Cargo.toml").read_text())
    workspace_dependencies = workspace.get("workspace", {}).get("dependencies", {})
    dependencies: list[Dependency] = []
    for manifest_path in package_manifests(root, workspace):
        manifest = tomllib.loads(manifest_path.read_text())
        crate = manifest.get("package", {}).get("name", str(manifest_path.parent))
        for key, spec in dependency_specs(manifest):
            if isinstance(spec, dict) and spec.get("workspace"):
                base = workspace_dependencies.get(key, {})
                spec = {**base, **spec} if isinstance(base, dict) else {"version": base}
            package = spec.get("package", key) if isinstance(spec, dict) else key
            dependencies.append(Dependency(crate, package))
    for key, spec in workspace_dependencies.items():
        package = spec.get("package", key) if isinstance(spec, dict) else key
        dependencies.append(Dependency("[workspace]", package))
    return dependencies


def dependency_errors(contract: dict[str, Any], root: Path) -> list[str]:
    errors: list[str] = []
    names, prefixes = forbidden_rules(contract)
    for dependency in crate_dependencies(root):
        if dependency.package == SCENE_SEAM_PACKAGE:
            if dependency.crate not in PINNED_ADAPTER_CRATES | {"[workspace]"}:
                errors.append(
                    f"{dependency.crate} depends on {SCENE_SEAM_PACKAGE} outside the declared scene adapter seam"
                )
            continue
        if dependency.package in names or dependency.package.startswith(prefixes):
            errors.append(
                f"{dependency.crate} depends on {dependency.package}, which belongs to an excluded authority"
            )
    return errors


def bun_errors(contract: dict[str, Any], root: Path) -> list[str]:
    errors: list[str] = []
    names, prefixes = forbidden_rules(contract)
    manifests = [root / "package.json", *sorted((root / "packages").glob("*/package.json"))]
    for manifest in manifests:
        if not manifest.is_file():
            continue
        document = json.loads(manifest.read_text())
        for section in BUN_SECTIONS:
            for name in document.get(section, {}):
                bare = name.rsplit("/", 1)[-1]
                if any(
                    candidate in names or candidate.startswith(prefixes) for candidate in (name, bare)
                ):
                    errors.append(
                        f"{manifest.relative_to(root)} {section} depends on excluded package {name}"
                    )
    return errors


def validate(root: Path = ROOT) -> list[str]:
    contract = load_contract(root)
    return contract_errors(contract) + dependency_errors(contract, root) + bun_errors(contract, root)


def main() -> int:
    errors = validate()
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if errors:
        return 1
    print("visual capability boundary checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
