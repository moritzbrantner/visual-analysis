#!/usr/bin/env python3
"""Generate the frozen visual extraction inventories from one source commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "b8b29cf8db0b86ed1b133a18155adf24992f9483"
SOURCE_REPOSITORY = "moritzbrantner/rust-packages"
DESTINATION_REPOSITORY = "moritzbrantner/visual-analysis"
IMAGE_FAMILIES = (
    "captioning",
    "classification",
    "core",
    "detection",
    "embeddings",
    "io",
    "ocr",
    "processing",
    "segmentation",
    "synthesis",
)
VIDEO_FAMILIES = (
    "core",
    "data",
    "dataset",
    "detectors",
    "editing",
    "features",
    "ffmpeg",
    "ingest",
    "output",
    "recognition",
    "segmentation",
    "split",
    "storage",
    "synthesis",
    "tracking",
    "transform",
)


def family_names() -> tuple[str, ...]:
    return tuple(f"image-analysis-{name}" for name in IMAGE_FAMILIES) + tuple(
        f"video-analysis-{name}" for name in VIDEO_FAMILIES
    ) + ("vision-core",)


def package_roots() -> list[str]:
    roots: list[str] = []
    for base in family_names():
        domain = "image" if base.startswith("image-") else "video" if base.startswith("video-") else "vision"
        roots.extend(
            [
                f"crates/{domain}/{base}",
                f"crates/{domain}/{base}-cli",
                f"crates/{domain}/{base}-server",
                f"crates/bindings/{base}-wasm",
                f"packages/{base}-app",
                f"packages/{base}-wasm",
            ]
        )
    return roots


def git(source: Path, *args: str, text: bool = False) -> bytes | str:
    return subprocess.check_output(["git", "-C", str(source), *args], text=text)


def source_bytes(source: Path, path: str) -> bytes:
    value = git(source, "show", f"{SOURCE_SHA}:{path}")
    assert isinstance(value, bytes)
    return value


def canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def generate(source: Path) -> None:
    git(source, "cat-file", "-e", f"{SOURCE_SHA}^{{commit}}")
    ownership_source = json.loads(
        source_bytes(source, "docs/repository-split/package-ownership.json")
    )
    source_records = [
        record
        for record in ownership_source["packages"]
        if record.get("target_repository") == "visual-analysis"
    ]
    cargo_records = [record for record in source_records if record["ecosystem"] == "cargo"]
    bun_records = [record for record in source_records if record["ecosystem"] == "bun"]
    if (len(cargo_records), len(bun_records)) != (108, 54):
        raise SystemExit(
            f"source ownership must contain 108 Cargo and 54 Bun records, found {len(cargo_records)}/{len(bun_records)}"
        )

    adapter_record = {
        "automatic_publish_eligible": False,
        "contract_owner": "visual-app-ui",
        "current_domain": "frontend adapter",
        "current_package_name": "@moritzbrantner/visual-app-ui",
        "current_published_version": None,
        "current_release_workflow": "none",
        "ecosystem": "bun",
        "extraction_phase": "visual",
        "id": "bun:@moritzbrantner/visual-app-ui",
        "intended_next_release_owner": DESTINATION_REPOSITORY,
        "intended_package_name": "@moritzbrantner/visual-app-ui",
        "manifest_path": "packages/visual-app-ui/package.json",
        "migration_status": "destination-authored focused adapter; private and nonpublishing",
        "package_kind": "private adapter",
        "provenance": {
            "kind": "destination-authored",
            "source_commit": SOURCE_SHA,
            "source_paths": [
                "packages/video-analysis-ui/src/package-surface/**",
                "packages/video-analysis-ui/src/shared/primitives.tsx",
                "packages/video-analysis-ui/src/shared/utils.ts",
            ],
        },
        "publication_class": "private; not publishable",
        "source_version": "0.1.0",
        "target_repository": "visual-analysis",
        "temporary_boundary_violations": [],
        "wrapped_library": None,
    }
    ownership = {
        "schema_version": 1,
        "repository": DESTINATION_REPOSITORY,
        "source_repository": SOURCE_REPOSITORY,
        "extraction_sha": SOURCE_SHA,
        "source_ownership_document_sha": SOURCE_SHA,
        "source_ownership_records_sha256": canonical_digest(source_records),
        "inventory": {
            "families": len(family_names()),
            "cargo_source_packages": len(cargo_records),
            "bun_source_packages": len(bun_records),
            "destination_authored_packages": 1,
        },
        "packages": source_records + [adapter_record],
    }
    write_json(ROOT / "docs/repository-split/package-ownership.json", ownership)

    release_packages = []
    for record in cargo_records:
        manifest = ROOT / record["manifest_path"]
        package = tomllib.loads(manifest.read_text())["package"]
        raw_version = package.get("version")
        version = raw_version if isinstance(raw_version, str) else "0.1.0"
        release_packages.append(
            {
                "manifest_path": record["manifest_path"],
                "name": package["name"],
                "source_version": version,
            }
        )
    release_plan = {
        "schema_version": 1,
        "repository": DESTINATION_REPOSITORY,
        "source_commit": SOURCE_SHA,
        "status": "blocked-prerequisites",
        "publication_authorized": False,
        "package_count": len(release_packages),
        "blockers": [
            {
                "package": "moenarch-audio-contracts",
                "version": "0.1.0",
                "reason": "exact registry version is absent",
            },
            {
                "package": "scenedetect-core",
                "version": "0.1.0",
                "reason": "accepted canonical scene contract is not registry-visible",
            },
        ],
        "packages": sorted(release_packages, key=lambda item: item["name"]),
    }
    write_json(ROOT / "docs/repository-split/release-plan.json", release_plan)

    raw_paths = git(
        source,
        "ls-tree",
        "-r",
        "--name-only",
        SOURCE_SHA,
        "--",
        *package_roots(),
        text=True,
    )
    assert isinstance(raw_paths, str)
    records = []
    for path in raw_paths.splitlines():
        expected = source_bytes(source, path)
        destination = ROOT / path
        if not destination.is_file():
            raise SystemExit(f"missing extracted source file: {path}")
        actual = destination.read_bytes()
        records.append(
            {
                "path": path,
                "source_sha256": hashlib.sha256(expected).hexdigest(),
                "status": "byte-identical" if actual == expected else "adapted",
            }
        )
    identity = {
        "schema_version": 1,
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_SHA,
        "file_count": len(records),
        "byte_identical_count": sum(record["status"] == "byte-identical" for record in records),
        "adapted_count": sum(record["status"] == "adapted" for record in records),
        "files": records,
    }
    write_json(ROOT / "docs/repository-split/source-byte-identity.json", identity)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-repo",
        type=Path,
        default=Path("/home/moenarch/moritzbrantner/rust-packages"),
    )
    args = parser.parse_args()
    generate(args.source_repo)


if __name__ == "__main__":
    main()
