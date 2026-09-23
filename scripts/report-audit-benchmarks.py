#!/usr/bin/env python3
"""Collect Criterion confidence intervals as evidence, never as a speed gate."""
import json
import hashlib
import platform
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
records = []
for name in ("scene_ingest", "text_regressions"):
    for path in sorted((root / "target/criterion" / name).glob("*/*/new/estimates.json")):
        estimate = json.loads(path.read_text())["mean"]
        records.append({"benchmark": "/".join(path.relative_to(root / "target/criterion").parts[:-2]),
            "unit": "nanoseconds", "mean": estimate["point_estimate"], "confidenceInterval": estimate["confidence_interval"]})
if not records:
    raise SystemExit("No audit Criterion estimates found; run the benchmarks first")
output = root / ".artifacts/audit/benchmarks.json"
output.parent.mkdir(parents=True, exist_ok=True)
diff = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=root)
output.write_text(json.dumps({"schemaVersion": 1, "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "baseline": "281d4c553beed5c151e5a7e66a41309631ae6fff", "timingGate": False,
    "dirty": bool(diff), "workingDiffSha256": hashlib.sha256(diff).hexdigest() if diff else None,
    "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(), "platform": platform.platform(),
    "comparison": "scene_ingest: legacy growing-prefix adapter versus one-pass public ingest in one process; text_regressions: current semantic workload scaling, not old/new equivalence",
    "equivalence": "same cuts, source timestamps, frame count, and all prior content scores; one-pass additionally exposes component metrics",
    "records": records}, indent=2) + "\n")
print(output)
