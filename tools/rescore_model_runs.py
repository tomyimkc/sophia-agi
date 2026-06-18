#!/usr/bin/env python3
"""Re-score saved model_runs/*.json after benchmark heuristic updates."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "benchmark" / "model_runs"
DOMAINS = ("philosophy", "psychology", "history", "religion")
SCORE = ROOT / "tools" / "score_benchmark.py"


def main() -> int:
    updated = 0
    for path in sorted(RUNS.glob("local-*.json")):
        if path.name.endswith(".report.json"):
            continue
        domain = path.stem.split("-")[-1]
        if domain not in DOMAINS:
            continue
        report_path = path.with_suffix("").with_name(path.stem + ".report.json")
        proc = subprocess.run(
            [sys.executable, str(SCORE), str(path), "--domain", domain, "--out", str(report_path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode not in (0, 1):
            print(proc.stderr or proc.stdout)
            return proc.returncode
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(f"{path.name}: {report['passed']}/{report['total']} ({report['score_pct']}%)")
        updated += 1

    if updated:
        subprocess.run([sys.executable, str(ROOT / "tools" / "update_leaderboards.py")], cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(ROOT / "tools" / "build_web_data.py")], cwd=ROOT, check=True)
    print(f"Rescored {updated} run(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())