#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Refresh domain leaderboards from scored report files."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark" / "results"
REF = ROOT / "benchmark" / "reference"
RUNS = ROOT / "benchmark" / "model_runs"
DOMAINS = ("philosophy", "psychology", "history", "religion", "personality")


def load_report(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect_entries(domain: str) -> list[dict]:
    entries = []
    sources = [
        (REF / f"responses-{domain}.json", RESULTS / f"reference-{domain}.report.json", "sophia-teacher-reference"),
    ]
    for resp_path, report_path, model in sources:
        if not report_path.exists() and resp_path.exists():
            import subprocess
            import sys

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "score_benchmark.py"),
                    str(resp_path),
                    "--domain",
                    domain,
                    "--out",
                    str(report_path),
                ],
                check=False,
            )
        report = load_report(report_path)
        if report:
            entries.append({
                "model": model,
                "score_pct": report["score_pct"],
                "passed": report["passed"],
                "total": report["total"],
            })

    for report_path in sorted(RUNS.glob(f"*-{domain}.report.json")):
        report = load_report(report_path)
        if not report:
            continue
        model = report.get("model", report_path.stem)
        passed = report.get("passed", 0)
        total = report.get("total", 0)
        score_pct = report.get("score_pct")
        if score_pct is None and total:
            score_pct = round(100.0 * passed / total, 1)
        entries.append({
            "model": model,
            "score_pct": score_pct or 0.0,
            "passed": passed,
            "total": total,
        })

    return entries


def main() -> int:
    for domain in DOMAINS:
        bench = json.loads((ROOT / "tests" / f"benchmark-{domain}.json").read_text(encoding="utf-8"))
        payload = {
            "domain": domain,
            "benchmark": f"sophia-{domain}-v1",
            "updated": "2026-06-18",
            "cases": len(bench.get("cases", [])),
            "entries": collect_entries(domain),
        }
        out = RESULTS / f"leaderboard-{domain}.json"
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Updated {out} ({len(payload['entries'])} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())