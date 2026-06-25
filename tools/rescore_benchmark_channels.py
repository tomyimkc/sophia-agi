#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Re-score committed benchmark response JSONs with FORMAT/CONTENT/COMBINED channels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.benchmark_checks import score_domain_channels  # noqa: E402
from tools.eval_ladder import _summarize_reports, _write_json  # noqa: E402
from tools.score_benchmark import load_responses  # noqa: E402

OUT_DIR = ROOT / "benchmark" / "model_runs"


def rescore_response_file(path: Path, *, traditions: dict) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    domain = payload.get("domain") or path.stem.split("-")[-1]
    responses = load_responses(payload)
    report = score_domain_channels(domain, responses, traditions)
    report["model"] = payload.get("model", path.stem)
    if path.stem.endswith(".report"):
        report_path = path
    else:
        report_path = path.with_suffix(".report.json")
        if report_path.name.endswith(".json.report.json"):
            report_path = path.parent / f"{path.stem}.report.json"
    existing = {}
    if report_path.exists():
        try:
            existing = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    for key in ("backend", "gateFailures"):
        if key in existing:
            report[key] = existing[key]
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def rebuild_ladder(ladder_path: Path) -> None:
    ladder = json.loads(ladder_path.read_text(encoding="utf-8"))
    label = None
    for rung in ladder.get("rungs", []):
        name = rung.get("rung", "")
        cmd = rung.get("command", [])
        if name in ("base", "adapter") and cmd:
            for i, part in enumerate(cmd):
                if part == "--adapter" and i + 1 < len(cmd):
                    label = Path(cmd[i + 1]).name.replace("/", "-").replace(" ", "-").lower()
                    break
                if part == "--model" and i + 1 < len(cmd) and name == "base":
                    label = cmd[i + 1].replace("/", "-").replace(" ", "-").lower()
        if not label:
            continue
        reports = []
        for domain in ("philosophy", "psychology", "history", "religion"):
            rp = OUT_DIR / f"local-{label}-{domain}.report.json"
            if rp.exists():
                reports.append(json.loads(rp.read_text(encoding="utf-8")))
        summary = _summarize_reports(reports)
        if summary:
            rung["summary"] = summary
    ladder["schema"] = "sophia.eval_ladder.v2"
    ladder["headline"] = "FORMAT / CONTENT / COMBINED per suite; protected gate uses CONTENT channel."
    _write_json(ladder_path, ladder)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="local-*-*.json", help="response JSON glob under benchmark/model_runs")
    ap.add_argument("--rebuild-ladder", action="append", default=[], help="ladder JSON paths to refresh")
    args = ap.parse_args(argv)

    traditions = json.loads((ROOT / "data" / "traditions.json").read_text(encoding="utf-8"))
    paths = sorted(p for p in OUT_DIR.glob(args.glob) if not p.name.endswith(".report.json"))
    for path in paths:
        rescore_response_file(path, traditions=traditions)
        print(f"rescored {path.name}")

    for ladder in args.rebuild_ladder:
        rebuild_ladder(ROOT / ladder)
        print(f"rebuilt ladder {ladder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
