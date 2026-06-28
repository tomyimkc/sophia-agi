# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""CLI: Sentinel supervision report over the agent's run logs.

Reads ``agent/memory/agent_runs/*.jsonl``, aggregates failure classes, verdict
distribution, ArkDistill savings, and (optionally) compares the recent verdict
distribution against a committed baseline to flag drift. Emits JSON to stdout and,
with ``--out``, writes a public-report file under ``agi-proof/sentinel/``.

Pure/offline/deterministic — same logs in ⇒ same report out.

    python -m tools.sentinel_report
    python -m tools.sentinel_report --baseline agi-proof/sentinel/baseline.json --out
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.config import ROOT
from agent.sentinel import RUNS_DIR, detect_drift, scan_runs

OUT_DIR = ROOT / "agi-proof" / "sentinel"


def build_report(runs_dir: Path, baseline_path: Path | None, threshold: float) -> dict:
    rep = scan_runs(runs_dir)
    out = rep.to_dict()
    if baseline_path and baseline_path.exists():
        try:
            base = json.loads(baseline_path.read_text(encoding="utf-8"))
            base_verdicts = base.get("verdicts", base)
            out["drift"] = detect_drift(rep.verdicts, base_verdicts, threshold=threshold)
        except Exception as exc:  # report-building must not crash on a bad baseline
            out["drift"] = {"error": repr(exc)}
    return out


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="Sentinel supervision report")
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--baseline", default=None, help="JSON file with a baseline verdict distribution")
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--out", action="store_true", help="also write agi-proof/sentinel/report.json")
    args = ap.parse_args(argv)

    report = build_report(
        Path(args.runs_dir),
        Path(args.baseline) if args.baseline else None,
        args.threshold,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
