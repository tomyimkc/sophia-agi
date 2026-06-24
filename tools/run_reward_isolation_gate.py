#!/usr/bin/env python3
"""Run the SSIL G2 reward-isolation tripwire and emit a candidate artifact.

The gate checks a *monitored access trace* for a proposed self-modification: it
must not write the reward/verification surface (tampering) nor read the blind
surface — reward scalar and hidden-eval packs (wireheading / leakage). Unmonitored
deltas quarantine; clean monitored deltas promote. Output is candidate
infrastructure only: ``candidateOnly: true``, ``level3Evidence: false``.

See docs/11-Platform/Safe-Self-Improvement-Loop.md (gate G2).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.reward_isolation import demo_reward_isolation_report  # noqa: E402

DEFAULT_OUT = ROOT / "agi-proof" / "self-gate" / "reward-isolation-gate.public-report.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="SSIL G2 reward-isolation tripwire")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output report path")
    ap.add_argument("--print", action="store_true", help="print the report to stdout")
    args = ap.parse_args()

    report = demo_reward_isolation_report()
    ok = all(report["invariants"].values())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.print:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"reward-isolation gate invariants: {'PASS' if ok else 'FAIL'} -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
