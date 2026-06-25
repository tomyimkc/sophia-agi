#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Re-apply CONTENT pass-gate headline to committed eval ladder JSON artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LADDERS = [
    ROOT / "training" / "local_sophia_v2" / "eval_ladder_baseline.json",
    ROOT / "training" / "local_sophia_v2" / "eval_ladder_adapter.json",
]


def _rescore_summary(summary: dict) -> dict:
    domains = summary.get("domains") or {}
    if not domains:
        return summary
    fmt_passed = sum(d["format"]["passed"] for d in domains.values())
    content_passed = sum(d["content"]["passed"] for d in domains.values())
    combined_passed = sum(d["combined"]["passed"] for d in domains.values())
    total = sum(d["total"] for d in domains.values())
    pct = lambda n: round(100.0 * n / total, 1) if total else 0.0
    for dom in domains.values():
        dom["passGate"] = "content"
        dom["passed"] = dom["content"]["passed"]
        dom["score_pct"] = dom["content"]["score_pct"]
    summary["passGate"] = "content"
    summary["channels"] = {
        "format": {"passed": fmt_passed, "total": total, "score_pct": pct(fmt_passed)},
        "content": {"passed": content_passed, "total": total, "score_pct": pct(content_passed)},
        "combined": {"passed": combined_passed, "total": total, "score_pct": pct(combined_passed)},
    }
    summary["passed"] = content_passed
    summary["score_pct"] = pct(content_passed)
    return summary


def rescore_ladder(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["headline"] = "PASS gate = CONTENT channel per suite; FORMAT and COMBINED reported only."
    payload["passGate"] = "content"
    for rung in payload.get("rungs", []):
        summary = rung.get("summary")
        if summary:
            rung["summary"] = _rescore_summary(summary)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"rescored {path}")


def main() -> int:
    for ladder in LADDERS:
        if ladder.exists():
            rescore_ladder(ladder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
