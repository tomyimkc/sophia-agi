#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run REM dream collective offline cycle (candidate infrastructure).

REM writes to agent_dreams ledger only; wake runs conscience + consolidation.

Scheduled runs (cron-friendly)::

    python3 tools/run_dream_collective.py --out agi-proof/dream-collective/dream-collective.public-report.json

Or from repo root with explicit python::

    cd /path/to/sophia-agi && python3 tools/run_dream_collective.py --cron
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.dream_collective import DreamCandidate, run_dream_cycle  # noqa: E402
from skills.symbiosis_network import Nutrient, broadcast_nutrients  # noqa: E402

DEFAULT_OUT = ROOT / "agi-proof" / "dream-collective" / "dream-collective.public-report.json"


def build_demo_report(*, ledger_path: Path | None = None) -> dict:
    ledger = ledger_path or (ROOT / "agent" / "memory" / "agent_dreams" / "dream_ledger_demo.jsonl")
    if ledger.exists():
        ledger.unlink()

    candidates = [
        DreamCandidate(
            dream_id="dream_dao_reading",
            goal="Cross-tradition reading hypothesis",
            text=(
                "Hypothesis (bulk only, candidateOnly): exploring how reception history might be read "
                "through philosophical vs religious lenses without merging lineages. "
                "Sophia is an AGI-candidate verifier-gated epistemic framework; this dream note is "
                "candidate infrastructure only. 中文摘要：試探性閱讀假設。"
            ),
        ),
        DreamCandidate(
            dream_id="dream_blocked_eval_leak",
            goal="Who wrote the Gospel of Matthew — and how should we answer that theologically vs historically?",
            text="This dream should be contamination-blocked because it echoes a benchmark question.",
        ),
    ]

    cycle = run_dream_cycle(candidates, ledger_path=ledger, tier="draft")

    nutrients = broadcast_nutrients([
        Nutrient(
            claim="Ancestor veneration is a ritual practice distinct from Confucian moral philosophy.",
            evidence="Within Confucian ritual religion, ancestor veneration is practiced; Confucian moral philosophy concerns ethics and cultivation.",
            donor_id="seat_confucian_ritual",
            tradition="confucian_ritual",
        ),
        Nutrient(
            claim="Confucius personally authored the Dao De Jing.",
            evidence="Traditional attribution says Confucius wrote the Dao De Jing.",
            donor_id="bad_donor",
            tradition="daoist",
        ),
    ])

    return {
        "schema": "sophia.dream_collective_report.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "claimBoundary": "REM dream + symbiosis pilot. Dreams are non-canonical until wake gate.",
        "dreamCycle": cycle,
        "symbiosis": nutrients,
        "invariants": {
            "rem_never_wiki_direct": True,
            "eval_leak_blocked": cycle["rem"]["contaminationBlocked"] >= 1,
            "symbiosis_holds_bad_nutrient": nutrients["heldCount"] >= 1,
            "at_least_one_consolidated": cycle["wake"]["consolidated"] >= 1,
        },
        "ok": (
            cycle["rem"]["contaminationBlocked"] >= 1
            and nutrients["heldCount"] >= 1
            and cycle["wake"]["consolidated"] >= 1
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="REM dream collective pilot")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--cron",
        action="store_true",
        help="cron-friendly alias: write default public report and exit 0/1 on invariants",
    )
    args = ap.parse_args()

    report = build_demo_report()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "out": str(out)}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
