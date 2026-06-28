#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Measure the LIVE graded-confidence signal: does it downgrade weakly-sourced answers?

S2 closed L1's gap — the graded router's confidence is now derived live from the routed
page's provenance neighborhood (`agent.grounded_confidence`) instead of being caller-
supplied. This harness measures whether that signal does the right thing across the real
OKF wiki corpus: for each page it pools a provenance confidence over the page + its k-hop
neighborhood, runs the graded router, and buckets the resulting action (answer / hedge /
abstain) by the page's ``authorConfidence`` tier.

The discrimination claim: **strong** sources (consensus / attributed / compiled) keep their
answer; **weak** sources (disputed / legendary / anachronism_risk / none_extant) are
downgraded to hedge or abstain. Reported as keep-rate (strong) and downgrade-rate (weak),
with the per-tier means. Deterministic, offline, no model call, no LLM judge.

Honest bound: this scores *how well-sourced the routed page is* and that the router reacts
monotonically — NOT whether a generated sentence is true. Candidate, not a headline.
Reproduce: ``python tools/eval_graded_confidence.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "agi-proof" / "benchmark-results" / "graded-confidence.public-report.json"

_STRONG = ("consensus", "attributed", "compiled", "layered")
_WEAK = ("disputed", "legendary", "anachronism_risk", "none_extant")


def run(*, hops: int = 1, thresholds: dict | None = None) -> dict:
    from agent.config import WIKI_DIR
    from agent.graded_decision import decide
    from agent.grounded_confidence import grounded_source_confidence
    from okf.page import load_pages

    pages = load_pages(WIKI_DIR)
    per_tier: dict[str, list] = {}
    for p in pages:
        conf = grounded_source_confidence(p.id, pages, hops=hops)
        if conf is None:
            continue
        action = decide(gate_passed=True, confidence=conf, thresholds=thresholds)["action"]
        tier = p.meta.get("authorConfidence") or "unspecified"
        per_tier.setdefault(tier, []).append((conf, action))

    tiers = {}
    for tier, rows in per_tier.items():
        actions = Counter(a for _, a in rows)
        tiers[tier] = {
            "n": len(rows),
            "meanConfidence": round(sum(c for c, _ in rows) / len(rows), 4),
            "actions": dict(actions),
        }

    def _bucket(names):
        rows = [r for t in names for r in per_tier.get(t, [])]
        n = len(rows)
        downgraded = sum(1 for _, a in rows if a in ("hedge", "abstain"))
        kept = n - downgraded
        return n, kept, downgraded

    s_n, s_keep, s_down = _bucket(_STRONG)
    w_n, w_keep, w_down = _bucket(_WEAK)
    return {
        "benchmark": "graded-confidence discrimination (live provenance signal over OKF wiki)",
        "candidateOnly": True,
        "validated": False,
        "level3Evidence": False,
        "hops": hops,
        "thresholds": thresholds or {"hi": 0.7, "lo": 0.4},
        "perTier": tiers,
        "strongSources": {"tiers": list(_STRONG), "n": s_n,
                          "keepRate": round(s_keep / s_n, 4) if s_n else None},
        "weakSources": {"tiers": list(_WEAK), "n": w_n,
                        "downgradeRate": round(w_down / w_n, 4) if w_n else None},
        "honestBound": ("Scores how well-sourced the routed page is and that the router "
                        "reacts monotonically to it — NOT whether a generated sentence is "
                        "true. Self-authored over the OKF wiki; deterministic; candidate."),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Graded-confidence discrimination over OKF wiki")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args(argv)

    report = run(hops=args.hops)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"OKF wiki — graded-confidence discrimination (hops={report['hops']})")
        for tier in (*_STRONG, *_WEAK, "unspecified"):
            t = report["perTier"].get(tier)
            if t:
                print(f"  {tier:16s} n={t['n']:2d} meanConf={t['meanConfidence']:.3f} "
                      f"actions={t['actions']}")
        s, w = report["strongSources"], report["weakSources"]
        print(f"  STRONG keep-rate     {s['keepRate']}  (n={s['n']})")
        print(f"  WEAK   downgrade-rate {w['downgradeRate']}  (n={w['n']})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.json:
        print(f"Wrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
