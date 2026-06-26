#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Measure grounded search's calibrated abstention: does it answer strong sources and
downgrade (hedge/abstain) weak ones?

The grounded-search reflex (`agent.grounded_search`) should serve a well-sourced query as an
answer and downgrade a weakly-sourced one — the "fail-closed perception" property. This harness
turns that into a measured discrimination over the OKF wiki: it builds one probe per page,
labels it strong/weak by the page's ``authorConfidence`` tier, runs `grounded_search`, and
reports how often strong sources are kept vs weak sources downgraded.

Fully offline & deterministic (committed local embedder + OKF provenance graph; no model, no
LLM judge). Honest bound: probes are **self-authored** over the existing corpus and the label is
the page's own declared confidence tier — so this validates the reflex + the harness, not a
third-party number. Mirrors tools/eval_graded_confidence.py. Reproduce: `python tools/eval_grounded_search.py`.

  python tools/eval_grounded_search.py            # full run, writes a candidate report
  python tools/eval_grounded_search.py --json      # machine-readable summary
  python tools/eval_grounded_search.py --limit 12  # quick smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "agi-proof" / "benchmark-results" / "grounded-search.public-report.json"

STRONG_TIERS = {"consensus", "attributed", "compiled", "layered"}
WEAK_TIERS = {"disputed", "legendary", "anachronism_risk", "none_extant"}


def build_probes(limit: int | None = None) -> "list[dict]":
    """One probe per OKF page that declares a confidence tier, labeled strong/weak."""
    from okf import load_pages

    from agent.config import WIKI_DIR

    probes: list[dict] = []
    for page in load_pages(WIKI_DIR):
        tier = page.meta.get("authorConfidence")
        if tier not in STRONG_TIERS and tier not in WEAK_TIERS:
            continue
        title = page.meta.get("canonicalTitleEn") or page.id.replace("_", " ")
        # An attribution-framed query for pages that carry attribution discipline; else generic.
        if page.meta.get("attributedAuthor") or page.meta.get("doNotAttributeTo"):
            q = f"Who is the author of {title}?"
        else:
            q = f"Tell me about {title}."
        probes.append({"q": q, "id": page.id, "tier": tier,
                       "strong": tier in STRONG_TIERS})
    if limit:
        probes = probes[:limit]
    return probes


def run(*, limit: int | None = None) -> dict:
    from agent.grounded_search import grounded_search

    probes = build_probes(limit=limit)
    pages = _pages()
    rows = []
    for p in probes:
        r = grounded_search(p["q"], pages=pages)
        rows.append({**p, "action": r.action, "grounded": r.grounded,
                     "confidence": r.confidence, "target": r.target})

    strong = [r for r in rows if r["strong"]]
    weak = [r for r in rows if not r["strong"]]
    strong_answered = sum(1 for r in strong if r["action"] == "answer")
    weak_downgraded = sum(1 for r in weak if r["action"] in {"hedge", "abstain"})

    def _frac(num: int, den: int) -> float:
        return round(num / den, 4) if den else 0.0

    strong_keep = _frac(strong_answered, len(strong))
    weak_down = _frac(weak_downgraded, len(weak))
    return {
        "benchmark": "grounded-search calibrated abstention (self-authored OKF probes)",
        "candidateOnly": True,
        "validated": False,
        "level3Evidence": False,
        "probes": len(rows),
        "strong": {"n": len(strong), "answeredFraction": strong_keep},
        "weak": {"n": len(weak), "downgradedFraction": weak_down},
        # A single number: how cleanly the reflex separates strong-kept from weak-downgraded.
        "discrimination": round(strong_keep + weak_down - 1.0, 4),
        "actionByTier": _action_by_tier(rows),
        "honestBound": (
            "Self-authored probes over the live OKF wiki; the label is each page's own declared "
            "authorConfidence tier, not third-party truth. Measures whether the serve/abstain "
            "reflex tracks source quality (a calibrated prior), not whether any sentence is true."
        ),
    }


def _action_by_tier(rows) -> dict:
    out: dict[str, dict] = {}
    for r in rows:
        bucket = out.setdefault(r["tier"], {"answer": 0, "hedge": 0, "abstain": 0, "n": 0})
        bucket[r["action"]] += 1
        bucket["n"] += 1
    return out


_PAGES = None


def _pages():
    global _PAGES
    if _PAGES is None:
        from okf import load_pages

        from agent.config import WIKI_DIR

        _PAGES = list(load_pages(WIKI_DIR))
    return _PAGES


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Grounded-search calibrated-abstention benchmark")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args(argv)

    report = run(limit=args.limit)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        s, w = report["strong"], report["weak"]
        print(f"Probes: {report['probes']}")
        print(f"  strong sources answered : {s['answeredFraction']:.3f}  (n={s['n']})")
        print(f"  weak sources downgraded : {w['downgradedFraction']:.3f}  (n={w['n']})")
        print(f"  discrimination          : {report['discrimination']:+.3f}")
        print("  action by tier:")
        for tier, b in sorted(report["actionByTier"].items()):
            print(f"    {tier:18s} answer={b['answer']} hedge={b['hedge']} abstain={b['abstain']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.json:
        print(f"Wrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
