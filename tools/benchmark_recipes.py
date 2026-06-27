#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Benchmark the candidate gemma-3 adapter recipes under the measurement contract
(agi-proof/measurement-thesis.md) and rank them — "which recipe is best".

For each recipe it reads whatever artifacts exist (primary eval, 3-family judge, N=70 retention)
and scores three axes with their PROPER resolution:
  * PRIMARY source-discipline (N=354x3 — well powered): count + magnitude of CI-clean improving
    pre-registered metrics, minus protected-suite regressions. THIS is the discriminating axis.
  * JUDGE (semantic): majority adapter-vs-base win-rate across 3 independent families.
  * RETENTION (N=70 — COARSE, MDE ~0.24): a GO/NO-GO guardrail only; the contract forbids ranking
    on it because the probe cannot resolve a 5pt effect (eval_stats.mde_at_n).
Ranking = guardrails first (retention GO if measured, judge winrate>0.5, no protected regression),
then by primary strength. Honest by construction: it will not call a retention difference "best"
when the instrument can't see it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.eval_stats import mde_at_n  # noqa: E402

WM = ROOT / "agi-proof" / "benchmark-results" / "wisdom-market"
PRIMARY = ["qualification_rate_on_contested", "tradition_merge_rate",
           "false_attribution_rate", "moral_route_accuracy", "citation_fidelity", "provenance_accuracy"]
PROTECTED = ["protected_history_regression", "protected_religion_regression"]

# name -> file prefix on the branch
RECIPES = {
    "M3-SFT (rank16, baseline)": "M3-pilot",
    "M4-ORPO (from base)": "M4-orpo",
    "M4-ORPO-on-SFT (stack)": "M4-orpo-sft",
    "M3-stable (rank8+KL+replay)": "M3-stable",
}


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _ci_clean(ci) -> bool:
    return isinstance(ci, list) and None not in ci and (ci[0] > 0 or ci[1] < 0)


def score_recipe(prefix: str) -> dict:
    ev = _load(WM / f"{prefix}-eval.json")
    jg = _load(WM / f"{prefix}-judge.json")
    rt = _load(WM / f"{prefix}-retention.json") or _load(WM / f"{prefix}-retention-eval.json")
    out = {"prefix": prefix, "hasEval": ev is not None, "hasJudge": jg is not None, "hasRetention": rt is not None}
    if ev:
        deltas = ev.get("adapterPromptVsBasePrompt", {})
        wins, mag, names = 0, 0.0, []
        for m in PRIMARY:
            d = deltas.get(m) or {}
            if d.get("improves") and _ci_clean(d.get("ci")):
                wins += 1; mag += abs(d.get("delta") or 0.0); names.append(m)
        # protected regression = a protected metric WORSENS with CI clear of 0
        prot = []
        for m in PROTECTED:
            d = deltas.get(m) or {}
            if (d.get("delta") or 0) > 0 and _ci_clean(d.get("ci")):
                prot.append(m)
        out.update(primaryWins=wins, primaryMag=round(mag, 4), primaryMetrics=names,
                   protectedRegressions=prot, nCases=ev.get("nCases"), runs=ev.get("runs"))
    if jg:
        wr = [v.get("adapter_winrate", 0) for v in (jg.get("perJudge") or {}).values()]
        out["judgeMeanWinrate"] = round(sum(wr) / len(wr), 4) if wr else None
        out["judgeMajority"] = (jg.get("majorityVote") or {}).get("adapter_winrate")
    if rt:
        out.update(retentionDelta=rt.get("delta"), retentionCI=rt.get("delta_ci95"),
                   retains=rt.get("retains"), retentionN=rt.get("nTasks"))
    return out


def rank(scores: "list[dict]") -> "list[dict]":
    def key(s):
        # guardrails (higher = better): judge corroborated, retention GO (or unmeasured=neutral),
        # no protected regression; then primary strength.
        judge_ok = 1 if (s.get("judgeMeanWinrate") or 0) > 0.5 else 0
        ret_ok = 0 if s.get("retains") is False else 1            # measured-and-fails -> demote
        prot_ok = 0 if s.get("protectedRegressions") else 1
        return (judge_ok, ret_ok, prot_ok, s.get("primaryWins", 0), s.get("primaryMag", 0))
    return sorted(scores, key=key, reverse=True)


# The simplest recipe — must always be in the comparison as the baseline (principle #9: a "new
# recipe wins" claim is meaningless without the simple baseline in the same table).
SIMPLE_BASELINE = "M3-SFT (rank16, baseline)"


def superiority_receipt(ranked: "list[dict]") -> dict:
    """PILLAR 8/9 receipt for a 'recipe X is best' claim. A ranking is only trustworthy if the
    decision axis can RESOLVE the gap between the top two recipes (else 'best' is noise) AND the
    simplest recipe was included as a baseline. Emits GO/NO-GO."""
    checks = []
    has_baseline = any(s["name"] == SIMPLE_BASELINE for s in ranked)
    checks.append({"check": "simple-baseline-included", "ok": has_baseline,
                   "detail": f"'{SIMPLE_BASELINE}' in comparison = {has_baseline}"})
    # gap between #1 and #2 on the powered primary magnitude vs the primary MDE.
    powered_gap = True
    detail = "only one recipe measured — no ranking claim"
    if len(ranked) >= 2:
        n = (ranked[0].get("nCases") or 354) * (ranked[0].get("runs") or 3)
        mde = round(mde_at_n(max(1, n)), 3)
        gap = abs((ranked[0].get("primaryMag") or 0) - (ranked[1].get("primaryMag") or 0))
        powered_gap = gap >= mde
        detail = (f"#1 {ranked[0]['name']} vs #2 {ranked[1]['name']}: primaryMag gap={round(gap,3)} "
                  f"vs MDE@{n}={mde} -> {'resolvable' if powered_gap else 'NOT resolvable (gap is noise)'}")
    checks.append({"check": "ranking-axis-powered", "ok": powered_gap, "detail": detail})
    ok = all(c["ok"] for c in checks)
    return {"claim": "recipe ranking ('which recipe is best')", "verdict": "GO" if ok else "NO-GO",
            "ok": ok, "checks": checks, "best": (ranked[0]["name"] if ranked else None),
            "boundary": "ranking valid ONLY on the powered primary; retention is a guardrail, not a score"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=WM / "recipe-benchmark.json")
    ap.add_argument("--emit-receipt", action="store_true",
                    help="also write recipe-benchmark.gate.json (GO/NO-GO for a superiority claim)")
    args = ap.parse_args()

    scores = [dict(name=n, **score_recipe(p)) for n, p in RECIPES.items()]
    measured = [s for s in scores if s.get("hasEval")]
    ranked = rank(measured)
    report = {
        "benchmark": "gemma-3 adapter recipes under the measurement contract",
        "discriminatingAxis": "PRIMARY source-discipline (N=354x3, well powered)",
        "powerNote": {
            "primary_mde_at_354": round(mde_at_n(354), 3),
            "retention_mde_at_70": round(mde_at_n(70), 3),
            "caveat": ("Retention (N=70) cannot resolve a 5pt effect (MDE ~0.24) -> used ONLY as a "
                       "GO/NO-GO guardrail, never to rank. Primary (N=354x3) is the decision axis."),
        },
        "ranked": ranked,
        "best": (ranked[0]["name"] if ranked else None),
        "boundary": "Single-seed per recipe except M3-SFT (3-seed); candidate_only; canClaimAGI:false.",
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.emit_receipt:
        rcpt = superiority_receipt(ranked)
        (WM / "recipe-benchmark.gate.json").write_text(
            json.dumps(rcpt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"superiority receipt: {rcpt['verdict']} (best={rcpt['best']}) -> recipe-benchmark.gate.json")
        for c in rcpt["checks"]:
            print(f"  {'✓' if c['ok'] else '✗'} {c['check']}: {c['detail']}")
    # human summary
    print(f"\n=== RECIPE BENCHMARK (best = {report['best']}) ===")
    print(f"power: primary MDE@354={report['powerNote']['primary_mde_at_354']} | "
          f"retention MDE@70={report['powerNote']['retention_mde_at_70']} (coarse guardrail)")
    for i, s in enumerate(ranked, 1):
        print(f"{i}. {s['name']:30s} primaryWins={s.get('primaryWins')} mag={s.get('primaryMag')} "
              f"judgeWR={s.get('judgeMeanWinrate')} retainsΔ={s.get('retentionDelta')}({s.get('retains')}) "
              f"protReg={s.get('protectedRegressions')}")
    print(f"wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
