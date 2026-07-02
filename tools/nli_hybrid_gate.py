#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Pre-registered gate for the CONTRADICTION-ONLY HYBRID vs the incumbent lexical screen.

Motivated by the NLI-entailment NO-GO: NLI *entailment* over-abstains, but the lexical screen's
negation cues + NLI's strong *contradiction* detection are the load-bearing signal. The hybrid keeps
admission on the cheap lexical screen and lets NLI reject only detected contradictions.

Same sealed snapshot + same pre-registered protocol as nli_acceptance_gate.py (no post-hoc moves):
  primary paired ΔF1 (hybrid − lexical) at matched coverage ≥ +0.05, CI excluding 0, ≥3 seeds;
  calibrated-abstention guard (coverage drop ≤ 0.01); fail-closed asserted.
Also reports NLI-entailment as the reference arm (the one that failed) for contrast.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent.fact_check_gate import AtomicClaim, external_ground, classify_claim, risk_for, lexical_entailment
from agent.nli_grounding import build_nli_entailment, build_hybrid_entailment
from nli_acceptance_gate import (build_snapshot, sources_of, admission_score, f1_at_topk,
                                 selective_risk_aurc, paired_df1)


def arm_scores(snap, entailment):
    out = []
    for r in snap:
        srcs = sources_of(r)
        claim = AtomicClaim(text=r["claim"], type=classify_claim(r["claim"]), risk=risk_for(r["claim"]))
        retr = lambda c, _s=srcs: list(_s)
        out.append(admission_score(external_ground(claim, retr, entailment=entailment)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--f1-floor", type=float, default=0.05)
    ap.add_argument("--coverage-drop-max", type=float, default=0.01)
    ap.add_argument("--output", default=None)
    a = ap.parse_args(argv)

    snap, snap_hash = build_snapshot()
    y = [r["supported"] for r in snap]
    print(f"sealed snapshot n={len(snap)} supported={sum(y)} sha256={snap_hash}", file=sys.stderr)

    nli_fn = build_nli_entailment()                                  # reference (failed) arm
    hybrid_fn = build_hybrid_entailment(lexical_entailment)          # admission=lexical, reject=NLI-contradict

    lex = arm_scores(snap, None)
    hyb = arm_scores(snap, hybrid_fn)
    nli = arm_scores(snap, nli_fn)

    def cov(s):
        return sum(1 for x in s if x > 0) / len(s)
    lex_cov = cov(lex); matched = max(0.1, lex_cov)

    hyb_df1 = {s: paired_df1(hyb, lex, y, matched, a.n_boot, s) for s in (0, 1, 7)}
    nli_df1 = {s: paired_df1(nli, lex, y, matched, a.n_boot, s) for s in (0, 1, 7)}
    hyb_pass = all(hyb_df1[s]["ci95"][0] > 0 and hyb_df1[s]["deltaF1"] >= a.f1_floor for s in (0, 1, 7))
    cov_drop = lex_cov - cov(hyb)
    cov_ok = cov_drop <= a.coverage_drop_max

    report = {
        "schema": "sophia.nli_hybrid_gate.v1", "candidateOnly": True, "canClaimAGI": False,
        "arm": "contradiction-only hybrid (admission=lexical, reject=NLI-contradiction) vs lexical incumbent",
        "sealedSnapshot": {"n": len(snap), "supported": sum(y), "sha256": snap_hash},
        "matchedCoverage": round(matched, 4),
        "hybrid_vs_lexical_deltaF1": {s: hyb_df1[s]["deltaF1"] for s in hyb_df1},
        "hybrid_vs_lexical_ci95": {s: hyb_df1[s]["ci95"] for s in hyb_df1},
        "hybrid_primary_pass": hyb_pass,
        "reference_NLIentailment_vs_lexical_deltaF1": {s: nli_df1[s]["deltaF1"] for s in nli_df1},
        "coverage": {"lexical": round(lex_cov, 4), "hybrid": round(cov(hyb), 4),
                     "drop": round(cov_drop, 4), "guardPass": cov_ok},
        "selectiveAURC": {"lexical": round(selective_risk_aurc(lex, y), 4),
                          "hybrid": round(selective_risk_aurc(hyb, y), 4)},
        "verdict": ("GO — the contradiction-hybrid beats the lexical incumbent" if (hyb_pass and cov_ok)
                    else "NO-GO — the contradiction-hybrid does not clear the pre-registered gate vs lexical"),
    }
    txt = json.dumps(report, indent=2)
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
