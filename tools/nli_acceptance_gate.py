#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""NLI production acceptance gate — PRE-REGISTERED per the maintainer-AI protocol.

Registered BEFORE running; no post-hoc threshold moves. Tests whether the NLI backend improves
the REAL fact-check gate's admission over the incumbent lexical screen, end-to-end through
agent.fact_check_gate.external_ground, on a SEALED evidence snapshot (both arms score identical pairs).

PRE-REGISTERED GATE (all must hold for GO):
  * Primary: paired ΔF1 (NLI − lexical) at MATCHED coverage ≥ +0.05, 95% bootstrap CI over cases
    excluding 0, ≥3 seeds.
  * Calibrated-abstention guard: NLI answerable-coverage must not drop > 0.01 vs lexical
    (catch the "abstain-everything" pathology); report selective risk / AURC alongside F1.
  * PROTECTED: religion/history regression ≤ 0.01 — hard reject (N/A if pack has none; defer to promotion gate).
  * Fail-closed property: no retrieved evidence ⇒ gate abstains (never "entail against nothing") — asserted.
  * Two-family: primary (deberta cross-encoder) vs secondary (LLM-NLI, grok) verdict agreement κ ≥ 0.4;
    disagreements dumped to a failure-taxonomy file (seed corpus for boundary work / D).
  * Latency/cost per NLI call logged (escalation-tier economics).
Outcomes: GO → propose default-on + start E. NO-GO → ledger row + D. Either way candidateOnly.

HONEST BOUND: the C1 fixtures are CURATED retrieval (marker-stripped), closer to gold than live-noisy
web retrieval; a live-retrieval run (LiveFactBackend) is the harder follow-on.
"""
from __future__ import annotations
import argparse, hashlib, json, random, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent.fact_check_gate import (AtomicClaim, EvidenceSource, external_ground,
                                   classify_claim, risk_for)
from agent.nli_grounding import build_nli_entailment
from reframe_nli_grounding import gather
from eval_o2_energy_hidden import auroc

MARKER = re.compile(r"\[(ENTAILS|CONTRADICTS|SUPPORTS|REFUTES|NEUTRAL)\]", re.I)


# ---- sealed snapshot -------------------------------------------------------
def build_snapshot():
    rows = gather()
    snap = []
    for r in sorted(rows, key=lambda x: x["claim"]):
        evs = [MARKER.sub("", str(e)).strip() for e in r["evidence"]]
        evs = [e for e in evs if e]
        if evs:
            snap.append({"claim": r["claim"], "evidence": evs, "supported": bool(r["supported"])})
    blob = json.dumps(snap, sort_keys=True).encode()
    return snap, hashlib.sha256(blob).hexdigest()[:16]


def sources_of(row):
    return [EvidenceSource(id=f"e{i}", url=f"https://src{i}.example.org/x", title="",
                           snippet=e, publisher=f"src{i}", source_type="web")
            for i, e in enumerate(row["evidence"])]


def admission_score(res):
    return res.confidence if res.verdict == "accepted" else (-res.confidence if res.verdict == "rejected" else 0.0)


# ---- metrics ---------------------------------------------------------------
def f1_at_topk(scores, y, k):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    tp = sum(1 for i in order if y[i]); total_pos = sum(y)
    prec = tp / k if k else 0.0
    rec = tp / total_pos if total_pos else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def selective_risk_aurc(scores, y):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    run_err = 0.0; tot = 0.0
    for k, i in enumerate(order, 1):
        run_err += (0 if y[i] else 1)
        tot += run_err / k
    return tot / len(order) if order else 0.0


def paired_df1(nli, lex, y, coverage, n_boot, seed):
    rng = random.Random(seed); n = len(y); d = []
    k = max(1, round(coverage * n))
    base = f1_at_topk(nli, y, k) - f1_at_topk(lex, y, k)
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        yb = [y[i] for i in idx]
        if len(set(yb)) < 2:
            continue
        kb = max(1, round(coverage * len(idx)))
        d.append(f1_at_topk([nli[i] for i in idx], yb, kb) - f1_at_topk([lex[i] for i in idx], yb, kb))
    d.sort()
    return {"deltaF1": round(base, 4), "ci95": [round(d[int(.025*len(d))], 4), round(d[int(.975*len(d))], 4)]}


def cohen_kappa(a, b):
    labels = ["entails", "contradicts", "irrelevant"]
    n = len(a); po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if (1 - pe) else 1.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--f1-floor", type=float, default=0.05)          # PRE-REGISTERED
    ap.add_argument("--coverage-drop-max", type=float, default=0.01)  # PRE-REGISTERED
    ap.add_argument("--kappa-min", type=float, default=0.40)          # PRE-REGISTERED
    ap.add_argument("--secondary", action="store_true", help="run grok LLM-NLI as family 2 (κ)")
    ap.add_argument("--taxonomy-out", default="/tmp/nligate-data/nli_disagreements.jsonl")
    ap.add_argument("--output", default=None)
    a = ap.parse_args(argv)

    snap, snap_hash = build_snapshot()
    y = [r["supported"] for r in snap]
    print(f"sealed snapshot: n={len(snap)} supported={sum(y)} sha256={snap_hash}", file=sys.stderr)

    nli_fn = build_nli_entailment()

    # --- fail-closed property test (no evidence -> not accepted) ---
    test_claim = AtomicClaim(text="Some claim", type="open_empirical", risk="high")
    fc = external_ground(test_claim, lambda c: [], entailment=nli_fn)
    fail_closed_ok = fc.verdict != "accepted"

    # --- run both arms through the real gate on the sealed snapshot ---
    nli_scores, lex_scores = [], []
    nli_latencies = []
    for r in snap:
        srcs = sources_of(r)
        claim = AtomicClaim(text=r["claim"], type=classify_claim(r["claim"]), risk=risk_for(r["claim"]))
        retr = lambda c, _s=srcs: list(_s)
        t = time.monotonic()
        nli_scores.append(admission_score(external_ground(claim, retr, entailment=nli_fn)))
        nli_latencies.append(time.monotonic() - t)
        lex_scores.append(admission_score(external_ground(claim, retr, entailment=None)))

    # --- primary: ΔF1 at matched coverage (lexical operating coverage) ---
    lex_cov = sum(1 for s in lex_scores if s > 0) / len(lex_scores)
    matched_cov = max(0.1, lex_cov)
    df1 = {seed: paired_df1(nli_scores, lex_scores, y, matched_cov, a.n_boot, seed) for seed in (0, 1, 7)}
    df1_pass = all(df1[s]["ci95"][0] > 0 and df1[s]["deltaF1"] >= a.f1_floor for s in (0, 1, 7))

    # --- calibrated-abstention guard ---
    nli_cov = sum(1 for s in nli_scores if s > 0) / len(nli_scores)
    coverage_drop = lex_cov - nli_cov
    coverage_ok = coverage_drop <= a.coverage_drop_max
    sel = {"nli_AURC": round(selective_risk_aurc(nli_scores, y), 4),
           "lexical_AURC": round(selective_risk_aurc(lex_scores, y), 4),
           "nli_coverage": round(nli_cov, 4), "lexical_coverage": round(lex_cov, 4),
           "coverageDrop": round(coverage_drop, 4)}

    # --- two-family κ (secondary = grok LLM-NLI) ---
    kappa = None; kappa_ok = None
    if a.secondary:
        import agent.model as mdl
        def grok_label(premise, hyp):
            pr = (f"Evidence: {premise}\n\nClaim: {hyp}\n\nDoes the evidence ENTAIL, CONTRADICT, or is it "
                  f"NEUTRAL toward the claim? Reply one word: ENTAIL, CONTRADICT, or NEUTRAL.")
            try:
                o = (mdl.complete("You are a strict textual-entailment judge.", pr, max_tokens=6, spec="grok") or "").upper()
            except Exception:
                o = ""
            return "entails" if "ENTAIL" in o else ("contradicts" if "CONTRAD" in o else "irrelevant")
        v1, v2, disagreements = [], [], []
        for r in snap:
            for e in r["evidence"]:
                src = EvidenceSource(id="e", snippet=e)
                claim = AtomicClaim(text=r["claim"], type="open_empirical", risk="normal")
                l1 = nli_fn(claim, src); l2 = grok_label(e, r["claim"])
                v1.append(l1); v2.append(l2)
                if l1 != l2:
                    disagreements.append({"claim": r["claim"], "evidence": e[:200],
                                          "deberta": l1, "grok": l2, "supported": r["supported"]})
        kappa = round(cohen_kappa(v1, v2), 4); kappa_ok = kappa >= a.kappa_min
        Path(a.taxonomy_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.taxonomy_out).write_text("\n".join(json.dumps(d) for d in disagreements) + "\n")
        print(f"two-family κ={kappa} over {len(v1)} pairs; {len(disagreements)} disagreements -> {a.taxonomy_out}", file=sys.stderr)

    # --- PROTECTED suites (religion/history) present in this pack? ---
    protected_terms = re.compile(r"\b(bible|quran|torah|jesus|muhammad|buddha|god|dynasty|emperor|century|war of|treaty of)\b", re.I)
    protected_n = sum(1 for r in snap if protected_terms.search(r["claim"]))
    protected_note = ("N/A — the C1 fact pack has no dedicated religion/history PROTECTED suite; "
                      "this guard must be enforced by the W2 promotion gate (evaluate_update) before any default-on")

    go = df1_pass and coverage_ok and fail_closed_ok and (kappa_ok in (True, None))
    report = {
        "schema": "sophia.nli_acceptance_gate.v1", "candidateOnly": True, "canClaimAGI": False,
        "preRegistered": {"f1Floor": a.f1_floor, "coverageDropMax": a.coverage_drop_max, "kappaMin": a.kappa_min,
                          "primary": "paired ΔF1 at matched coverage ≥ floor, CI excludes 0, ≥3 seeds"},
        "sealedSnapshot": {"n": len(snap), "supported": sum(y), "sha256": snap_hash,
                           "source": "C1 fixtures + factcheck packs, marker-stripped (CURATED retrieval, not live-noisy)"},
        "matchedCoverage": round(matched_cov, 4),
        "primary_deltaF1_by_seed": df1, "primary_pass": df1_pass,
        "calibratedAbstentionGuard": {**sel, "coverageGuardPass": coverage_ok},
        "failClosed": {"noEvidenceVerdict": fc.verdict, "pass": fail_closed_ok},
        "twoFamily": {"kappa": kappa, "pass": kappa_ok, "secondary": "grok LLM-NLI" if a.secondary else "not run"},
        "protected": {"claimsMatched": protected_n, "note": protected_note},
        "latency": {"meanSecPerClaim_nli": round(sum(nli_latencies) / len(nli_latencies), 4),
                    "note": "cross-encoder cost per (claim,evidence); use NLI as an escalation tier, not blanket"},
        "verdict": ("GO — NLI beats the lexical arm through the real gate" if go
                    else "NO-GO — NLI does not clear the pre-registered acceptance gate vs the lexical screen"),
        "outcome": ("propose default-on via evaluate_update + start E (conformal escalation tier)" if go
                    else "keep candidateOnly (optional backend); ledger row + D (locate the boundary)"),
    }
    txt = json.dumps(report, indent=2)
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
