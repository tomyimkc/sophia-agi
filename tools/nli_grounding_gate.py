#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Self-contained, pre-registerable acceptance gate for grounding EntailmentFn backends.

Reproducible on main (no feat-only deps): compares admission arms through the REAL
agent.fact_check_gate.external_ground on a sha256-SEALED evidence snapshot from the C1 fact pack
+ factcheck packs. Arms: lexical incumbent (entailment=None) vs NLI (build_nli_entailment) vs
contradiction-hybrid (build_hybrid_entailment). Pre-registered metric: paired ΔF1 (arm − lexical)
at matched coverage ≥ floor, 95% bootstrap CI excluding 0, ≥3 seeds; calibrated-abstention guard
(coverage drop ≤ 0.01); fail-closed asserted.

RECORDED RESULT (2026-07-02, curated fixtures retrieval): NLI −0.098 (over-abstains), hybrid TIE
(ΔF1 0.0) — grounding here is retrieval-bound, not mechanism-bound (see coherence-reframes reports).
This harness is the base for the retrieval experiment (A): swap `gather` for a sentence-level
retriever and re-run with the manipulation check (≥50% fact-bearing evidence units).
"""
from __future__ import annotations
import argparse, hashlib, json, random, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.fact_check_gate import (AtomicClaim, EvidenceSource, external_ground,
                                   classify_claim, risk_for, lexical_entailment)
from agent.nli_grounding import build_nli_entailment, build_hybrid_entailment

MARKER = re.compile(r"\[(ENTAILS|CONTRADICTS|SUPPORTS|REFUTES|NEUTRAL)\]", re.I)


def gather():
    """C1 fixtures + factcheck packs -> {claim, evidence:[str], supported}; marker-stripped, deduped."""
    rows, seen = [], set()

    def add(claim, evs, supported):
        evs = [MARKER.sub("", str(e)).strip() for e in evs]
        evs = [e for e in evs if e]
        if claim in seen or not evs:
            return
        seen.add(claim); rows.append({"claim": claim, "evidence": evs, "supported": bool(supported)})

    fx = json.load(open(ROOT / "eval/fact_check/fixtures_v1.json"))["claims"]
    for r in (json.loads(l) for l in open(ROOT / "eval/fact_check/heldout_v1.jsonl")):
        if r["claim"] in fx:
            add(r["claim"], [f"{e.get('title','')} {e.get('snippet','')}" for e in fx[r["claim"]]], r["label"] == "true")
    for p in ["agi-proof/fact-check-live/fact-check-live-eval.LIVE-2026-06-24.json",
              "agi-proof/external-eval/factcheck-full-r1.json", "agi-proof/external-eval/factcheck-full-r2.json"]:
        try:
            for cs in json.load(open(ROOT / p))["cases"]:
                if cs.get("label") not in ("true", "false"):
                    continue
                evs = [" ".join(str(e.get(k, "")) for k in ("title", "publisher") if e.get(k))
                       for cl in (cs.get("claims") or []) for l in (cl.get("layers") or []) for e in (l.get("evidence") or [])]
                add(cs["claim"], evs, cs["label"] == "true")
        except Exception:
            pass
    return sorted(rows, key=lambda x: x["claim"])


def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]; neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def sources_of(row):
    return [EvidenceSource(id=f"e{i}", url=f"https://src{i}.example.org/x", title="", snippet=e,
                           publisher=f"src{i}", source_type="web") for i, e in enumerate(row["evidence"])]


def admission_score(res):
    return res.confidence if res.verdict == "accepted" else (-res.confidence if res.verdict == "rejected" else 0.0)


def f1_at_topk(scores, y, k):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    tp = sum(1 for i in order if y[i]); pos = sum(y)
    p = tp / k if k else 0.0; r = tp / pos if pos else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def selective_aurc(scores, y):
    order = sorted(range(len(scores)), key=lambda i: -scores[i]); run = 0.0; tot = 0.0
    for k, i in enumerate(order, 1):
        run += (0 if y[i] else 1); tot += run / k
    return tot / len(order) if order else 0.0


def paired_df1(arm, lex, y, cov, n_boot, seed):
    rng = random.Random(seed); n = len(y); k = max(1, round(cov * n)); d = []
    base = f1_at_topk(arm, y, k) - f1_at_topk(lex, y, k)
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        yb = [y[i] for i in idx]
        if len(set(yb)) < 2:
            continue
        kb = max(1, round(cov * len(idx)))
        d.append(f1_at_topk([arm[i] for i in idx], yb, kb) - f1_at_topk([lex[i] for i in idx], yb, kb))
    d.sort()
    return {"deltaF1": round(base, 4), "ci95": [round(d[int(.025*len(d))], 4), round(d[int(.975*len(d))], 4)]}


def arm_scores(snap, entailment):
    out = []
    for r in snap:
        srcs = sources_of(r)
        claim = AtomicClaim(text=r["claim"], type=classify_claim(r["claim"]), risk=risk_for(r["claim"]))
        out.append(admission_score(external_ground(claim, (lambda c, _s=srcs: list(_s)), entailment=entailment)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--f1-floor", type=float, default=0.05)
    ap.add_argument("--coverage-drop-max", type=float, default=0.01)
    ap.add_argument("--output", default=None)
    a = ap.parse_args(argv)

    snap = gather(); y = [r["supported"] for r in snap]
    snap_hash = hashlib.sha256(json.dumps(snap, sort_keys=True).encode()).hexdigest()[:16]
    print(f"sealed snapshot n={len(snap)} supported={sum(y)} sha256={snap_hash}", file=sys.stderr)

    nli_fn = build_nli_entailment()
    hybrid_fn = build_hybrid_entailment(lexical_entailment)
    fc = external_ground(AtomicClaim(text="x", type="open_empirical", risk="high"), lambda c: [], entailment=nli_fn)

    lex = arm_scores(snap, None); nli = arm_scores(snap, nli_fn); hyb = arm_scores(snap, hybrid_fn)
    cov = lambda s: sum(1 for x in s if x > 0) / len(s)
    lex_cov = cov(lex); matched = max(0.1, lex_cov)

    report = {"schema": "sophia.nli_grounding_gate.v1", "candidateOnly": True, "canClaimAGI": False,
              "sealedSnapshot": {"n": len(snap), "supported": sum(y), "sha256": snap_hash,
                                 "retrieval": "curated fixtures/metadata (swap gather() for sentence-level retrieval = experiment A)"},
              "matchedCoverage": round(matched, 4),
              "failClosed": {"noEvidenceVerdict": fc.verdict, "pass": fc.verdict != "accepted"},
              "arms": {}}
    for name, sc in [("NLI", nli), ("hybrid", hyb)]:
        df1 = {s: paired_df1(sc, lex, y, matched, a.n_boot, s) for s in (0, 1, 7)}
        report["arms"][name] = {
            "deltaF1_by_seed": {s: df1[s]["deltaF1"] for s in df1},
            "ci95_seed0": df1[0]["ci95"],
            "coverageDrop": round(lex_cov - cov(sc), 4),
            "selectiveAURC": round(selective_aurc(sc, y), 4),
            "pass": all(df1[s]["ci95"][0] > 0 and df1[s]["deltaF1"] >= a.f1_floor for s in (0, 1, 7))
                    and (lex_cov - cov(sc)) <= a.coverage_drop_max,
        }
    report["lexicalSelectiveAURC"] = round(selective_aurc(lex, y), 4)
    txt = json.dumps(report, indent=2)
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
