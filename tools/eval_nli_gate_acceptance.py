#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""NLI production acceptance gate: does the NLI backend improve the REAL fact-check gate's
admission over the incumbent lexical screen — end-to-end through agent.fact_check_gate.external_ground?

For each claim+evidence, build the AtomicClaim (real classify/risk) + a retriever returning the
evidence as EvidenceSources on distinct domains, then run external_ground twice:
  lexical : entailment=None      (incumbent conservative lexical screen)
  nli     : entailment=build_nli_entailment()   (the validated primitive)
Evidence is marker-stripped ([ENTAILS]/[CONTRADICTS] are a fixtures oracle artifact; real retrieval
has none) so NLI cannot cheat. Admission score per arm = +conf if accepted, -conf if rejected, 0 if
held; gold = supported (label==true).

Gate: AUROC(NLI admission vs supported) beats AUROC(lexical), paired-bootstrap CI excluding 0,
>=2 seeds; report F1 at each arm's natural operating point too. Runs on py3.10 (fence-regex fix).
candidateOnly. No weights updated.
"""
from __future__ import annotations
import argparse, json, random, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent.fact_check_gate import AtomicClaim, EvidenceSource, external_ground, classify_claim, risk_for
from agent.nli_grounding import build_nli_entailment
from reframe_nli_grounding import gather  # {claim, evidence:[str], supported}
from eval_o2_energy_hidden import auroc

MARKER = re.compile(r"\[(ENTAILS|CONTRADICTS|SUPPORTS|REFUTES|NEUTRAL)\]", re.I)


def make_sources(evidence):
    out = []
    for i, e in enumerate(evidence):
        s = MARKER.sub("", str(e)).strip()
        if s:
            out.append(EvidenceSource(id=f"e{i}", url=f"https://src{i}.example.org/x",
                                      title="", snippet=s, publisher=f"src{i}", source_type="web"))
    return out


def admission_score(res) -> float:
    if res.verdict == "accepted":
        return res.confidence
    if res.verdict == "rejected":
        return -res.confidence
    return 0.0


def f1_at_operating_point(scores, y):
    # natural operating point: admit if score>0 (accepted)
    tp = sum(1 for s, l in zip(scores, y) if s > 0 and l)
    fp = sum(1 for s, l in zip(scores, y) if s > 0 and not l)
    fn = sum(1 for s, l in zip(scores, y) if s <= 0 and l)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    cov = sum(1 for s in scores if s > 0) / len(scores)
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "coverage": round(cov, 4)}


def paired(a, b, y, n_boot, seed):
    rng = random.Random(seed); n = len(y); d = []
    ba, bb = auroc(a, y), auroc(b, y)
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        la = [y[i] for i in idx]
        if len(set(la)) < 2:
            continue
        xa, xb = auroc([a[i] for i in idx], la), auroc([b[i] for i in idx], la)
        if xa is not None and xb is not None:
            d.append(xa - xb)
    d.sort()
    return {"aurocNLI": round(ba, 4), "aurocLexical": round(bb, 4), "delta": round(ba - bb, 4),
            "ci95": [round(d[int(.025*len(d))], 4), round(d[int(.975*len(d))], 4)]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=5000); ap.add_argument("--output", default=None)
    a = ap.parse_args(argv)

    rows = gather()
    print(f"claim+evidence rows={len(rows)} supported={sum(r['supported'] for r in rows)}", file=sys.stderr)
    print("loading NLI backend ...", file=sys.stderr)
    nli_fn = build_nli_entailment()

    nli_scores, lex_scores, y = [], [], []
    for r in rows:
        srcs = make_sources(r["evidence"])
        if not srcs:
            continue
        claim = AtomicClaim(text=r["claim"], type=classify_claim(r["claim"]), risk=risk_for(r["claim"]))
        retr = lambda c, _s=srcs: list(_s)
        nli_scores.append(admission_score(external_ground(claim, retr, entailment=nli_fn)))
        lex_scores.append(admission_score(external_ground(claim, retr, entailment=None)))
        y.append(bool(r["supported"]))
    print(f"scored {len(y)} claims through the real gate", file=sys.stderr)

    report = {"schema": "sophia.nli_gate_acceptance.v1", "candidateOnly": True, "canClaimAGI": False,
              "test": "NLI vs lexical admission through agent.fact_check_gate.external_ground (marker-stripped evidence)",
              "n": len(y), "nSupported": sum(y),
              "f1_naturalOperatingPoint": {"NLI": f1_at_operating_point(nli_scores, y),
                                           "lexical": f1_at_operating_point(lex_scores, y)},
              "seeds": {}}
    for seed in (0, 1, 7):
        report["seeds"][seed] = paired(nli_scores, lex_scores, y, a.n_boot, seed)
    nli_beats = all(report["seeds"][s]["ci95"][0] > 0 for s in (0, 1, 7))
    report["NLI_beatsLexical_allSeeds"] = nli_beats
    report["verdict"] = ("ACCEPTANCE_GATE_MET: NLI admission beats the lexical arm through the real gate"
                         if nli_beats else "gate_not_met: NLI does not significantly beat the lexical arm here")
    txt = json.dumps(report, indent=2)
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
