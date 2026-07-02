#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Experiment A (pre-registered by maintainer-AI): does SENTENCE-LEVEL fact-bearing retrieval let an
entailment backend beat the incumbent lexical screen through the real gate — the pre-registered test
of the 'grounding is retrieval-bound' diagnosis?

Frozen protocol: same C1 pack, same harness/seeds as the rung-5 acceptance gate; ONLY the retriever
changes — keyless Wikipedia passage retrieval (sentence-level) instead of the fixtures'
titles/metadata. New evidence snapshot sha256-sealed.

MANIPULATION CHECK (interpretability clause): ≥50% of retrieved units must be fact-bearing sentences,
vs ~0% in the metadata arm. If unmet, the verdict is "manipulation failed — build retrieval first,"
NOT "hypothesis falsified." Also excludes the DETERMINISTIC C1 claim types (math/doi/url/date) which
the gate verifies symbolically and never grounded in evidence.

Primary: paired ΔF1 (arm − lexical) @ matched coverage ≥ +0.05 vs lexical, 95% CI excluding 0, 3 seeds;
MDE reported honestly (no self-authored task extension). Guards: coverage drop ≤ 0.01, fail-closed,
latency logged. Pre-declared outcomes: GO → retrieval-bound confirmed → fund D. NO-GO + manip PASSED →
mechanism loses even on good evidence → bank per B. Manip FAILED → retrieval build-out is prerequisite.
"""
from __future__ import annotations
import argparse, hashlib, json, random, re, ssl, sys, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from agent.fact_check_gate import (AtomicClaim, EvidenceSource, external_ground,
                                   classify_claim, risk_for, lexical_entailment)
from agent.nli_grounding import build_nli_entailment, build_hybrid_entailment

DETERMINISTIC = ("math", "doi", "url", "date")   # verified symbolically, never evidence-grounded


def wiki_summary(title, timeout=8):
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title.replace(" ", "_"))
    req = urllib.request.Request(url, headers={"User-Agent": "sophia-bench/1.0"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=timeout, context=ssl.create_default_context())).get("extract", "")
    except Exception:
        return ""


def entity(claim):
    m = re.search(r"\b(?:wrote|authored|penned|composed)\s+(.+?)[.?]?$", claim, re.I)
    if m:
        return m.group(1).strip(" .?\"'")
    m = re.search(r"(.+?)\s+was\s+written\s+by", claim, re.I)
    if m:
        return m.group(1).strip(" .?\"'")
    m = re.search(r"^(.+?)\s+(?:is|was|are|were|increased|decreased|rose|fell)\b", claim)
    return (m.group(1).strip(" .?\"'") if m else claim.strip(" .?\"'"))[:90]


def factbearing(s):
    return len(s) >= 40 and bool(re.search(r"\b(is|was|are|were|wrote|written|by|has|had|born|located|published|increased|decreased)\b", s, re.I))


def retrieve(claim_text):
    ex = wiki_summary(entity(claim_text))
    if not ex:
        return []
    return [s.strip() for s in re.split(r"(?<=[.])\s+", ex) if s.strip()][:5]


def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]; neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    return sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg) / (len(pos) * len(neg))


def admission_score(res):
    return res.confidence if res.verdict == "accepted" else (-res.confidence if res.verdict == "rejected" else 0.0)


def f1_at_topk(scores, y, k):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    tp = sum(1 for i in order if y[i]); pos = sum(y)
    p = tp / k if k else 0.0; r = tp / pos if pos else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=4000); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--output", default=None)
    a = ap.parse_args(argv)

    ho = [json.loads(l) for l in open(ROOT / "eval/fact_check/heldout_v1.jsonl")]
    groundable = [r for r in ho if r["label"] in ("true", "false")
                  and not any(d in (r.get("type") or "") for d in DETERMINISTIC)]
    print(f"C1 total={len(ho)} | evidence-groundable (non-deterministic, true/false)={len(groundable)}", file=sys.stderr)

    # sentence-level Wikipedia retrieval (the only variable changed vs rung 5)
    lat = []
    def build(r):
        t = time.monotonic(); ev = retrieve(r["claim"]); lat.append(time.monotonic() - t)
        return {"claim": r["claim"], "evidence": ev, "supported": r["label"] == "true"}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(build, groundable))
    got = [r for r in rows if r["evidence"]]
    all_units = [s for r in got for s in r["evidence"]]
    fb = sum(1 for s in all_units for _ in [0] if factbearing(s))
    manip_frac = fb / max(len(all_units), 1)
    manip_pass = manip_frac >= 0.50 and len(got) >= 10
    print(f"retrieved evidence for {len(got)}/{len(groundable)} claims | fact-bearing units {fb}/{len(all_units)} = {manip_frac:.2f}", file=sys.stderr)

    report = {"schema": "sophia.retrieval_experiment_A.v1", "candidateOnly": True, "canClaimAGI": False,
              "retriever": "keyless Wikipedia REST passage retrieval (sentence-level)",
              "c1_claim_mix": {"total": len(ho), "deterministic_excluded": len(ho) - len(groundable),
                               "evidence_groundable": len(groundable), "got_wiki_evidence": len(got)},
              "manipulationCheck": {"factBearingFraction": round(manip_frac, 3), "floor": 0.50,
                                    "nClaimsWithEvidence": len(got), "pass": manip_pass}}

    if not manip_pass:
        report["verdict"] = ("MANIPULATION FAILED / claim-mix inadequate — C1 is dominated by deterministic "
                             "claims + econ-empirical claims poorly served by single-sentence retrieval; only a "
                             "few authorship claims ground cleanly. A proper A-test needs an evidence-groundable "
                             "pack (authorship/factual) with a real sentence retriever. RETRIEVAL BUILD-OUT is the "
                             "prerequisite work item — the retrieval-bound diagnosis is NOT yet gated (nor falsified).")
        report["latency"] = {"meanSecPerClaim": round(sum(lat)/len(lat), 3)}
        txt = json.dumps(report, indent=2)
        if a.output:
            Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(txt + "\n")
        print(txt); return 0

    # manipulation passed -> seal snapshot + run the gate
    snap = sorted(got, key=lambda x: x["claim"])
    snap_hash = hashlib.sha256(json.dumps(snap, sort_keys=True).encode()).hexdigest()[:16]
    y = [r["supported"] for r in snap]

    def arm_scores(entailment):
        out = []
        for r in snap:
            srcs = [EvidenceSource(id=f"e{i}", url=f"https://src{i}.example.org/x", snippet=s, publisher=f"s{i}")
                    for i, s in enumerate(r["evidence"])]
            claim = AtomicClaim(text=r["claim"], type=classify_claim(r["claim"]), risk=risk_for(r["claim"]))
            out.append(admission_score(external_ground(claim, (lambda c, _s=srcs: list(_s)), entailment=entailment)))
        return out

    nli_fn = build_nli_entailment(); hyb_fn = build_hybrid_entailment(lexical_entailment)
    lex = arm_scores(None); nli = arm_scores(nli_fn); hyb = arm_scores(hyb_fn)
    cov = lambda s: sum(1 for x in s if x > 0) / len(s)
    matched = max(0.1, cov(lex))
    report["sealedSnapshot"] = {"n": len(snap), "supported": sum(y), "sha256": snap_hash}
    report["matchedCoverage"] = round(matched, 4)
    report["arms"] = {}
    for name, sc in [("NLI", nli), ("hybrid", hyb)]:
        df1 = {s: paired_df1(sc, lex, y, matched, a.n_boot, s) for s in (0, 1, 7)}
        report["arms"][name] = {"deltaF1": {s: df1[s]["deltaF1"] for s in df1}, "ci95_seed0": df1[0]["ci95"],
                                "coverageDrop": round(cov(lex) - cov(sc), 4),
                                "pass": all(df1[s]["ci95"][0] > 0 and df1[s]["deltaF1"] >= 0.05 for s in (0, 1, 7))}
    report["latency"] = {"meanSecPerClaim": round(sum(lat)/len(lat), 3)}
    report["MDE_note"] = f"n={len(snap)} with {sum(y)} supported — report CI width as power; small n likely underpowered for a +0.05 floor (no self-authored extension)"
    go = any(v["pass"] for v in report["arms"].values())
    report["verdict"] = ("GO — good evidence lets the mechanism beat the incumbent -> retrieval-bound CONFIRMED -> fund D"
                         if go else "NO-GO (manip passed) — mechanism does not beat the incumbent even on sentence-level evidence -> bank per B")
    txt = json.dumps(report, indent=2)
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(txt + "\n")
    print(txt); return 0


if __name__ == "__main__":
    raise SystemExit(main())
