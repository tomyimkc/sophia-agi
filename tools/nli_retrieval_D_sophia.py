#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Experiment D — sophia-domain retrieval test (pre-registered; verifier FROZEN).

PRIMARY finding (pre-run, per protocol): the non-forge sophia-domain evidence-groundable pack caps at
n=106 (30 supported attributions + 76 doNotAttributeTo negatives); MDE ~0.21 >> the +0.05 floor, so it
CANNOT power the pre-registered gate. Reported BEFORE running. The prerequisite work item is building a
larger curated evidence-groundable attribution pack (data curation) — the binding constraint in sophia's
domain is labeled evidence-groundable DATA, not the retriever or the verifier.

This harness still runs the frozen NLI-backed gate on the real n=106 pack to check the two clauses that
do NOT require power: (1) incumbent-health (lexical coverage ≥ 0.10 — the guard against FEVER-style
baseline collapse) and (2) the manipulation check (≥50% fact-bearing retrieved units). Evidence is
retrieved from sophia's OWN wiki corpus prose (the actual deployment grounding source), sentence-split.
ΔF1 is reported with its honestly-wide CI and the underpowered caveat; NO GO/NO-GO is declared.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, random, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from agent.fact_check_gate import (AtomicClaim, EvidenceSource, external_ground,
                                   classify_claim, risk_for, lexical_entailment)
from agent.nli_grounding import build_nli_entailment


def _dna(v):
    x = v.get("doNotAttributeTo", [])
    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except Exception:
            x = []
    return x if isinstance(x, list) else []


def _cap(s):
    return " ".join(w.capitalize() for w in str(s).replace("_", " ").split())


def build_pack():
    attr = json.load(open(ROOT / "data/attributions.json"))
    rows = []
    for rid, v in attr.items():
        if not isinstance(v, dict):
            continue
        title = v.get("canonicalTitleEn"); author = v.get("attributedAuthor")
        if not title or not author:
            continue
        rows.append({"id": rid, "title": title, "claim": f"{_cap(author)} wrote {title}", "supported": True})
        for f in _dna(v):
            rows.append({"id": rid, "title": title, "claim": f"{_cap(f)} wrote {title}", "supported": False})
    return rows


def wiki_prose(rid):
    """Sentence-level evidence from sophia's own wiki page prose (the deployment grounding source)."""
    for sub in ("text", "figure", "concept", "event"):
        p = ROOT / "wiki" / sub / f"{rid}.md"
        if p.exists():
            body = p.read_text()
            body = re.sub(r"^---.*?---", "", body, flags=re.S)          # strip frontmatter
            body = re.sub(r"^#.*$", "", body, flags=re.M)                 # strip headings
            sents = [s.strip() for s in re.split(r"(?<=[.。])\s+", body) if len(s.strip()) >= 30]
            return sents[:6]
    return []


def factbearing(s):
    return len(s) >= 30 and bool(re.search(r"\b(is|was|are|were|wrote|written|by|attributed|compiled|authored|tradition)\b", s, re.I))


def sources_of(sents):
    return [EvidenceSource(id=f"e{i}", url=f"https://wiki.local/{i}", snippet=s, publisher=f"wiki{i}", source_type="wiki")
            for i, s in enumerate(sents)]


def admission_score(res):
    return res.confidence if res.verdict == "accepted" else (-res.confidence if res.verdict == "rejected" else 0.0)


def f1_topk(scores, y, k):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    tp = sum(1 for i in order if y[i]); pos = sum(y)
    p = tp / k if k else 0.0; r = tp / pos if pos else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def paired_df1(arm, lex, y, cov, n_boot, seed):
    rng = random.Random(seed); n = len(y); k = max(1, round(cov * n)); d = []
    base = f1_topk(arm, y, k) - f1_topk(lex, y, k)
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        yb = [y[i] for i in idx]
        if len(set(yb)) < 2:
            continue
        kb = max(1, round(cov * len(idx)))
        d.append(f1_topk([arm[i] for i in idx], yb, kb) - f1_topk([lex[i] for i in idx], yb, kb))
    d.sort()
    return {"deltaF1": round(base, 4), "ci95": [round(d[int(.025*len(d))], 4), round(d[int(.975*len(d))], 4)]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=4000); ap.add_argument("--output", default=None)
    a = ap.parse_args(argv)

    pack = build_pack()
    # retrieve sophia-wiki prose evidence per work
    ev_by_id = {}
    for rid in {r["id"] for r in pack}:
        ev_by_id[rid] = wiki_prose(rid)
    rows = [r for r in pack if ev_by_id.get(r["id"])]
    for r in rows:
        r["evidence"] = ev_by_id[r["id"]]
    y = [r["supported"] for r in rows]
    snap_hash = hashlib.sha256(json.dumps([{k: r[k] for k in ("claim", "supported", "evidence")} for r in rows], sort_keys=True).encode()).hexdigest()[:16]

    all_units = [s for r in rows for s in r["evidence"]]
    fb = sum(1 for s in all_units if factbearing(s))
    manip = fb / max(len(all_units), 1)

    nli_fn = build_nli_entailment()
    def arm(entail):
        out = []
        for r in rows:
            claim = AtomicClaim(text=r["claim"], type=classify_claim(r["claim"]), risk=risk_for(r["claim"]))
            srcs = sources_of(r["evidence"])
            out.append(admission_score(external_ground(claim, (lambda c, _s=srcs: list(_s)), entailment=entail)))
        return out
    lex = arm(None); nli = arm(nli_fn)
    cov = lambda s: sum(1 for x in s if x > 0) / len(s)
    lex_cov = cov(lex); matched = max(0.1, lex_cov)
    df1 = {s: paired_df1(nli, lex, y, matched, a.n_boot, s) for s in (0, 1, 7)}

    report = {
        "schema": "sophia.retrieval_experiment_D.v1", "candidateOnly": True, "canClaimAGI": False,
        "domain": "sophia provenance/attribution (philosophy/religion/history)",
        "retriever": "sophia's own wiki-corpus prose, sentence-split (offline; the deployment grounding source)",
        "pack": {"n": len(rows), "supported": sum(y), "refuted": len(y) - sum(y),
                 "worksWithWikiEvidence": len({r["id"] for r in rows}), "sha256": snap_hash},
        "preRunMDE": {"estimatedCIHalfWidth": 0.21, "floor": 0.05,
                      "powered": False, "note": "MDE ~0.21 >> +0.05 floor at n=106/30-supported; reported BEFORE running per protocol"},
        "manipulationCheck": {"factBearingFraction": round(manip, 3), "floor": 0.50, "pass": manip >= 0.50},
        "incumbentHealth": {"lexicalCoverage": round(lex_cov, 4), "floor": 0.10,
                            "healthy": lex_cov >= 0.10,
                            "note": "guards against FEVER-style baseline collapse (there lexical cov was 0.003)"},
        "NLI_vs_lexical_deltaF1_by_seed": {s: df1[s]["deltaF1"] for s in df1},
        "ci95_seed0": df1[0]["ci95"],
        "coverageDrop": round(lex_cov - cov(nli), 4),
        "verdict": ("UNDERPOWERED (primary) — the sophia-domain non-forge pack (n=%d, %d supported) cannot power a "
                    "+0.05 ΔF1 gate (MDE ~0.21). Per the frozen protocol this is declared, not a GO/NO-GO. "
                    "Incumbent health + manipulation checks reported below; ΔF1 point estimate carries a wide CI. "
                    "PREREQUISITE WORK ITEM: build a larger curated evidence-groundable attribution pack "
                    "(labeled data is the binding constraint in sophia's domain, not the retriever or verifier)."
                    % (len(rows), sum(y))),
    }
    txt = json.dumps(report, indent=2)
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
