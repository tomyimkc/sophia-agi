#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Redirect-and-fund capstone: an INDEPENDENT evidence corpus ingested into a retrieval path, with
the curated pack as its HELD-OUT acceptance instrument. Two arms, per the maintainer-AI's amended gate.

Corpus: independent multi-sentence Wikipedia lead passages for ~90 canonically-attributed works
(curated by a separate agent; curator != gate-runner). Claims: BLUNT attributions ('{author} wrote
{work}') to match Wikipedia's blunt lead prose, supported (canonical) + refuted (sibling author).

Arm 1 — evidence-as-curated: attach the work's own passages directly (mechanism CEILING in-domain).
Arm 2 — evidence-through-RETRIEVAL: retrieve top-k passages from the whole corpus by the real pipeline
        (sophia's production hash index AND a semantic index = deployable-if-upgraded). This is the
        DEPLOYABLE system — default-on requires THIS arm to pass, since over-abstention lives here.

Frozen gate: NLI (build_nli_entailment) vs the HEALTHY semantic incumbent (build_semantic_entailment)
through the real fact_check_gate.external_ground. Guards: paired ΔF1 @ matched coverage ≥ +0.05 (CI∌0,
3 seeds); incumbent-health cov ≥ 0.10; ANSWERABLE-coverage drop ≤ 0.01 (load-bearing on the retrieval
arm); fail-closed; manipulation ≥50% fact-bearing (retrieval arm). MDE pre-computed from the achieved n.
Outcomes: RETRIEVAL arm GO -> default-on candidate; arm1 pass & arm2 fail -> 'mechanism confirmed
in-domain, retrieval still the blocker'; both NO-GO w/ power -> close. candidateOnly, canClaimAGI=false.
"""
from __future__ import annotations
import argparse, hashlib, json, random, re, ssl, sys, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
from agent.fact_check_gate import AtomicClaim, EvidenceSource, external_ground, classify_claim, risk_for
from agent.nli_grounding import build_nli_entailment, build_semantic_entailment

CLAIMS = "/tmp/corpus-data/claims.json"
_HEDGE = "{a} wrote {w}"   # blunt claims to match Wikipedia's blunt attribution prose


def _summary(title, timeout=10):
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title.replace(" ", "_"))
    req = urllib.request.Request(url, headers={"User-Agent": "sophia-bench/1.0"})
    d = json.load(urllib.request.urlopen(req, timeout=timeout, context=ssl.create_default_context()))
    return d.get("extract", "") if d.get("type") != "disambiguation" else ""


def _opensearch(query, timeout=10):
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "opensearch", "search": query, "limit": "1", "namespace": "0", "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": "sophia-bench/1.0"})
    d = json.load(urllib.request.urlopen(req, timeout=timeout, context=ssl.create_default_context()))
    return d[1][0] if len(d) > 1 and d[1] else ""


def wiki_lead(title, timeout=12, work=None):
    import time
    for attempt in range(3):
        for t in [title, work]:
            if not t:
                continue
            try:
                ex = _summary(t, timeout)
                if ex:
                    return ex
            except Exception:
                pass
        for q in [title, work]:
            if not q:
                continue
            try:
                best = _opensearch(q, timeout)
                if best:
                    ex = _summary(best, timeout)
                    if ex:
                        return ex
            except Exception:
                pass
        time.sleep(1.5 * (attempt + 1))
    return ""


def sents(text):
    return [s.strip() for s in re.split(r"(?<=[.])\s+", text or "") if len(s.strip()) >= 30][:5]


def factbearing(s):
    return len(s) >= 30 and bool(re.search(r"\b(is|was|are|were|wrote|written|by|author|published|philosopher|treatise|work|book)\b", s, re.I))


# --- retrieval indices over the corpus passages ---
class SemIndex:
    def __init__(self, passages):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.p = passages
        self.E = self.m.encode([t for t, _ in passages], normalize_embeddings=True)

    def topk(self, q, k=4):
        qe = self.m.encode(q, normalize_embeddings=True)
        sims = self.E @ qe
        return [self.p[i] for i in np.argsort(-sims)[:k]]


class HashIndex:
    """sophia's PRODUCTION lexical retrieval stand-in (local-hash-v1 style bag-of-token-hashes)."""
    def __init__(self, passages, dim=256):
        self.p = passages; self.dim = dim
        self.E = np.stack([self._emb(t) for t, _ in passages])

    def _emb(self, text):
        v = np.zeros(self.dim)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.blake2b(tok.encode(), digest_size=8).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 1) & 1 else -1.0
        n = np.linalg.norm(v); return v / n if n else v

    def topk(self, q, k=4):
        qe = self._emb(q); sims = self.E @ qe
        return [self.p[i] for i in np.argsort(-sims)[:k]]


def srcs(passages):
    return [EvidenceSource(id=f"e{i}", url=f"https://ext{i}.example.org/x", snippet=t,
                           publisher=f"ext{i}", source_type="web") for i, (t, _wid) in enumerate(passages)]


def admission_score(res):
    return res.confidence if res.verdict == "accepted" else (-res.confidence if res.verdict == "rejected" else 0.0)


def f1_topk(scores, y, k):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    tp = sum(1 for i in order if y[i]); pos = sum(y)
    p = tp / k if k else 0.0; r = tp / pos if pos else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def paired_df1(arm, base, y, cov, n_boot, seed):
    rng = random.Random(seed); n = len(y); k = max(1, round(cov * n)); d = []
    v = f1_topk(arm, y, k) - f1_topk(base, y, k)
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        yb = [y[i] for i in idx]
        if len(set(yb)) < 2:
            continue
        kb = max(1, round(cov * len(idx)))
        d.append(f1_topk([arm[i] for i in idx], yb, kb) - f1_topk([base[i] for i in idx], yb, kb))
    d.sort()
    return {"deltaF1": round(v, 4), "ci95": [round(d[int(.025*len(d))], 4), round(d[int(.975*len(d))], 4)]}


def mde_halfwidth(n, npos, seed=0, n_boot=1500, reps=25):
    rng = random.Random(seed); hw = []
    for _ in range(reps):
        y = [1]*npos + [0]*(n-npos); rng.shuffle(y)
        base = [(0.44 if yy else 0.0) + rng.gauss(0, 1) for yy in y]
        arm = [(0.44 if yy else 0.0) + rng.gauss(0, 1) for yy in y]
        cov = npos/n; k = max(1, round(cov*n)); d = []
        for _ in range(n_boot):
            idx = [rng.randrange(n) for _ in range(n)]; yb = [y[i] for i in idx]
            if len(set(yb)) < 2: continue
            kb = max(1, round(cov*len(idx))); d.append(f1_topk([arm[i] for i in idx], yb, kb)-f1_topk([base[i] for i in idx], yb, kb))
        d.sort(); hw.append((d[int(.975*len(d))]-d[int(.025*len(d))])/2)
    return round(sum(hw)/len(hw), 3)


def gate_arm(rows, evidence_fn, n_boot):
    """rows carry claim + supported + work_id; evidence_fn(row) -> list[(passage, work_id)]."""
    y = [r["supported"] for r in rows]
    nli_fn = build_nli_entailment(); sem_fn = build_semantic_entailment()
    units = [p for r in rows for p in evidence_fn(r)]
    manip = sum(1 for t, _ in units if factbearing(t)) / max(len(units), 1)

    def arm(entail):
        out = []
        for r in rows:
            ev = evidence_fn(r)
            c = AtomicClaim(text=r["claim"], type=classify_claim(r["claim"]), risk=risk_for(r["claim"]))
            s = srcs(ev)
            out.append(admission_score(external_ground(c, (lambda cc, _s=s: list(_s)), entailment=entail)))
        return out
    sem = arm(sem_fn); nli = arm(nli_fn)
    cov = lambda s: sum(1 for x in s if x > 0) / len(s)
    ans = lambda s: sum(1 for sc, yy in zip(s, y) if yy and sc > 0) / max(sum(y), 1)
    sem_cov, matched = cov(sem), max(0.1, cov(sem))
    df1 = {s: paired_df1(nli, sem, y, matched, n_boot, s) for s in (0, 1, 7)}
    return {"n": len(rows), "supported": sum(y), "manipFactBearing": round(manip, 3),
            "incumbentSemanticCoverage": round(sem_cov, 4), "incumbentHealthy": sem_cov >= 0.10,
            "answerableCovSemantic": round(ans(sem), 4), "answerableCovNLI": round(ans(nli), 4),
            "answerableCovDrop": round(ans(sem) - ans(nli), 4), "abstentionGuardPass": (ans(sem) - ans(nli)) <= 0.01,
            "NLI_vs_semantic_deltaF1": {s: df1[s]["deltaF1"] for s in df1}, "ci95_seed0": df1[0]["ci95"],
            "primaryPass": all(df1[s]["ci95"][0] > 0 and df1[s]["deltaF1"] >= 0.05 for s in df1)}


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--output", default=None)
    a = ap.parse_args(argv)

    items = json.load(open(CLAIMS))
    # fetch independent Wikipedia evidence per work
    def fetch(it):
        return (it, sents(wiki_lead(it.get("wikipedia_title"), work=it["work"])))
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        fetched = list(ex.map(fetch, items))
    corpus = []            # (passage, work_id)
    ev_by_work = {}
    for it, ss in fetched:
        wid = it["work"]
        ev_by_work[wid] = [(s, wid) for s in ss]
        corpus += ev_by_work[wid]
    # build claims (supported + refuted), keep only works that got evidence
    rows = []
    for it, ss in fetched:
        if not ss:
            continue
        w = it["work"]
        rows.append({"claim": _HEDGE.format(a=it["correct_author"], w=w), "supported": True, "work_id": w})
        rows.append({"claim": _HEDGE.format(a=it["wrong_author"], w=w), "supported": False, "work_id": w})
    print(f"works with evidence: {len(ev_by_work)} | claims: {len(rows)} | corpus passages: {len(corpus)}", file=sys.stderr)

    sem_idx = SemIndex(corpus); hash_idx = HashIndex(corpus)
    ev_curated = lambda r: ev_by_work[r["work_id"]]
    ev_sem = lambda r: sem_idx.topk(r["claim"], 4)
    ev_hash = lambda r: hash_idx.topk(r["claim"], 4)

    npos = sum(r["supported"] for r in rows)
    report = {"schema": "sophia.evidence_corpus_gate.v1", "candidateOnly": True, "canClaimAGI": False,
              "corpus": {"works": len(ev_by_work), "passages": len(corpus), "evidence": "independent Wikipedia leads (multi-sentence), NOT sophia's own wiki"},
              "n": len(rows), "supported": npos,
              "preRunMDE": {"estimatedCIHalfWidth": mde_halfwidth(len(rows), npos), "floor": 0.05},
              "arm1_evidence_as_curated": gate_arm(rows, ev_curated, a.n_boot),
              "arm2_retrieval_semantic": gate_arm(rows, ev_sem, a.n_boot),
              "arm2_retrieval_hash_production": gate_arm(rows, ev_hash, a.n_boot)}
    r2 = report["arm2_retrieval_semantic"]
    default_on = r2["primaryPass"] and r2["abstentionGuardPass"] and r2["incumbentHealthy"] and r2["manipFactBearing"] >= 0.5
    a1 = report["arm1_evidence_as_curated"]
    report["verdict"] = (
        "DEFAULT-ON CANDIDATE — retrieval arm passes the frozen gate" if default_on else
        ("MECHANISM CONFIRMED IN-DOMAIN, RETRIEVAL STILL THE BLOCKER — arm1 (curated) passes but arm2 (retrieval) does not"
         if (a1["primaryPass"] and a1["abstentionGuardPass"]) else
         "NO-GO — mechanism does not clear the gate even on curated in-domain evidence")) + \
        f" | MDE {report['preRunMDE']['estimatedCIHalfWidth']} (floor 0.05)"
    txt = json.dumps(report, indent=2)
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
