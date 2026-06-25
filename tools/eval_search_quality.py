#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Search-quality benchmark: graded nDCG / recall / MRR + a badcase taxonomy.

The JD asks for a *搜索质量评估体系* — automated evaluation plus badcase analysis that makes
algorithm iteration "clear and explainable". This harness is that, over Sophia's corpus:

  - reuses the self-authored attribution probes (``tools.eval_retrieval_recall.build_probes``);
  - scores three backends over the SAME committed index — **keyword** (lexical overlap),
    **vector** (dense cosine, ``local-hash-v1``), and **hybrid** (dense+sparse RRF,
    ``agent.hybrid_retrieval``);
  - reports **graded** metrics: recall@k, MRR, and **nDCG@k** (exact record = gain 3, any
    chunk about it = gain 1), so quality — not just hit/miss — is visible;
  - mines a **badcase taxonomy** so every miss is attributable to a stage:
      * ``lexical_gap``    — keyword misses the exact record but a vector view finds it;
      * ``semantic_gap``   — vector misses but keyword/hybrid finds it (hash-embedding blur);
      * ``tied_burial``    — exact is retrievable (in the pool) but buried below top-k;
      * ``absent_from_pool`` — no backend surfaces the exact record (a real recall hole).

Fully offline & deterministic: local hashing embedder (no API key), exact-match-against-gold
(no LLM judge). Honest bound: probes are **self-authored** over the existing corpus, and
nDCG's IDCG is computed by **pooling** (the ideal is the best ordering of gains any backend
surfaced for that probe) — so this validates the ranking deltas + the harness end-to-end,
not a third-party headline number. Reproduce: ``python tools/eval_search_quality.py``.

  python tools/eval_search_quality.py            # full run, writes a candidate report
  python tools/eval_search_quality.py --json      # machine-readable summary to stdout
  python tools/eval_search_quality.py --limit 8   # quick smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_retrieval_recall import _norm, build_probes  # noqa: E402

OUT_PATH = ROOT / "agi-proof" / "benchmark-results" / "search-quality.public-report.json"

# Backends scored for metrics. The badcase taxonomy reasons over the core three (keyword /
# vector / hybrid); hybrid_dedup is an ablation that adds near-duplicate collapse on top of
# hybrid, so its delta vs hybrid isolates what dedup buys.
BACKENDS = ("keyword", "vector", "hybrid", "hybrid_dedup")
TAXONOMY_BACKENDS = ("keyword", "vector", "hybrid")
EXACT_GAIN = 3.0
TOPICAL_GAIN = 1.0


def _gain(title: str, path: str, gold: str) -> float:
    """Graded relevance of a retrieved chunk to ``gold`` (the canonical record key)."""
    title_n, path_n = _norm(title), _norm(path)
    if title_n == gold:
        return EXACT_GAIN
    if gold in title_n or gold in path_n:
        return TOPICAL_GAIN
    return 0.0


def _hits(backend: str, query: str, *, top_k: int):
    """Run one backend, returning its top-k hits (objects with .title/.path)."""
    if backend in {"hybrid", "hybrid_dedup"}:
        os.environ["SOPHIA_RAG_BACKEND"] = "auto"
        from agent.hybrid_retrieval import retrieve_hybrid

        return retrieve_hybrid(query, top_k=top_k, dedupe=(backend == "hybrid_dedup"))
    os.environ["SOPHIA_RAG_BACKEND"] = "auto" if backend == "vector" else "keyword"
    from agent.retrieval import retrieve

    return retrieve(query, top_k=top_k)


def _dcg(gains: "list[float]") -> float:
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains, 1) if g > 0)


def _metrics(gains_by_probe: "list[list[float]]", pooled_by_probe: "list[list[float]]", *, top_k: int) -> dict:
    """Graded recall@k / MRR / nDCG@k from per-probe gain lists (rank order) + pooled gains."""
    n = len(gains_by_probe) or 1
    recall = mrr = ndcg = 0.0
    for gains, pooled in zip(gains_by_probe, pooled_by_probe):
        topk = gains[:top_k]
        # recall@k / MRR keyed on the EXACT record (gain == EXACT_GAIN).
        first_exact = next((i for i, g in enumerate(topk, 1) if g >= EXACT_GAIN), 0)
        if first_exact:
            recall += 1.0
            mrr += 1.0 / first_exact
        idcg = _dcg(sorted(pooled, reverse=True)[:top_k])
        if idcg > 0:
            ndcg += _dcg(topk) / idcg
    return {
        f"recall@{top_k}": round(recall / n, 4),
        "mrr": round(mrr / n, 4),
        f"ndcg@{top_k}": round(ndcg / n, 4),
    }


def run(*, limit: int | None = None, top_k: int = 5) -> dict:
    saved = os.environ.get("SOPHIA_RAG_BACKEND")
    probes = build_probes(limit=limit)
    # Pool depth ≥ top_k so IDCG can see relevant chunks a backend buried just past top_k.
    pool_k = max(top_k * 4, 20)
    try:
        # Per backend, per probe: the gain list in rank order (depth pool_k).
        per_backend: dict[str, list[list[float]]] = {}
        for backend in BACKENDS:
            rows: list[list[float]] = []
            for p in probes:
                hits = _hits(backend, p["q"], top_k=pool_k)
                rows.append([_gain(getattr(h, "title", ""), getattr(h, "path", ""), p["gold"]) for h in hits])
            per_backend[backend] = rows
    finally:
        if saved is None:
            os.environ.pop("SOPHIA_RAG_BACKEND", None)
        else:
            os.environ["SOPHIA_RAG_BACKEND"] = saved

    # Pooled ideal gains per probe = best (highest) gain seen at each position across backends,
    # collapsed to the multiset of distinct relevant grades any backend surfaced.
    pooled: list[list[float]] = []
    for i in range(len(probes)):
        bag: list[float] = []
        seen_exact = False
        topical = 0
        for backend in BACKENDS:
            for g in per_backend[backend][i]:
                if g >= EXACT_GAIN:
                    seen_exact = True
                elif g > 0:
                    topical += 1
        if seen_exact:
            bag.append(EXACT_GAIN)
        bag.extend([TOPICAL_GAIN] * min(topical, pool_k))
        pooled.append(bag or [0.0])

    metrics = {b: _metrics(per_backend[b], pooled, top_k=top_k) for b in BACKENDS}
    taxonomy = _badcase_taxonomy(probes, per_backend, top_k=top_k)
    best = max(BACKENDS, key=lambda b: (metrics[b][f"ndcg@{top_k}"], metrics[b][f"recall@{top_k}"]))

    return {
        "benchmark": "search-quality (self-authored attribution probes; graded nDCG)",
        "candidateOnly": True,
        "validated": False,
        "level3Evidence": False,
        "topK": top_k,
        "poolK": pool_k,
        "probes": len(probes),
        "metrics": metrics,
        "bestBackend": best,
        "badcaseTaxonomy": taxonomy,
        "honestBound": (
            "Self-authored probes over the live corpus; exact-match scorer (no LLM judge). "
            "nDCG IDCG is POOLED (ideal = best gain ordering any backend surfaced), so metrics "
            "validate the ranking deltas + the harness, not a third-party headline. "
            "hybrid_dedup = hybrid + near-duplicate collapse (ablation). All backends search "
            "the SAME committed index."
        ),
    }


def _badcase_taxonomy(probes, per_backend, *, top_k: int) -> dict:
    """Classify every probe where the BEST single backend still missed the exact record."""
    counts = {"lexical_gap": 0, "semantic_gap": 0, "tied_burial": 0, "absent_from_pool": 0}
    examples: dict[str, list[str]] = {k: [] for k in counts}

    def _exact_rank(gains: "list[float]", k: int) -> int:
        return next((i for i, g in enumerate(gains[:k], 1) if g >= EXACT_GAIN), 0)

    for i, p in enumerate(probes):
        kw, vec, hyb = (per_backend[b][i] for b in TAXONOMY_BACKENDS)
        hit_top = {b: _exact_rank(per_backend[b][i], top_k) > 0 for b in TAXONOMY_BACKENDS}
        if any(hit_top.values()):
            # Surface asymmetries even when *some* backend wins — these are the actionable
            # badcases (one view's blind spot that fusion should cover).
            if not hit_top["keyword"] and (hit_top["vector"] or hit_top["hybrid"]):
                _bump(counts, examples, "lexical_gap", p)
            if not hit_top["vector"] and (hit_top["keyword"] or hit_top["hybrid"]):
                _bump(counts, examples, "semantic_gap", p)
            continue
        # No backend put the exact record in top-k. Is it retrievable at all (in the pool)?
        in_pool = any(_exact_rank(g, len(g)) > 0 for g in (kw, vec, hyb))
        _bump(counts, examples, "tied_burial" if in_pool else "absent_from_pool", p)

    return {"counts": counts, "examples": {k: v[:5] for k, v in examples.items()}}


def _bump(counts: dict, examples: dict, key: str, probe: dict) -> None:
    counts[key] += 1
    if len(examples[key]) < 5:
        examples[key].append(probe["q"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Search-quality benchmark (graded nDCG + badcases)")
    ap.add_argument("--json", action="store_true", help="print the summary as JSON to stdout")
    ap.add_argument("--limit", type=int, default=None, help="cap probe count (smoke runs)")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args(argv)

    report = run(limit=args.limit, top_k=args.top_k)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        k = report["topK"]
        print(f"Probes: {report['probes']}  (top_k={k}, pool_k={report['poolK']})")
        for backend in BACKENDS:
            m = report["metrics"][backend]
            print(f"  {backend:8s}  recall@{k}={m[f'recall@{k}']:.3f}  "
                  f"mrr={m['mrr']:.3f}  ndcg@{k}={m[f'ndcg@{k}']:.3f}")
        print(f"  best backend: {report['bestBackend']}")
        print("  badcase taxonomy:")
        for name, c in report["badcaseTaxonomy"]["counts"].items():
            print(f"    {name:18s} {c}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.json:
        print(f"Wrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
