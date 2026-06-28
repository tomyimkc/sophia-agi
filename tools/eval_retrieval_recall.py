#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Measure retrieval recall: local vector recall vs keyword-only, over the SAME index.

L2 put a committed `rag/index/embeddings.npz` + cosine search in the live retrieval path
(`agent.retrieval.retrieve`). This harness turns "vector recall is live" into a measured
delta: it builds labeled probes from the curated attribution corpus (each probe is a
natural-language question whose gold answer is a specific record/chunk), runs the query
through both backends over the identical indexed chunk set, and reports recall@1, recall@5,
and MRR for each — plus the vector-minus-keyword delta.

Fully offline and deterministic: the local hashing embedder (`agent.rag_local_embed`) needs
no API key, and exact-match-against-gold needs no LLM judge. Honest bounds: the probe set is
**self-authored** over the existing corpus (not a third-party retrieval benchmark), and the
gold is the canonical record id — so this validates the retrieval delta + the harness
end-to-end, not a headline capability. Reproduce: `python tools/eval_retrieval_recall.py`.

  python tools/eval_retrieval_recall.py            # full run, writes a candidate report
  python tools/eval_retrieval_recall.py --json     # machine-readable summary to stdout
  python tools/eval_retrieval_recall.py --limit 5  # quick smoke
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "agi-proof" / "benchmark-results" / "retrieval-recall.public-report.json"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")


def build_probes(limit: int | None = None) -> "list[dict]":
    """Natural-language probes whose gold is a specific attribution record (chunk title)."""
    data = json.loads((ROOT / "data" / "attributions.json").read_text(encoding="utf-8"))
    probes: list[dict] = []
    for key, rec in data.items():
        if not isinstance(rec, dict):
            continue
        title_en = rec.get("canonicalTitleEn") or key.replace("_", " ")
        gold = _norm(key)
        # Two surface forms per record: an authorship question and a tradition question.
        # Neither copies the indexed JSON body verbatim, so this is a fair recall probe.
        probes.append({"q": f"Who is the author of {title_en}?", "gold": gold, "kind": "author"})
        if rec.get("tradition"):
            probes.append({"q": f"Which tradition does {title_en} belong to?",
                           "gold": gold, "kind": "tradition"})
    if limit:
        probes = probes[:limit]
    return probes


def _ranks_for(query: str, gold: str, *, top_k: int) -> "tuple[int, int]":
    """(exactRank, topicalRank) 1-based ranks in retrieve()'s top_k, 0 if absent.

    exact = the retrieved chunk IS the gold record (title == gold key); topical = any
    retrieved chunk is ABOUT the gold record (its normalized title/path contains the key,
    e.g. a teacher example "...-deny-plato-dao-de-jing-r0"). Topical is robust to the corpus
    being dominated by teacher examples that bury the canonical record.
    """
    from agent.retrieval import retrieve

    exact = topical = 0
    for i, h in enumerate(retrieve(query, top_k=top_k), 1):
        title_n = _norm(getattr(h, "title", ""))
        path_n = _norm(getattr(h, "path", ""))
        if exact == 0 and title_n == gold:
            exact = i
        if topical == 0 and (gold in title_n or gold in path_n):
            topical = i
    return exact, topical


def _metrics(ranks: "list[int]") -> dict:
    n = len(ranks) or 1
    return {
        "recall@1": round(sum(1 for r in ranks if r == 1) / n, 4),
        "recall@5": round(sum(1 for r in ranks if 1 <= r <= 5) / n, 4),
        "mrr": round(sum((1.0 / r) for r in ranks if r > 0) / n, 4),
    }


def score_backend(backend: str, probes: "list[dict]", *, top_k: int = 5) -> dict:
    """Run all probes through one backend; return exact-record and topical metrics."""
    saved = os.environ.get("SOPHIA_RAG_BACKEND")
    try:
        os.environ["SOPHIA_RAG_BACKEND"] = backend
        exact_ranks: list[int] = []
        topical_ranks: list[int] = []
        for p in probes:
            e, t = _ranks_for(p["q"], p["gold"], top_k=top_k)
            exact_ranks.append(e)
            topical_ranks.append(t)
    finally:
        # Restore so direct callers (e.g. tests) don't leak the backend selection into the
        # rest of the process — ``run()`` already wraps this, but ``score_backend`` is also
        # called standalone and a leaked ``keyword`` backend would silently change every
        # later ``retrieve()``'s tier.
        if saved is None:
            os.environ.pop("SOPHIA_RAG_BACKEND", None)
        else:
            os.environ["SOPHIA_RAG_BACKEND"] = saved
    return {"backend": backend, "n": len(probes),
            "exact": _metrics(exact_ranks), "topical": _metrics(topical_ranks)}


def run(*, limit: int | None = None, top_k: int = 5) -> dict:
    saved = os.environ.get("SOPHIA_RAG_BACKEND")
    try:
        probes = build_probes(limit=limit)
        # "auto" resolves to the local vector embedder because the committed index is
        # local-hash-v1; "keyword" forces the lexical-overlap scorer over the same chunks.
        vector = score_backend("auto", probes, top_k=top_k)
        keyword = score_backend("keyword", probes, top_k=top_k)
    finally:
        if saved is None:
            os.environ.pop("SOPHIA_RAG_BACKEND", None)
        else:
            os.environ["SOPHIA_RAG_BACKEND"] = saved

    delta = {
        view: {m: round(vector[view][m] - keyword[view][m], 4)
               for m in ("recall@1", "recall@5", "mrr")}
        for view in ("exact", "topical")
    }
    return {
        "benchmark": "retrieval-recall (self-authored attribution probes)",
        "candidateOnly": True,
        "validated": False,
        "level3Evidence": False,
        "topK": top_k,
        "probes": len(probes),
        "vector_local": vector,
        "keyword": keyword,
        "vectorMinusKeyword": delta,
        "honestBound": ("Self-authored probe set over the existing live corpus; exact-match "
                        "scorer (no LLM judge). 'exact' gold = the canonical record; 'topical' "
                        "= any chunk about it (robust to the corpus being dominated by teacher "
                        "examples that bury the record). Validates the retrieval delta + "
                        "harness, not a headline capability. Both backends search the SAME index."),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Vector vs keyword retrieval recall")
    ap.add_argument("--json", action="store_true", help="print the summary as JSON to stdout")
    ap.add_argument("--limit", type=int, default=None, help="cap probe count (smoke runs)")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args(argv)

    report = run(limit=args.limit, top_k=args.top_k)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        v, k, d = report["vector_local"], report["keyword"], report["vectorMinusKeyword"]
        print(f"Probes: {report['probes']}  (top_k={report['topK']})")
        for view in ("exact", "topical"):
            print(f"  [{view}]")
            print(f"    vector(local)  recall@1={v[view]['recall@1']:.3f}  "
                  f"recall@5={v[view]['recall@5']:.3f}  mrr={v[view]['mrr']:.3f}")
            print(f"    keyword        recall@1={k[view]['recall@1']:.3f}  "
                  f"recall@5={k[view]['recall@5']:.3f}  mrr={k[view]['mrr']:.3f}")
            print(f"    delta (v-k)    recall@1={d[view]['recall@1']:+.3f}  "
                  f"recall@5={d[view]['recall@5']:+.3f}  mrr={d[view]['mrr']:+.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.json:
        print(f"Wrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
