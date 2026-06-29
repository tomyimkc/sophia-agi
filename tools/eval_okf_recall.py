#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""OKF provenance-aware multi-hop recall benchmark (first-party, decontaminated).

This is the OKF analogue of a graph-RAG recall benchmark (cf. Zleap-AI/SAG's
Recall@K on HotpotQA / 2WikiMultiHop / MuSiQue), with one metric a recall-only engine
cannot report: **provenance-faithfulness** — does the surfaced result carry the correct
provenance verdict (capped vs not) for the answer the corpus actually licenses?

Pipeline (fully offline, deterministic — no API key, no LLM):
  1. DECONTAM gate — every probe query must be shingle-disjoint from its gold page body
     (Jaccard < threshold), so we measure retrieval, not leakage. Reuses the repo's
     shared decontam primitives (tools.assert_decontam / provenance_bench.dataset_guard).
  2. EXTRACT — okf.extract.extract_events over wiki/ + entity index.
  3. RECALL — okf.extract.multi_hop_recall per probe.
  4. METRICS — Recall@{1,3,5}, MRR, multi-hop reach, and provenance-faithfulness.

HONEST BOUND (mirrors tools/eval_search_quality.py): the probes are **self-authored**
over the committed wiki/ corpus. This validates the harness, the entity-index recall,
and the provenance-floor propagation end-to-end on first-party data — it is NOT a
third-party headline number. A decontaminated HotpotQA/2Wiki/MuSiQue run remains a gated
TODO in the failure ledger; no cross-system claim is made here. The gates decide
validity, never this script.

    python tools/eval_okf_recall.py            # human report
    python tools/eval_okf_recall.py --json     # machine-readable summary to stdout
    python tools/eval_okf_recall.py --jaccard 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from okf import extract, page as okf_page  # noqa: E402
from okf.trace import format_trace  # noqa: E402
from provenance_bench.dataset_guard import normalize  # noqa: E402
from tools.assert_decontam import _jaccard, _shingles  # noqa: E402

PROBES = ROOT / "eval" / "okf_recall" / "probes.jsonl"
WIKI = ROOT / "wiki"
RECALL_KS = (1, 3, 5)


def _load_probes(path: Path) -> "list[dict]":
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _decontam(probes, pages, *, jaccard: float, k: int) -> "list[dict]":
    """Flag any probe whose query near-duplicates its gold page body (would be leakage)."""
    body_by_id = {p.id: p.body for p in pages}
    leaks: list[dict] = []
    for pr in probes:
        gold = pr["goldPage"]
        body = body_by_id.get(gold, "")
        sim = _jaccard(_shingles(pr["query"], k), _shingles(body, k))
        if sim >= jaccard:
            leaks.append({"id": pr["id"], "gold": gold, "jaccard": round(sim, 3)})
    return leaks


def _rank_of(hits, gold: str) -> "int | None":
    for i, h in enumerate(hits, 1):
        if h.event.page_id == gold:
            return i
    return None


def evaluate(*, jaccard: float = 0.6, shingle: int = 5, top_k: int = 5) -> dict:
    probes = _load_probes(PROBES)
    pages = okf_page.load_pages(WIKI)

    leaks = _decontam(probes, pages, jaccard=jaccard, k=shingle)

    events = extract.extract_events(pages)
    index = extract.build_entity_index(events)

    # Ablation: direct lexical recall (max_hops=0) vs provenance-aware 2-hop. This is the
    # shape of a graph-RAG claim ("multi-hop > vector-only") — reported honestly as a
    # first-party delta, not a cross-system headline.
    def _recall_at_5(max_hops: int) -> float:
        hit = 0
        for pr in probes:
            hits = extract.multi_hop_recall(pr["query"], events, index=index,
                                            max_hops=max_hops, top_k=5)
            if _rank_of(hits, pr["goldPage"]) is not None:
                hit += 1
        return hit / len(probes) if probes else 0.0

    ablation = {"directOnly_R@5": round(_recall_at_5(0), 3),
                "twoHop_R@5": round(_recall_at_5(2), 3)}

    per: list[dict] = []
    for pr in probes:
        hits = extract.multi_hop_recall(pr["query"], events, index=index,
                                        max_hops=2, top_k=top_k)
        gold = pr["goldPage"]
        rank = _rank_of(hits, gold)
        gold_hit = next((h for h in hits if h.event.page_id == gold), None)
        per.append({
            "id": pr["id"],
            "gold": gold,
            "rank": rank,
            "hops": (gold_hit.hops if gold_hit else None),
            "provenanceFloor": (gold_hit.provenance_floor if gold_hit else None),
            "capped": (gold_hit.capped if gold_hit else None),
            "weakPathGold": bool(pr.get("weakPath")),
            # faithful == recall's capped verdict matches the corpus's ground-truth weakness
            "faithful": (gold_hit is not None and gold_hit.capped == bool(pr.get("weakPath"))),
        })

    n = len(per)
    found = [r for r in per if r["rank"] is not None]
    recall_at = {k: sum(1 for r in found if r["rank"] <= k) / n for k in RECALL_KS}
    mrr = sum((1.0 / r["rank"]) for r in found) / n if n else 0.0
    multihop = [r for r in found if (r["hops"] or 0) >= 1]
    faithful = [r for r in per if r["faithful"]]

    return {
        "probes": n,
        "decontam": {"jaccard_threshold": jaccard, "shingle_k": shingle,
                     "leaks": leaks, "clean": not leaks},
        "metrics": {
            "recallAt": {str(k): round(v, 3) for k, v in recall_at.items()},
            "mrr": round(mrr, 3),
            "multiHopReach": round(len(multihop) / n, 3),
            "provenanceFaithfulness": round(len(faithful) / n, 3),
            "ablation": ablation,
        },
        "perProbe": per,
        "honestBound": ("self-authored first-party probes over wiki/; validates the "
                        "harness + provenance-floor propagation end-to-end, NOT a "
                        "third-party headline. HotpotQA/2Wiki/MuSiQue remain a gated TODO."),
    }


def _print_human(result: dict) -> None:
    d = result["decontam"]
    print("OKF provenance-aware recall benchmark")
    print("=" * 64)
    status = "CLEAN" if d["clean"] else f"LEAKS: {d['leaks']}"
    print(f"decontam (jaccard<{d['jaccard_threshold']}, k={d['shingle_k']}): {status}")
    m = result["metrics"]
    print(f"probes={result['probes']}  "
          f"R@1={m['recallAt']['1']}  R@3={m['recallAt']['3']}  R@5={m['recallAt']['5']}  "
          f"MRR={m['mrr']}")
    print(f"multi-hop reach={m['multiHopReach']}  "
          f"provenance-faithfulness={m['provenanceFaithfulness']}")
    ab = m["ablation"]
    print(f"ablation R@5: direct-only={ab['directOnly_R@5']} -> 2-hop={ab['twoHop_R@5']}")
    print("-" * 64)
    for r in result["perProbe"]:
        ok = "ok " if r["faithful"] else "!! "
        cap = "capped" if r["capped"] else ("clear" if r["capped"] is not None else "MISS ")
        print(f"  {ok}{r['id']:24} rank={str(r['rank']):>4} hops={str(r['hops']):>4} "
              f"floor={str(r['provenanceFloor']):>4} {cap:>6} "
              f"(weakGold={r['weakPathGold']})")
    print("-" * 64)
    print(result["honestBound"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="machine-readable summary to stdout")
    ap.add_argument("--jaccard", type=float, default=0.6, help="decontam Jaccard threshold")
    ap.add_argument("--shingle", type=int, default=5, help="decontam shingle size (words)")
    ap.add_argument("--top-k", type=int, default=5, help="recall depth")
    ap.add_argument("--trace", metavar="QUERY", help="print a provenance trace for one query and exit")
    args = ap.parse_args()

    if args.trace:
        events = extract.extract_events(okf_page.load_pages(WIKI))
        hits = extract.multi_hop_recall(args.trace, events, top_k=args.top_k)
        print(format_trace(args.trace, hits))
        return 0

    result = evaluate(jaccard=args.jaccard, shingle=args.shingle, top_k=args.top_k)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_human(result)
    # Non-zero exit if a probe leaked (a real contamination failure), else clean.
    return 1 if not result["decontam"]["clean"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
