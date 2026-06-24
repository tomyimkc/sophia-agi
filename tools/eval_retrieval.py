#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Retrieval-quality eval: recall@k and MRR over golden queries.

Measures whether the RAG layer actually surfaces the right source for a query, so
chunking/reranking changes are measurable instead of vibes. Offline (uses the
keyword retriever path; no embeddings/network needed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.retrieval import retrieve  # noqa: E402

# Each golden item: a query and a substring its correct source path should contain.
GOLDEN: list[dict[str, str]] = [
    {"query": "Did Confucius write the Dao De Jing?", "expect": "attributions.json"},
    {"query": "when did the Western Roman Empire fall", "expect": "history_events.json"},
    {"query": "do we only use 10 percent of our brain myth", "expect": "psychology_concepts.json"},
    {"query": "scripture attribution and sect boundaries in religion", "expect": "religion_concepts.json"},
]


def evaluate(golden: list[dict[str, str]], *, ks: tuple[int, ...] = (1, 3, 5), top_k: int = 8) -> dict[str, Any]:
    per_query = []
    recall_hits = {k: 0 for k in ks}
    rr_sum = 0.0
    for item in golden:
        chunks = retrieve(item["query"], top_k=top_k)
        paths = [c.path for c in chunks]
        rank = next((i + 1 for i, p in enumerate(paths) if item["expect"] in p), 0)
        for k in ks:
            if rank and rank <= k:
                recall_hits[k] += 1
        rr_sum += (1.0 / rank) if rank else 0.0
        per_query.append({"query": item["query"], "expect": item["expect"], "rank": rank, "topPaths": paths[:3]})
    n = len(golden)
    return {
        "queryCount": n,
        "recallAtK": {f"@{k}": round(recall_hits[k] / n, 3) if n else 0.0 for k in ks},
        "mrr": round(rr_sum / n, 3) if n else 0.0,
        "perQuery": per_query,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval recall@k / MRR eval")
    parser.add_argument("golden", nargs="?", type=Path, default=None, help="golden JSON [{query,expect}]; defaults to built-in")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    golden = json.loads(args.golden.read_text(encoding="utf-8")) if args.golden else GOLDEN
    report = evaluate(golden, top_k=args.top_k)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
