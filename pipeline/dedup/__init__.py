# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deduplication stage (Phase 2): URL hygiene + MinHash near-dup (+ optional vector near-dup).

``dedup_documents`` is the pipeline stage: it canonicalizes each document's URL
(``pipeline.url_canonical``), then clusters near-duplicate *content* with MinHash-LSH
(``pipeline.dedup.minhash``). Each input doc gets a ``doc['dedup']`` block
(``{sim_cluster, is_duplicate, minhash_sig}``) and a ``doc['canonical_url']``; the stage
returns the kept (non-duplicate) documents plus a stats dict. Vector near-dup
(``pipeline.dedup.vector``) is available as a stronger second pass when numpy is present.
"""

from __future__ import annotations

from pipeline import url_canonical
from pipeline.dedup import minhash as _mh

__all__ = ["dedup_documents"]


def dedup_documents(docs, *, threshold: float = 0.8, num_perm: int = 64, bands: int = 16) -> dict:
    """Canonicalize URLs and remove near-duplicate documents by content.

    Mutates each doc in ``docs`` (sets ``canonical_url`` and ``dedup``). Returns
    ``{"kept": [...], "removed": [...], "stats": {...}}``. The first document in each
    similarity cluster is kept; the rest are flagged ``is_duplicate``.
    """
    docs = list(docs)
    for doc in docs:
        url_canonical.annotate(doc)

    texts = [doc.get("content") or "" for doc in docs]
    clusters, sigs = _mh.cluster(texts, threshold=threshold, num_perm=num_perm, bands=bands)

    seen: set[int] = set()
    kept, removed = [], []
    for i, doc in enumerate(docs):
        cluster_id = clusters[i]
        is_dup = cluster_id in seen
        seen.add(cluster_id)
        doc["dedup"] = {
            "sim_cluster": f"c{cluster_id}",
            "is_duplicate": is_dup,
            "minhash_sig": list(sigs[i]),
        }
        (removed if is_dup else kept).append(doc)

    n = len(docs)
    stats = {
        "input": n,
        "kept": len(kept),
        "removed": len(removed),
        "clusters": len(set(clusters)),
        "dedupRatio": round(len(removed) / n, 6) if n else 0.0,
        "threshold": threshold,
    }
    return {"kept": kept, "removed": removed, "stats": stats}
