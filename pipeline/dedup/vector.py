# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Vector (embedding) near-duplicate detection (Phase 2).

MinHash catches lexical near-dups; a vector pass catches paraphrase/semantic near-dups the
JD also asks for ("向量去重"). It reuses Sophia's committed offline embedder
(``agent.rag_local_embed``, a deterministic ``blake2b`` hashing embedder) so the pass stays
airgap-safe and reproducible — no API key, no neural model download.

numpy is an optional dependency here (already required wherever the RAG index is built); if
it is unavailable, ``available()`` returns False and callers should fall back to MinHash only.
Greedy single-linkage clustering by cosine similarity keeps it dependency-light.
"""

from __future__ import annotations


def available() -> bool:
    """True iff the numpy-backed embedder can be imported (airgap-safe, no network)."""
    try:
        import numpy  # noqa: F401

        from agent import rag_local_embed  # noqa: F401
    except Exception:
        return False
    return True


def _cosine(a, b) -> float:
    import numpy as np

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def cluster_vectors(vectors, *, threshold: float = 0.9) -> list[int]:
    """Greedy single-linkage clustering of L2-comparable vectors by cosine similarity.

    Returns ``cluster_ids`` aligned to input order (each id is a representative index).
    """
    reps: list[tuple[int, object]] = []  # (representative index, vector)
    cluster_ids: list[int] = []
    for i, vec in enumerate(vectors):
        assigned = None
        for rep_idx, rep_vec in reps:
            if _cosine(vec, rep_vec) >= threshold:
                assigned = rep_idx
                break
        if assigned is None:
            reps.append((i, vec))
            assigned = i
        cluster_ids.append(assigned)
    return cluster_ids


def cluster_documents(docs, *, threshold: float = 0.9) -> list[int]:
    """Embed each doc's ``content`` with the offline embedder and cluster by cosine.

    Raises ``RuntimeError`` if numpy/embedder are unavailable (check ``available()`` first).
    """
    if not available():
        raise RuntimeError("vector dedup requires numpy + agent.rag_local_embed")
    from agent import rag_local_embed

    docs = list(docs)
    vectors = [rag_local_embed.embed_text(doc.get("content") or "") for doc in docs]
    return cluster_vectors(vectors, threshold=threshold)


__all__ = ["available", "cluster_vectors", "cluster_documents"]
