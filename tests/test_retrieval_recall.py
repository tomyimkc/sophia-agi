# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Regression guard for the vector-vs-keyword retrieval-recall benchmark.

Deterministic (local hashing embedder, exact-match scorer — no API key, no LLM judge),
so the measured delta is a stable invariant: the committed local vector index must keep
beating keyword-only recall over the same corpus. If this flips, the embedding backend or
the index regressed. Offline; runs in the numpy-equipped pytest job.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_retrieval_recall import build_probes, run, score_backend  # noqa: E402


def test_probes_are_built_from_corpus():
    probes = build_probes()
    assert len(probes) >= 30
    assert all(p["q"] and p["gold"] for p in probes)


def test_vector_beats_keyword_on_both_views():
    report = run()
    v, k = report["vector_local"], report["keyword"]
    # Exact-record recall: vector should clearly outrank keyword (keyword can't break the
    # many tied full-overlap chunks; cosine discriminates).
    assert v["exact"]["recall@5"] > k["exact"]["recall@5"]
    assert v["exact"]["mrr"] > k["exact"]["mrr"]
    # Topical recall (robust to teacher-example burial): vector at least matches keyword.
    assert v["topical"]["recall@5"] >= k["topical"]["recall@5"]
    assert v["topical"]["mrr"] >= k["topical"]["mrr"]


def test_report_is_marked_candidate_not_validated():
    report = run(limit=6)
    assert report["candidateOnly"] is True and report["validated"] is False
    assert report["level3Evidence"] is False


def test_keyword_backend_runs_over_same_index():
    # Both backends must score; a zero-probe run would silently pass the delta asserts.
    # score_backend sets SOPHIA_RAG_BACKEND as a side effect (run() normally restores it);
    # called directly here, so save/restore to avoid leaking the backend into later tests.
    import os

    saved = os.environ.get("SOPHIA_RAG_BACKEND")
    try:
        k = score_backend("keyword", build_probes(limit=4))
    finally:
        if saved is None:
            os.environ.pop("SOPHIA_RAG_BACKEND", None)
        else:
            os.environ["SOPHIA_RAG_BACKEND"] = saved
    assert k["n"] == 4 and "exact" in k and "topical" in k


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
