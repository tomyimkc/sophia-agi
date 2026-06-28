# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Regression guard for the vector-vs-keyword retrieval-recall benchmark.

Deterministic (local hashing embedder, exact-match scorer — no API key, no LLM judge),
so the measured delta is a stable invariant: the committed local vector index must keep
beating keyword-only recall over the same corpus. ``SOPHIA_RAG_BACKEND=keyword`` forces
the TRUE keyword tier (token-overlap; ``agent.retrieval._retrieve_keyword``) — this is the
documented contract (docs/09-Agent/Online-RAG.md: "keyword ⇒ keyword retrieve") and the
correct baseline for "does the dense vector index add value over plain keyword?".
If this flips, the embedding backend or the index regressed. Offline; numpy pytest job.
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
    # The validated retrieval delta: the committed dense vector index (local-hash-v1)
    # outranks TRUE keyword (token-overlap) on exact-record recall and MRR. The margin on
    # recall@5 is thin (≈0.517 vs ≈0.500) so this is asserted as >=, not strict >; the
    # cleaner signal is MRR (rank quality) and the nDCG guard in test_eval_search_quality.
    assert v["exact"]["recall@5"] >= k["exact"]["recall@5"]
    assert v["exact"]["mrr"] > k["exact"]["mrr"]
    # Neither backend is broken: both keep non-trivial exact-record recall.
    assert v["exact"]["recall@5"] >= 0.4
    assert k["exact"]["recall@5"] >= 0.3
    # Topical recall: the dense vectors match-or-beat keyword (robust to teacher-example
    # burial); vector should not regress below keyword here.
    assert v["topical"]["recall@5"] >= k["topical"]["recall@5"]
    assert v["topical"]["recall@5"] >= 0.9


def test_report_is_marked_candidate_not_validated():
    report = run(limit=6)
    assert report["candidateOnly"] is True and report["validated"] is False
    assert report["level3Evidence"] is False


def test_keyword_backend_runs_over_same_index():
    # Both backends must score; a zero-probe run would silently pass the delta asserts.
    k = score_backend("keyword", build_probes(limit=4))
    assert k["n"] == 4 and "exact" in k and "topical" in k


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
