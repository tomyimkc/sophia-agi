# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Regression guard for the vector-vs-keyword retrieval-recall benchmark.

Deterministic (local hashing embedder, exact-match scorer — no API key, no LLM judge),
so the measured relationship is a stable invariant. The honest relationship (measured):
on short attribution probes that share surface tokens with the gold record, **lexical
keyword leads exact-record recall** (the committed local-hash embedder is a weak semantic
proxy — see README: the Gemini backend is the higher-quality option), while on **topical
relevance the dense vectors match keyword**. These deterministic deltas are the guard: if
they shift materially, the embedder or index changed. Offline; runs in the numpy pytest job.
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


def test_keyword_leads_exact_vectors_match_topical():
    report = run()
    v, k = report["vector_local"], report["keyword"]
    # Exact-record recall: lexical KEYWORD leads on these token-overlapping probes; the
    # offline hash vector is a weak semantic proxy (a learned/Gemini backend would change
    # this — see README). This is the measured, deterministic relationship, not the
    # earlier (incorrect, never-true) "vector beats keyword" assertion.
    assert k["exact"]["recall@5"] > v["exact"]["recall@5"]
    assert k["exact"]["mrr"] > v["exact"]["mrr"]
    # Neither backend is broken: both keep non-trivial exact-record recall.
    assert k["exact"]["recall@5"] >= 0.8
    assert v["exact"]["recall@5"] >= 0.4
    # Topical recall: the dense vectors catch up to keyword (parity within a small margin).
    assert v["topical"]["recall@5"] >= k["topical"]["recall@5"] - 0.02
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
