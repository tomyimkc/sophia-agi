# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Regression guard for the search-quality benchmark (graded nDCG + badcase taxonomy).

Deterministic (local hashing embedder, exact-match scorer — no API key, no LLM judge), so
the measured ordering is a stable invariant. Honest ordering (measured): on these short
attribution probes the lexical KEYWORD backend leads the offline local-hash vector on both
recall and graded nDCG (the hash embedder is a weak semantic proxy — README notes Gemini is
the higher-quality backend). Hybrid fusion carries a **do-no-harm guard** so it no longer
underperforms its dense component (recall@k(hybrid) >= recall@k(vector)); it still trails
keyword on this query type, which we do not hide. Offline; runs in the numpy pytest job.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_search_quality import BACKENDS, _dcg, _gain, run  # noqa: E402


def test_gain_grades() -> None:
    assert _gain("dao_de_jing", "data/x.json", "dao_de_jing") == 3.0  # exact record
    assert _gain("Some essay about dao_de_jing", "p", "dao_de_jing") == 1.0  # topical
    assert _gain("unrelated", "p", "dao_de_jing") == 0.0


def test_dcg_rewards_earlier_gains() -> None:
    assert _dcg([3.0, 0.0]) > _dcg([0.0, 3.0])


def test_metrics_are_bounded_probabilities() -> None:
    report = run(limit=20)
    for backend in BACKENDS:
        m = report["metrics"][backend]
        for v in m.values():
            assert 0.0 <= v <= 1.0


def test_keyword_leads_lexical_attribution_probes() -> None:
    report = run()
    k = report["metrics"]["keyword"]
    v = report["metrics"]["vector"]
    # Measured truth with the OFFLINE deterministic backends: lexical KEYWORD leads the
    # dense local-hash vector on both recall and graded nDCG (the hash embedder is a weak
    # semantic proxy — a learned/Gemini backend would change this). This replaces the
    # earlier (incorrect, never-true) "vector beats keyword / hybrid beats keyword" claim.
    assert k["recall@5"] > v["recall@5"]
    assert k["ndcg@5"] > v["ndcg@5"]
    # Keyword is the current best backend by graded nDCG (and non-trivial); vector is not broken.
    assert all(k["ndcg@5"] >= report["metrics"][b]["ndcg@5"] for b in BACKENDS)
    assert v["recall@5"] >= 0.4
    # Hybrid trails keyword on this query type (sparse view is uninformative here); we do NOT
    # assert hybrid >= keyword (false). But the do-no-harm guard means hybrid never falls
    # below its dense component — asserted here on the SAME report (no extra eval cost) and
    # unit-tested structurally in tests/test_hybrid_retrieval.py::test_do_no_harm_guard_*.
    h = report["metrics"]["hybrid"]
    assert h["recall@5"] >= v["recall@5"]
    assert h["ndcg@5"] >= v["ndcg@5"] - 1e-9


def test_badcase_taxonomy_is_well_formed() -> None:
    report = run(limit=30)
    tax = report["badcaseTaxonomy"]
    assert set(tax["counts"]) == {"lexical_gap", "semantic_gap", "tied_burial", "absent_from_pool"}
    assert all(isinstance(c, int) and c >= 0 for c in tax["counts"].values())
    assert all(len(v) <= 5 for v in tax["examples"].values())


def test_report_marked_candidate_not_validated() -> None:
    report = run(limit=6)
    assert report["candidateOnly"] is True
    assert report["validated"] is False
    assert report["bestBackend"] in BACKENDS
