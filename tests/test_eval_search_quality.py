# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Regression guard for the search-quality benchmark (graded nDCG + badcase taxonomy).

Deterministic (local hashing embedder, exact-match scorer — no API key, no LLM judge), so
the measured ordering is a stable invariant. Honest ordering (measured): on these short
attribution probes the lexical KEYWORD backend leads the offline local-hash vector on both
recall and graded nDCG (the hash embedder is a weak semantic proxy — README notes Gemini is
the higher-quality backend), and the current hybrid fusion underperforms both (a known,
candidate-only weakness, not asserted as good). Offline; runs in the numpy pytest job.
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
    # NOTE: hybrid fusion currently underperforms BOTH components on these probes — a known
    # candidate-only weakness. We deliberately do NOT assert hybrid >= keyword (it is false).


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
