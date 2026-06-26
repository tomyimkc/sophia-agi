# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for grounded search — the verifiable-perception wiring over the AI-search pipeline.

Deterministic & offline (committed local embedder + OKF provenance graph; no model call).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ai_search import SearchResult  # noqa: E402
from agent.query_understanding import analyze  # noqa: E402
from agent import grounded_search as gs  # noqa: E402


def test_well_sourced_query_answers() -> None:
    r = gs.grounded_search("Tell me about the atomic bomb")
    assert r.grounded
    assert r.action == "answer"
    assert r.confidence is not None and r.confidence >= 0.7
    assert r.target == "atomic_bomb"


def test_weak_source_attribution_is_downgraded() -> None:
    # Disputed/legendary authorship → not served as a clean answer (hedge or abstain).
    r = gs.grounded_search("Who wrote the Dao De Jing?")
    assert r.grounded
    assert r.action in {"hedge", "abstain"}
    assert r.confidence is not None and r.confidence < 0.7
    # Source discipline travels with the result.
    assert "confucius" in (r.belief or {}).get("doNotAttributeTo", [])


def test_no_results_abstains() -> None:
    empty = SearchResult(query=analyze("zzz"), chunks=[])
    r = gs.grounded_search("zzz", search_result=empty)
    assert r.action == "abstain"
    assert r.served == []
    assert r.policy == "grounded_search_abstain"


def test_ungrounded_query_is_not_served_as_answer() -> None:
    r = gs.grounded_search("qwerty nonsense zzxx plooop")
    assert r.grounded is False
    # Fail-closed: nonsense is never served as a clean answer. Whether it hedges (some
    # low-score chunks retrieved) or abstains (no chunks) depends on the active retrieval
    # backend, but it must never be "answer".
    assert r.action in {"hedge", "abstain"}
    assert r.policy in {"grounded_search_ungrounded", "grounded_search_abstain"}


def test_confidence_laundered_belief_forces_hedge() -> None:
    # Unit-test the source-discipline reflex directly: a clean high-confidence pass that is
    # confidence-laundered must be downgraded to a hedge.
    laundered = {"confidenceLaundered": True}
    action, grounded, policy, _ = gs._decide_serve(
        chunks=[object()], target="x", confidence=0.95, belief_view=laundered, thresholds=None
    )
    assert action == "hedge" and grounded is True and policy == "grounded_search_hedge"


def test_thresholds_override_changes_action() -> None:
    # Force everything to abstain by demanding hi=lo=1.0 confidence.
    r = gs.grounded_search("Tell me about the atomic bomb", thresholds={"hi": 1.0, "lo": 1.0})
    assert r.action == "abstain"


def test_gap_logging_feeds_worklist(tmp_path) -> None:
    from agent.knowledge_gap_log import gap_worklist, load_gaps

    ledger = tmp_path / "gaps.jsonl"
    # A hedged (weak-source) query is a knowledge gap and must be logged.
    gs.grounded_search("Who wrote the Dao De Jing?", gap_log_path=ledger)
    gaps = load_gaps(ledger)
    assert gaps and gaps[0]["by"] == "grounded_search"
    assert gaps[0]["policy"].startswith("grounded_search_")
    work = gap_worklist(gaps)
    assert work["totalGaps"] >= 1


def test_clean_answer_is_not_logged_as_gap(tmp_path) -> None:
    from agent.knowledge_gap_log import load_gaps

    ledger = tmp_path / "gaps.jsonl"
    gs.grounded_search("Tell me about the atomic bomb", gap_log_path=ledger)
    # A clean grounded answer is not a gap → nothing written.
    assert load_gaps(ledger) == []


def test_to_dict_is_serializable() -> None:
    import json

    r = gs.grounded_search("Tell me about the atomic bomb")
    json.dumps(r.to_dict())  # must not raise
