# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the gap → draft-stub self-correction loop (fail-closed, no fabrication)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import gap_ingest as gi  # noqa: E402


def test_candidate_id_strips_interrogative_frame() -> None:
    assert gi.candidate_id("Who wrote the Dao De Jing?") == "dao_de_jing"
    assert gi.candidate_id("Tell me about quantum entanglement") == "quantum_entanglement"
    assert gi.candidate_id("???") == "untitled_gap"


def test_draft_stub_passes_provenance_gate_and_asserts_nothing() -> None:
    from agent.wiki_store import gate

    meta, body = gi.draft_stub("quantum_entanglement", queries=["What is it?"], gap_hits=3)
    ok, reasons = gate(meta, body)
    assert ok, reasons
    # Fail-closed provenance skeleton: weakest tier, flagged, no attribution, no sources.
    assert meta["authorConfidence"] == "none_extant"
    assert meta["needsReview"] is True
    assert "attributedAuthor" not in meta
    assert meta["sources"] == []
    assert "No claims are asserted" in body


def test_plan_separates_missing_from_existing() -> None:
    gaps = [
        {"query": "What is quantum entanglement?", "target": None, "policy": "grounded_search_ungrounded"},
        {"query": "Explain quantum entanglement", "target": None, "policy": "grounded_search_ungrounded"},
        {"query": "Who wrote the Dao De Jing?", "target": "dao_de_jing", "policy": "grounded_search_hedge"},
    ]
    plan = gi.plan_ingestion(gaps, existing_ids={"dao_de_jing"})
    assert [(i.page_id, i.hits) for i in plan.create] == [("quantum_entanglement", 2)]
    assert [(e.target, e.hits) for e in plan.enrich] == [("dao_de_jing", 1)]


def test_plan_is_idempotent_skips_existing_ids() -> None:
    gaps = [{"query": "Tell me about the atomic bomb", "target": None, "policy": "grounded_search_ungrounded"}]
    # If a page with that slug already exists, it is not re-created.
    plan = gi.plan_ingestion(gaps, existing_ids={"atomic_bomb"})
    assert plan.create == []
    assert [e.target for e in plan.enrich] == ["atomic_bomb"]


def test_min_hits_filters_one_off_gaps() -> None:
    gaps = [{"query": "Define epiphenomenalism", "target": None, "policy": "grounded_search_ungrounded"}]
    assert gi.plan_ingestion(gaps, existing_ids=set(), min_hits=2).create == []
    assert gi.plan_ingestion(gaps, existing_ids=set(), min_hits=1).create


def test_materialize_dry_run_writes_nothing(tmp_path, monkeypatch) -> None:
    import agent.wiki_store as ws

    monkeypatch.setattr(ws, "DRAFT_DIR", tmp_path / "drafts")
    plan = gi.IngestPlan(create=[gi.IngestItem(page_id="topic_x", queries=["q"], hits=1)])
    report = gi.materialize(plan, write=False)
    assert report["wrote"] is False
    assert report["wouldCreate"] and report["created"] == []
    assert not (tmp_path / "drafts").exists()  # truly nothing written


def test_materialize_write_creates_gated_draft_and_closes_loop(tmp_path, monkeypatch) -> None:
    import okf

    import agent.wiki_store as ws
    from agent.config import WIKI_DIR
    from agent.grounded_search import grounded_search

    draft_dir = tmp_path / "drafts"
    monkeypatch.setattr(ws, "DRAFT_DIR", draft_dir)

    plan = gi.IngestPlan(create=[
        gi.IngestItem(page_id="quantum_entanglement",
                      queries=["What is quantum entanglement?"], hits=2)])
    report = gi.materialize(plan, write=True)
    assert report["created"] and report["created"][0]["id"] == "quantum_entanglement"
    assert (draft_dir / "quantum_entanglement.md").exists()

    # Loop closure: the query that was ungrounded now routes to the stub and ABSTAINS
    # (none_extant → confidence 0.30) — a known unknown, fail-closed.
    pages = list(okf.load_pages(WIKI_DIR, draft_dir))
    r = grounded_search("What is quantum entanglement?", pages=pages)
    assert r.target == "quantum_entanglement"
    assert r.action == "abstain"


def test_close_gap_loop_tool_run(tmp_path, monkeypatch) -> None:
    import agent.wiki_store as ws
    from tools.close_gap_loop import run

    monkeypatch.setattr(ws, "DRAFT_DIR", tmp_path / "drafts")
    ledger = tmp_path / "gaps.jsonl"
    ledger.write_text(
        '{"query": "What is quantum entanglement?", "target": null, "policy": "grounded_search_ungrounded"}\n',
        encoding="utf-8",
    )
    dry = run(ledger, write=False)
    assert dry["gapsRead"] == 1 and dry["wrote"] is False and dry["wouldCreate"]
    wrote = run(ledger, write=True)
    assert wrote["created"]
