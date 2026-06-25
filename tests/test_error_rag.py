#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent/error_rag.py — precision gates + guard-rail framing."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.error_rag import (  # noqa: E402
    DEFAULT_GATES,
    PrecisionGates,
    build_guard_context,
    inject_error_rag,
    retrieve_and_build,
    validate_guard_framing,
)
from agent.failure_memory import FailureMemoryStore  # noqa: E402


def _populate_store(path: Path) -> FailureMemoryStore:
    store = FailureMemoryStore(path=path / "nodes.jsonl")
    store.ingest(
        question="ERROR-MEMORY-TEST-001: Who wrote the Dao De Jing?",
        wrong_claim="Confucius wrote the Dao De Jing.",
        correction_claim="Attributed to Laozi (legendary).",
        correction_citation="data/attributions.json#dao_de_jing",
        correction_source="eval",
        verifier_name="eval_label",
        verifier_verdict="label:false",
        run_id="rag-test",
        created_at="2026-06-25T10:00:00Z",
        work_id="dao_de_jing",
        forbidden_author="confucius",
    )
    store.ingest(
        question="ERROR-MEMORY-TEST-002: Who authored the I Ching?",
        wrong_claim="Buddha authored the I Ching.",
        correction_claim="Traditional attribution is Fu Xi / King Wen / Duke of Zhou.",
        correction_citation="data/attributions.json#i_ching",
        correction_source="eval",
        verifier_name="gate",
        verifier_verdict="provenance_violation",
        run_id="rag-test",
        created_at="2026-06-25T10:01:00Z",
        work_id="i_ching",
        forbidden_author="confucius",
    )
    return store


def test_nearby_different_question_injects_nothing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _populate_store(Path(tmp))
        # Zhuangzi question with a different wrong answer — not a would-repeat of DDJ node.
        result = inject_error_rag(
            "ERROR-MEMORY-TEST-099: Who wrote the Zhuangzi?",
            store=store,
            current_answer="Laozi wrote the Zhuangzi.",
            gates=DEFAULT_GATES,
            query_work_id="zhuangzi",
            query_forbidden_author="laozi",
        )
        assert not result.injected


def test_genuine_would_repeat_injects_guard() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _populate_store(Path(tmp))
        wrong = "Confucius wrote the Dao De Jing."
        result = inject_error_rag(
            "ERROR-MEMORY-TEST-100: Who composed the Dao De Jing?",
            store=store,
            current_answer=wrong,
            gates=DEFAULT_GATES,
            query_work_id="dao_de_jing",
            query_forbidden_author="confucius",
        )
        assert result.injected
        assert validate_guard_framing(result.context)
        assert "KNOWN PAST ERROR" in result.context
        assert "[data/attributions.json#dao_de_jing]" in result.context


def test_deterministic_retrieval_with_gates() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _populate_store(Path(tmp))
        wrong = "Confucius wrote the Dao De Jing."
        q = "ERROR-MEMORY-TEST-101: Who composed the Dao De Jing?"
        kwargs = dict(
            current_answer=wrong,
            gates=DEFAULT_GATES,
            query_work_id="dao_de_jing",
            query_forbidden_author="confucius",
        )
        r1 = retrieve_and_build(q, store, **kwargs)
        r2 = retrieve_and_build(q, store, **kwargs)
        assert r1.injected == r2.injected
        assert r1.node_ids == r2.node_ids
        assert r1.context == r2.context


def test_missing_correction_skipped() -> None:
    bad_node = {
        "id": "bad1",
        "wrongClaim": "x",
        "verifier": {"name": "gate", "verdict": "wrong"},
        "sourceEvent": {"question": "q"},
        "correction": {},
    }
    result = build_guard_context([bad_node])
    assert not result.injected
    assert "bad1" in result.skipped


def test_disabled_injects_nothing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _populate_store(Path(tmp))
        result = inject_error_rag("Dao De Jing", store=store, enabled=False)
        assert not result.injected


def test_fail_closed_empty_store() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = FailureMemoryStore(path=Path(tmp) / "empty.jsonl")
        result = inject_error_rag("any query", store=store, current_answer="wrong")
        assert not result.injected


def test_would_repeat_gate_blocks_different_answer() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _populate_store(Path(tmp))
        result = inject_error_rag(
            "ERROR-MEMORY-TEST-102: Who wrote the Dao De Jing?",
            store=store,
            current_answer="Plato wrote the Dao De Jing.",
            gates=PrecisionGates(require_would_repeat=True),
            query_work_id="dao_de_jing",
            query_forbidden_author="plato",
        )
        assert not result.injected
