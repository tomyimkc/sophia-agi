#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent/error_rag.py — deterministic retrieval + guard-rail framing."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.error_rag import (  # noqa: E402
    build_guard_context,
    inject_error_rag,
    retrieve_and_build,
    validate_guard_framing,
)
from agent.failure_memory import FailureMemoryStore  # noqa: E402


def _populate_store(path: Path) -> FailureMemoryStore:
    store = FailureMemoryStore(path=path / "nodes.jsonl")
    store.ingest(
        question="Who wrote the Dao De Jing?",
        wrong_claim="Confucius wrote the Dao De Jing.",
        correction_claim="Attributed to Laozi (legendary).",
        correction_citation="data/attributions.json#dao_de_jing",
        correction_source="eval",
        verifier_name="eval_label",
        verifier_verdict="label:false",
        run_id="rag-test",
        created_at="2026-06-25T10:00:00Z",
    )
    store.ingest(
        question="Who authored the I Ching?",
        wrong_claim="Buddha authored the I Ching.",
        correction_claim="Traditional attribution is Fu Xi / King Wen / Duke of Zhou.",
        correction_citation="data/attributions.json#i_ching",
        correction_source="eval",
        verifier_name="gate",
        verifier_verdict="provenance_violation",
        run_id="rag-test",
        created_at="2026-06-25T10:01:00Z",
    )
    return store


def test_deterministic_retrieval() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _populate_store(Path(tmp))
        q = "Who composed the Dao De Jing?"
        r1 = retrieve_and_build(q, store)
        r2 = retrieve_and_build(q, store)
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


def test_guard_context_framing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _populate_store(Path(tmp))
        result = inject_error_rag("Dao De Jing author attribution", store=store, enabled=True)
        assert result.injected
        assert validate_guard_framing(result.context)
        assert "KNOWN PAST ERROR" in result.context
        assert "this was WRONG" in result.context
        assert "The verified answer is" in result.context
        assert "[data/attributions.json#dao_de_jing]" in result.context
        assert "Do not repeat the error" in result.context


def test_disabled_injects_nothing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _populate_store(Path(tmp))
        result = inject_error_rag("Dao De Jing", store=store, enabled=False)
        assert not result.injected


def test_fail_closed_empty_store() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = FailureMemoryStore(path=Path(tmp) / "empty.jsonl")
        result = inject_error_rag("any query", store=store)
        assert not result.injected
