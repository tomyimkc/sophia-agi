#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent/failure_memory.py — ACCURATE + TRACEABLE + RELIABLE invariants."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.failure_memory import (  # noqa: E402
    FailureMemoryStore,
    build_contradiction_edge,
    check_store_decontamination,
    has_grounded_correction,
    overlaps_heldout,
    stable_id,
    stable_key,
)


def _store(tmp: str) -> FailureMemoryStore:
    return FailureMemoryStore(path=Path(tmp) / "nodes.jsonl")


def _ingest_kwargs(**overrides):
    base = {
        "question": "Who wrote the Dao De Jing?",
        "wrong_claim": "Confucius wrote the Dao De Jing.",
        "correction_claim": "The Dao De Jing is attributed to Laozi (confidence: legendary).",
        "correction_citation": "data/attributions.json#dao_de_jing",
        "correction_source": "provenance gate eval",
        "verifier_name": "eval_label",
        "verifier_verdict": "label:false",
        "run_id": "test-run-001",
        "created_at": "2026-06-25T10:00:00Z",
    }
    base.update(overrides)
    return base


def test_valid_error_stored_with_correction() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(tmp)
        res = store.ingest(**_ingest_kwargs())
        assert res.ok and not res.rejected
        assert res.node_id == stable_id(stable_key(
            "Who wrote the Dao De Jing?",
            "Confucius wrote the Dao De Jing.",
        ))
        nodes = store.list_nodes()
        assert len(nodes) == 1
        node = nodes[0]
        assert has_grounded_correction(node)
        assert node["correction"]["citation"] == "data/attributions.json#dao_de_jing"
        assert node["candidateOnly"] is True
        assert node["level3Evidence"] is False


def test_error_without_grounded_correction_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(tmp)
        res = store.ingest(**_ingest_kwargs(correction_citation=""))
        assert not res.ok and res.rejected
        assert store.list_nodes() == []


def test_reingest_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(tmp)
        r1 = store.ingest(**_ingest_kwargs())
        r2 = store.ingest(**_ingest_kwargs())
        assert r1.ok and r2.ok and r2.deduped
        assert len(store.list_nodes()) == 1
        assert len(store._read_all()) == 1


def test_contradiction_edge_recorded() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(tmp)
        res = store.ingest(**_ingest_kwargs(contradicts_belief_id="dao_de_jing"))
        assert res.ok
        edge = store.list_nodes()[0]["contradictionEdge"]
        assert edge["beliefId"] == "dao_de_jing"
        assert edge["kind"] == "contradicts"
        assert "NOT a wiki belief" in edge["note"]


def test_append_only_versioning() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(tmp)
        store.ingest(**_ingest_kwargs())
        store.ingest(
            **_ingest_kwargs(
                correction_claim="Revised: Laozi is the traditional attribution.",
                force_new_version=True,
            )
        )
        versions = store.versions_of(stable_id(stable_key(
            "Who wrote the Dao De Jing?",
            "Confucius wrote the Dao De Jing.",
        )))
        assert len(versions) == 2
        assert versions[0]["version"] == "v1"
        assert versions[1]["version"] == "v2"
        assert versions[0]["correction"]["claim"] != versions[1]["correction"]["claim"]


def test_heldout_overlap_rejected() -> None:
    if not overlaps_heldout("Who wrote the Dao De Jing?"):
        return  # skip if this prompt is not in sealed set on this checkout
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(tmp)
        res = store.ingest(**_ingest_kwargs())
        assert res.rejected


def test_decontamination_check_clean() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(tmp)
        store.ingest(**_ingest_kwargs(question="ERROR-MEMORY-TEST-001: Who authored the Zhuangzi commentary?"))
        audit = check_store_decontamination(store)
        assert audit["clean"] is True
        assert len(audit["heldoutPromptHash"]) == 16


def test_build_contradiction_edge_fields() -> None:
    edge = build_contradiction_edge(
        "analects",
        wrong_claim="Laozi wrote the Analects.",
        created_at="2026-06-25T10:00:00Z",
        run_id="run-x",
    )
    assert edge["beliefId"] == "analects"
    assert edge["wrongClaim"] == "Laozi wrote the Analects."
