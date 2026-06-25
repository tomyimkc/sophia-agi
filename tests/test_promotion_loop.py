#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for okf promotion loop (projection → pending queue → approve → commit)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import wiki_store  # noqa: E402
from okf.bulk_graph import BulkGraph  # noqa: E402
from okf.graph import build as build_graph  # noqa: E402
from okf.page import Page  # noqa: E402
from okf.projection import project_to_boundary  # noqa: E402
from okf.promotion_loop import (  # noqa: E402
    approve_projection_candidate,
    commit_approved_candidate,
    submit_projection_candidates,
)


def _page(page_id: str, meta: dict, body: str = "body") -> Page:
    return Page(path=f"wiki/{page_id}.md", meta=meta, body=body)


def _candidate_bulk() -> tuple[BulkGraph, str]:
    boundary = build_graph([
        _page("dao_de_jing", {"id": "dao_de_jing", "pageType": "text", "tradition": "daoist"}),
    ])
    bulk = BulkGraph(boundary=boundary)
    node_id = "clean"
    bulk.add_node(
        node_id,
        meta={"id": node_id, "pageType": "concept", "tradition": "daoist"},
        body=(
            "Candidate bulk note only. "
            "Sophia is an AGI-candidate verifier-gated epistemic framework; "
            "this bulk note is candidate infrastructure only."
        ),
    )
    return bulk, node_id


def test_submit_projection_candidates_writes_pending() -> None:
    bulk, node_id = _candidate_bulk()
    projection = project_to_boundary(bulk, skip_provenance=True, skip_conscience=True)

    with tempfile.TemporaryDirectory() as tmp:
        pending = Path(tmp) / "pending_projection_candidates.jsonl"
        result = submit_projection_candidates(projection, path=pending)
        assert result["submitted"] == 1
        assert pending.exists()
        line = pending.read_text(encoding="utf-8").strip()
        assert '"promoted": false' in line
        assert f'"nodeId": "{node_id}"' in line


def test_default_deny_blocks_unapproved_commit() -> None:
    bulk, node_id = _candidate_bulk()
    projection = project_to_boundary(bulk, skip_provenance=True, skip_conscience=True)

    with tempfile.TemporaryDirectory() as tmp:
        pending = Path(tmp) / "pending_projection_candidates.jsonl"
        submit_projection_candidates(projection, path=pending)
        denied = commit_approved_candidate(node_id, path=pending, tier="draft")
        assert denied.get("ok") is False
        assert denied.get("defaultDeny") is True


def test_approved_commit_is_idempotent() -> None:
    bulk, node_id = _candidate_bulk()
    projection = project_to_boundary(bulk, skip_provenance=True, skip_conscience=True)

    orig_memory_dir = wiki_store.MEMORY_DIR
    orig_draft_dir = wiki_store.DRAFT_DIR
    with tempfile.TemporaryDirectory() as tmp:
        wiki_tmp = Path(tmp) / "wiki"
        wiki_tmp.mkdir()
        wiki_store.MEMORY_DIR = wiki_tmp
        wiki_store.DRAFT_DIR = wiki_tmp.parent / "drafts"
        try:
            pending = Path(tmp) / "pending_projection_candidates.jsonl"
            submit_projection_candidates(projection, path=pending)
            approve_projection_candidate(node_id, path=pending, reviewer="test")
            first = commit_approved_candidate(node_id, path=pending, tier="draft")
            second = commit_approved_candidate(node_id, path=pending, tier="draft")
            assert first.get("ok") is True
            assert second.get("idempotent") is True
        finally:
            # Restore globals so this test can't leak into later tests under the
            # full pytest sweep (test-order-dependent flakiness otherwise).
            wiki_store.MEMORY_DIR = orig_memory_dir
            wiki_store.DRAFT_DIR = orig_draft_dir
