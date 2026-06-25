#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for okf promotion loop (projection → pending queue)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from okf.bulk_graph import BulkGraph  # noqa: E402
from okf.graph import build as build_graph  # noqa: E402
from okf.page import Page  # noqa: E402
from okf.projection import project_to_boundary  # noqa: E402
from okf.promotion_loop import submit_projection_candidates  # noqa: E402


def _page(page_id: str, meta: dict, body: str = "body") -> Page:
    return Page(path=f"wiki/{page_id}.md", meta=meta, body=body)


def test_submit_projection_candidates_writes_pending() -> None:
    boundary = build_graph([
        _page("dao_de_jing", {"id": "dao_de_jing", "pageType": "text", "tradition": "daoist"}),
    ])
    bulk = BulkGraph(boundary=boundary)
    bulk.add_node(
        "clean",
        meta={"id": "clean", "pageType": "concept", "tradition": "daoist"},
        body=(
            "Candidate bulk note only. "
            "Sophia is an AGI-candidate verifier-gated epistemic framework; "
            "this bulk note is candidate infrastructure only."
        ),
    )
    projection = project_to_boundary(bulk, skip_provenance=True, skip_conscience=True)

    with tempfile.TemporaryDirectory() as tmp:
        pending = Path(tmp) / "pending_projection_candidates.jsonl"
        result = submit_projection_candidates(projection, path=pending)
        assert result["submitted"] == 1
        assert pending.exists()
        line = pending.read_text(encoding="utf-8").strip()
        assert '"promoted": false' in line
        assert '"nodeId": "clean"' in line
