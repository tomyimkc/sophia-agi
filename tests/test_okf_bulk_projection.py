#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for okf bulk graph and boundary projection."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from okf import frontmatter  # noqa: E402
from okf.bulk_graph import BulkGraph  # noqa: E402
from okf.graph import build as build_graph  # noqa: E402
from okf.page import Page  # noqa: E402
from okf.projection import project_to_boundary  # noqa: E402


def _page(page_id: str, meta: dict, body: str = "body") -> Page:
    return Page(path=f"wiki/{page_id}.md", meta=meta, body=body)


def test_bulk_combined_graph_adds_hypothesis_edges() -> None:
    boundary = build_graph([
        _page("dao_de_jing", {"id": "dao_de_jing", "pageType": "text", "tradition": "daoist"}),
    ])
    bulk = BulkGraph(boundary=boundary)
    bulk.add_node("hyp", meta={"id": "hyp", "pageType": "concept", "tradition": "daoist"}, body="hypothesis text")
    bulk.add_hypothesis("hyp", "derivesFrom", "dao_de_jing")
    combined = bulk.combined_graph()
    assert "dao_de_jing" in combined.nodes["hyp"]["meta"].get("derivesFrom", [])


def test_projection_promotes_clean_abstains_lineage_trap() -> None:
    boundary = build_graph([
        _page("dao_de_jing", {"id": "dao_de_jing", "pageType": "text", "tradition": "daoist"}),
    ])
    bulk = BulkGraph(boundary=boundary)
    bulk.add_node(
        "clean",
        meta={"id": "clean", "pageType": "concept", "tradition": "daoist", "authorConfidence": "disputed"},
        body=(
            "Tentative reading without forbidden attribution. "
            "Sophia is an AGI-candidate verifier-gated epistemic framework; "
            "this bulk note is candidate infrastructure only. 中文摘要。"
        ),
    )
    bulk.add_node(
        "trap",
        meta={
            "id": "trap",
            "pageType": "text",
            "attributedAuthor": "confucius",
            "doNotAttributeTo": ["confucius"],
        },
        body="Confucius wrote the Dao De Jing.",
    )
    result = project_to_boundary(bulk, skip_provenance=True, skip_conscience=True)
    promoted_ids = {p.node_id for p in result.promoted}
    abstained_ids = {a["nodeId"] for a in result.abstained}
    assert "clean" in promoted_ids
    assert "trap" in abstained_ids


def test_projection_conscience_abstains_without_boundary_wording() -> None:
    boundary = build_graph([
        _page("dao_de_jing", {"id": "dao_de_jing", "pageType": "text", "tradition": "daoist"}),
    ])
    bulk = BulkGraph(boundary=boundary)
    bulk.add_node(
        "held_claim",
        meta={"id": "held_claim", "pageType": "concept", "tradition": "daoist"},
        body="US inflation increased in 2021.",
    )
    result = project_to_boundary(bulk, skip_provenance=True, skip_conscience=False)
    abstained_ids = {a["nodeId"] for a in result.abstained}
    assert "held_claim" in abstained_ids


def test_shadow_lattice_report_ok() -> None:
    from tools.run_shadow_lattice import build_demo_report

    report = build_demo_report(skip_provenance=True, skip_conscience=True)
    assert report["candidateOnly"] is True
    assert report["ok"] is True
