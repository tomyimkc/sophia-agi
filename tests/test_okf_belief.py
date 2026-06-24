#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for okf.belief — per-entity belief lookup exposing effectiveConfidenceRank.

belief() resolves an entity to its node and reports the min-over-derivesFrom-chain
confidence, so a confident claim resting on weak provenance is flagged (laundering)
rather than taken at face value. Dependency-free, offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import okf  # noqa: E402
from okf.page import Page  # noqa: E402


def _graph():
    pages = [
        Page(path=Path("primary.md"), meta={
            "id": "primary_source", "pageType": "concept",
            "authorConfidence": "consensus", "aliases": ["primary alias"],
        }),
        Page(path=Path("legend.md"), meta={
            "id": "legend", "pageType": "concept", "authorConfidence": "legendary",
        }),
        Page(path=Path("secondary.md"), meta={
            "id": "secondary", "pageType": "concept",
            "authorConfidence": "consensus", "derivesFrom": ["legend"],
            "attributedAuthor": "Someone", "doNotAttributeTo": ["Imposter"],
        }),
    ]
    return okf.build_graph(pages)


def test_effective_confidence_is_min_over_chain() -> None:
    b = okf.belief(_graph(), "secondary")
    assert b["found"] is True
    assert b["confidenceRank"] == 4          # its own declared "consensus"
    assert b["effectiveConfidenceRank"] == 1  # capped by "legendary" dependency
    assert b["confidenceLaundered"] is True


def test_root_node_not_laundered() -> None:
    b = okf.belief(_graph(), "primary_source")
    assert b["effectiveConfidenceRank"] == 4
    assert b["confidenceLaundered"] is False


def test_resolves_via_alias() -> None:
    b = okf.belief(_graph(), "primary alias")
    assert b["found"] is True and b["id"] == "primary_source"


def test_surfaces_provenance_fields() -> None:
    b = okf.belief(_graph(), "secondary")
    assert b["attributedAuthor"] == "Someone"
    assert b["doNotAttributeTo"] == ["Imposter"]


def test_missing_entity() -> None:
    b = okf.belief(_graph(), "does-not-exist")
    assert b["found"] is False and b["id"] is None


def main() -> int:
    test_effective_confidence_is_min_over_chain()
    test_root_node_not_laundered()
    test_resolves_via_alias()
    test_surfaces_provenance_fields()
    test_missing_entity()
    print("test_okf_belief: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
