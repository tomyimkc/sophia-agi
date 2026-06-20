#!/usr/bin/env python3
"""Tests for the sophia_belief MCP tool (sophia_mcp). All offline.

The corpus loader is stubbed so the belief lookup is deterministic and never
touches the live wiki.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import wiki_store  # noqa: E402
from okf.page import Page  # noqa: E402
from sophia_mcp import tools_impl  # noqa: E402

PAGES = [
    Page(path=Path("legend.md"), meta={"id": "legend", "pageType": "concept", "authorConfidence": "legendary"}),
    Page(path=Path("secondary.md"), meta={
        "id": "secondary", "pageType": "concept", "authorConfidence": "consensus", "derivesFrom": ["legend"],
    }),
]


def _patch():
    original = wiki_store.load_all_pages
    wiki_store.load_all_pages = lambda: PAGES
    return original


def test_belief_reports_effective_rank() -> None:
    original = _patch()
    try:
        out = tools_impl.belief("secondary")
    finally:
        wiki_store.load_all_pages = original
    assert out["found"] is True
    assert out["effectiveConfidenceRank"] == 1
    assert out["confidenceLaundered"] is True


def test_belief_missing_entity() -> None:
    original = _patch()
    try:
        out = tools_impl.belief("nope")
    finally:
        wiki_store.load_all_pages = original
    assert out["found"] is False


def main() -> int:
    test_belief_reports_effective_rank()
    test_belief_missing_entity()
    print("test_mcp_belief: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
