#!/usr/bin/env python3
"""Tests for the sophia_counterfactual + sophia_retract MCP tools. All offline.

The corpus loader is stubbed so the queries are deterministic and never touch the
live wiki. Mirrors tests/test_mcp_belief.py.
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
    Page(path=Path("primary.md"), meta={"id": "primary", "pageType": "concept", "authorConfidence": "consensus"}),
    Page(path=Path("solo.md"), meta={
        "id": "derived_solo", "pageType": "concept", "authorConfidence": "consensus", "derivesFrom": ["primary"],
    }),
]


def _patch():
    original = wiki_store.load_all_pages
    wiki_store.load_all_pages = lambda: PAGES
    return original


def test_counterfactual_reports_support_loss() -> None:
    original = _patch()
    try:
        out = tools_impl.counterfactual("primary", query="derived_solo")
    finally:
        wiki_store.load_all_pages = original
    assert out["found"] is True
    assert "derived_solo" in out["supportLost"]
    assert out["query"]["supportLost"] is True


def test_counterfactual_unknown_source() -> None:
    original = _patch()
    try:
        out = tools_impl.counterfactual("nope")
    finally:
        wiki_store.load_all_pages = original
    assert out["found"] is False


def test_retract_returns_audit_and_downstream() -> None:
    original = _patch()
    try:
        out = tools_impl.retract("primary", reason="forged", by="curator")
    finally:
        wiki_store.load_all_pages = original
    assert out["found"] is True
    assert "derived_solo" in out["downstream"]
    assert out["impact"]["found"] is True
    assert out["reason"] == "forged" and out["by"] == "curator"


def main() -> int:
    test_counterfactual_reports_support_loss()
    test_counterfactual_unknown_source()
    test_retract_returns_audit_and_downstream()
    print("test_mcp_counterfactual: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
