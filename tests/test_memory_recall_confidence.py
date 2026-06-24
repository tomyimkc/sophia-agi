#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for graph-confidence injection into harness._memory_recall. Offline.

Memory recall should surface the belief graph's EFFECTIVE confidence (min over the
derivesFrom chain), so a recalled page that declares high confidence while resting
on a weak source is shown as provenance-capped, not at face value.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import okf  # noqa: E402
from agent import harness  # noqa: E402
from okf.page import Page  # noqa: E402

PAGES = [
    Page(path=Path("legend.md"), meta={"id": "legend", "pageType": "concept", "authorConfidence": "legendary"}),
    Page(path=Path("secondary.md"), meta={
        "id": "secondary", "pageType": "concept", "authorConfidence": "consensus",
        "derivesFrom": ["legend"], "doNotAttributeTo": ["Imposter"],
    }),
    Page(path=Path("solid.md"), meta={"id": "solid", "pageType": "concept", "authorConfidence": "consensus"}),
]
GRAPH = okf.build_graph(PAGES)
BY_ID = {p.id: p for p in PAGES}


def _recall(page_ids):
    pages = [BY_ID[i] for i in page_ids]
    return harness._memory_recall(
        "goal", search_fn=lambda goal, top_k=3: pages, graph=GRAPH,
    )


def test_laundered_page_shows_effective_and_warns() -> None:
    out = _recall(["secondary"])
    assert "[[secondary]]" in out
    assert "1" in out                       # effective rank capped at "legendary" == 1
    low = out.lower()
    assert "effective" in low
    assert "cap" in low or "laundif" in low or "⚠" in out  # a provenance-cap warning
    assert "Imposter" in out                # existing do-not-attribute warning preserved


def test_solid_page_not_flagged() -> None:
    out = _recall(["solid"])
    assert "[[solid]]" in out
    assert "cap" not in out.lower()         # no laundering warning when self-sourced


def test_empty_recall_is_blank() -> None:
    out = harness._memory_recall("goal", search_fn=lambda goal, top_k=3: [], graph=GRAPH)
    assert out == ""


def test_never_raises_on_bad_graph() -> None:
    # a search_fn that explodes must degrade to "" (recall must not break planning)
    def boom(goal, top_k=3):
        raise RuntimeError("boom")

    assert harness._memory_recall("goal", search_fn=boom, graph=GRAPH) == ""


def main() -> int:
    test_laundered_page_shows_effective_and_warns()
    test_solid_page_not_flagged()
    test_empty_recall_is_blank()
    test_never_raises_on_bad_graph()
    print("test_memory_recall_confidence: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
