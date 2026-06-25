#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for okf.consistency_check — syntactic cross-context hole detection."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from okf.consistency_check import (  # noqa: E402
    consistency_report,
    find_epistemic_holes,
    propose_hole_patch,
    write_epistemic_holes,
)
from okf.graph import build  # noqa: E402
from okf.page import Page  # noqa: E402


def _pages_undeclared_hole() -> list[Page]:
    shared = "Shared Work"
    return [
        Page(path=Path("a.md"), meta={
            "id": "work_confucian",
            "pageType": "text",
            "tradition": "confucian",
            "canonicalTitleEn": shared,
            "attributedAuthor": "author_a",
            "authorConfidence": "attributed",
            "sources": ["data/test.json#work_confucian"],
        }),
        Page(path=Path("b.md"), meta={
            "id": "work_daoist",
            "pageType": "text",
            "tradition": "daoist",
            "canonicalTitleEn": shared,
            "attributedAuthor": "author_b",
            "authorConfidence": "attributed",
            "sources": ["data/test.json#work_daoist"],
        }),
    ]


def _pages_declared_contradiction() -> list[Page]:
    pages = _pages_undeclared_hole()
    pages[0].meta["contradicts"] = ["work_daoist"]
    return pages


def test_undeclared_disagreement_emits_one_hole() -> None:
    graph = build(_pages_undeclared_hole())
    holes = find_epistemic_holes(graph)
    assert len(holes) == 1
    h = holes[0]
    assert h["entity"] == "shared_work"
    assert h["contextA"] == "confucian"
    assert h["contextB"] == "daoist"
    assert h["claimA"]["attributedAuthor"] == "author_a"
    assert h["claimB"]["attributedAuthor"] == "author_b"
    assert h["resolved"] is False


def test_declared_contradiction_defers_to_ledger() -> None:
    graph = build(_pages_declared_contradiction())
    holes = find_epistemic_holes(graph)
    assert holes == []


def test_patch_without_source_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        holes_path = Path(tmp) / "holes.jsonl"
        pages = _pages_undeclared_hole()
        report = consistency_report(pages)
        write_epistemic_holes(report["epistemicHoles"], path=holes_path)
        hole_id = report["epistemicHoles"][0]["holeId"]
        result = propose_hole_patch(
            hole_id,
            "Align both traditions by declaring them consistent.",
            sources=[],
            path=holes_path,
            skip_conscience=True,
        )
        assert result["ok"] is False
        assert result["rejected"] is True
        assert result["defaultDeny"] is True
        assert "source" in result["reason"].lower()


def test_deterministic_report() -> None:
    pages = _pages_undeclared_hole()
    r1 = consistency_report(pages)
    r2 = consistency_report(pages)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)


def main() -> int:
    test_undeclared_disagreement_emits_one_hole()
    test_declared_contradiction_defers_to_ledger()
    test_patch_without_source_rejected()
    test_deterministic_report()
    print("test_consistency_check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
