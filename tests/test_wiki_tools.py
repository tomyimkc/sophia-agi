#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for tools/wiki_sync.py and tools/wiki_validate.py (offline)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from okf import frontmatter  # noqa: E402
from tools import wiki_sync, wiki_validate  # noqa: E402


def test_build_pages_carry_provenance() -> None:
    pages = {p["meta"]["id"]: p for p in wiki_sync.build_pages()}
    assert "dao_de_jing" in pages, "expected an entity page for dao_de_jing"
    meta = pages["dao_de_jing"]["meta"]
    assert meta["pageType"] == "text"
    assert "confucius" in meta["doNotAttributeTo"]
    assert meta["sources"] == ["data/attributions.json#dao_de_jing"]
    # provenance survives a frontmatter round-trip
    doc = frontmatter.serialize(meta, pages["dao_de_jing"]["body"])
    assert frontmatter.parse(doc)[0]["doNotAttributeTo"] == meta["doNotAttributeTo"]


def test_emit_and_check_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        wiki = Path(tmp) / "wiki"
        emitted = wiki_sync.emit(wiki)
        assert emitted["written"] > 0
        assert wiki_sync.check(wiki)["ok"] is True


def test_check_detects_drift() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        wiki = Path(tmp) / "wiki"
        wiki_sync.emit(wiki)
        target = wiki / "text" / "dao_de_jing.md"
        meta, body = frontmatter.parse(target.read_text(encoding="utf-8"))
        meta["attributedAuthor"] = "confucius"  # tamper provenance -> a lineage merge
        target.write_text(frontmatter.serialize(meta, body), encoding="utf-8")
        result = wiki_sync.check(wiki)
        assert result["ok"] is False
        assert any(d["key"] == "attributedAuthor" for d in result["drift"])


def test_wiki_validate_repo_clean() -> None:
    # the committed wiki/ + disputes must validate (CI guard)
    wiki_sync.emit()
    result = wiki_validate.run_validation()
    assert result["ok"] is True, result["errors"]


def test_retrieval_carries_provenance() -> None:
    # PR4: a retrieved wiki page carries provenance and surfaces doNotAttributeTo
    from agent import retrieval

    hits = retrieval._retrieve_keyword("Dao De Jing Laozi tradition daoist", top_k=15)
    wiki = [h for h in hits if h.page_id == "dao_de_jing"]
    assert wiki, "expected the dao_de_jing wiki page to be retrievable"
    chunk = wiki[0]
    assert chunk.author_confidence == "legendary"
    assert "confucius" in chunk.do_not_attribute_to
    ctx = retrieval.format_context(hits)
    assert "do NOT attribute" in ctx and "confidence=" in ctx


def main() -> int:
    test_build_pages_carry_provenance()
    test_emit_and_check_roundtrip()
    test_check_detects_drift()
    test_wiki_validate_repo_clean()
    print("test_wiki_tools: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
