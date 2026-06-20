#!/usr/bin/env python3
"""Tests for the OKF wiki MCP tools (read surface + audited write)."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import wiki_store  # noqa: E402
from sophia_mcp import audit, tools_impl as ti  # noqa: E402


def test_wiki_read_and_search() -> None:
    page = ti.wiki_read("dao_de_jing")
    assert page["frontmatter"]["doNotAttributeTo"] == ["confucius", "socrates", "plato"]
    assert ti.wiki_read("nope__missing")["error"]
    ids = [r["id"] for r in ti.wiki_search("dao de jing laozi tradition", top_k=6)["results"]]
    assert "dao_de_jing" in ids


def test_wiki_contradictions_and_validate() -> None:
    assert ti.wiki_contradictions()["selfMerges"] == []
    assert ti.wiki_validate_tool()["ok"] is True


def test_wiki_upsert_audited_and_gated() -> None:
    originals = (wiki_store.CANONICAL_DIR, wiki_store.MEMORY_DIR, wiki_store.DRAFT_DIR)
    with tempfile.TemporaryDirectory() as t:
        wiki_store.CANONICAL_DIR = Path(t) / "wiki"
        wiki_store.MEMORY_DIR = Path(t) / "memory"
        wiki_store.DRAFT_DIR = Path(t) / "wiki" / "drafts"
        (Path(t) / "wiki").mkdir(parents=True, exist_ok=True)
        os.environ.pop(audit.APPROVE_ENV, None)
        try:
            # denied without approval
            assert ti.wiki_upsert("term_a", '{"pageType": "concept"}', "# A\n\nclean.").get("denied") is True
            # approved -> clean page lands
            os.environ[audit.APPROVE_ENV] = "1"
            ok = ti.wiki_upsert("term_a", '{"pageType": "concept"}', "# A\n\nclean concept.")
            assert ok["ok"] is True, ok
            # approved but a lineage merge is STILL rejected by the gate
            bad = ti.wiki_upsert("term_b", '{"pageType": "text"}', "# B\n\nSocrates authored the Republic.")
            assert bad["ok"] is False and any("forbidden attribution" in r for r in bad["reasons"])
        finally:
            os.environ.pop(audit.APPROVE_ENV, None)
            wiki_store.CANONICAL_DIR, wiki_store.MEMORY_DIR, wiki_store.DRAFT_DIR = originals


def main() -> int:
    test_wiki_read_and_search()
    test_wiki_contradictions_and_validate()
    test_wiki_upsert_audited_and_gated()
    print("test_wiki_mcp: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
