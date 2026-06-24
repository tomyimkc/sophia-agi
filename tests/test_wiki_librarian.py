#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent/wiki_store.py and agent/wiki_librarian.py (offline)."""

from __future__ import annotations

import contextlib
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import wiki_librarian, wiki_store  # noqa: E402


@contextlib.contextmanager
def _redirect():
    """Point wiki_store at a temp dir, then RESTORE the originals on exit.

    Restoration matters under a single-process ``pytest tests/`` run: leaving the
    globals pointed at a deleted temp dir poisoned later tests (e.g. test_wiki_mcp
    reading dao_de_jing). No pytest dependency so CI can still run this as a script.
    """
    saved = (wiki_store.CANONICAL_DIR, wiki_store.MEMORY_DIR, wiki_store.DRAFT_DIR)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        wiki_store.CANONICAL_DIR = tmp / "wiki"
        wiki_store.MEMORY_DIR = tmp / "memory"
        wiki_store.DRAFT_DIR = tmp / "wiki" / "drafts"
        (tmp / "wiki").mkdir(parents=True, exist_ok=True)
        try:
            yield tmp
        finally:
            wiki_store.CANONICAL_DIR, wiki_store.MEMORY_DIR, wiki_store.DRAFT_DIR = saved


def test_store_accepts_clean_page() -> None:
    with _redirect():
        res = wiki_store.upsert(
            "test_concept",
            meta={"pageType": "concept", "domain": "psychology"},
            body="# Test Concept\n\nA clean concept with no attribution claims.",
            tier="memory",
        )
        assert res["ok"] is True, res
        assert wiki_store.read_page("test_concept") is not None
        assert res["revisions"] == 1


def test_store_rejects_self_merge() -> None:
    with _redirect():
        res = wiki_store.upsert(
            "bad_page",
            meta={"pageType": "text", "attributedAuthor": "confucius", "authorConfidence": "attributed",
                  "doNotAttributeTo": ["confucius"]},
            body="# Bad\n\nbody",
            tier="draft",
        )
        assert res["ok"] is False and res.get("rejected")
        assert any("lineage-merge" in r for r in res["reasons"])


def test_store_rejects_forbidden_attribution_in_body() -> None:
    with _redirect():
        res = wiki_store.upsert(
            "dao_note",
            meta={"pageType": "text"},
            body="# Note\n\nConfucius wrote the Dao De Jing himself.",
            tier="draft",
        )
        assert res["ok"] is False
        assert any("forbidden attribution" in r for r in res["reasons"])


def test_librarian_build_page() -> None:
    meta, body = wiki_librarian.build_page(
        {"id": "stoic_oikeiosis", "pageType": "concept", "title": "Oikeiosis",
         "tradition": "stoic", "summary": "A Stoic concept of appropriation."},
        "stoic-intro.txt",
    )
    assert meta["id"] == "stoic_oikeiosis"
    assert meta["sources"] == ["raw/stoic-intro.txt"]
    assert "Oikeiosis" in body


def test_librarian_ingest_accept_and_reject() -> None:
    with _redirect():
        ok = wiki_librarian.ingest_proposal(
            {"id": "new_term", "pageType": "concept", "title": "New Term", "summary": "A clean new term."},
            "src1.txt",
        )
        assert ok["ok"] is True, ok

        bad = wiki_librarian.ingest_proposal(
            {"id": "merge_term", "pageType": "text", "title": "Bad",
             "summary": "Socrates authored the Republic."},
            "src2.txt",
        )
        assert bad["ok"] is False and any("forbidden attribution" in r for r in bad["reasons"])


def test_librarian_ingest_text_with_stub_model() -> None:
    with _redirect():
        proposal_json = '{"id": "ren_concept", "pageType": "concept", "title": "Ren", "tradition": "confucian", "summary": "Confucian benevolence."}'
        stub = SimpleNamespace(generate=lambda s, u: SimpleNamespace(
            ok=True, text=f"Here is the page:\n```json\n{proposal_json}\n```", error=None,
            cost_usd=0.0, latency_sec=0.0, tool_calls=[]))
        res = wiki_librarian.ingest_text("Ren is the Confucian virtue of benevolence.", "ren.txt", client=stub)
        assert res["ok"] is True, res
        assert res["id"] == "ren_concept"


def main() -> int:
    test_store_accepts_clean_page()
    test_store_rejects_self_merge()
    test_store_rejects_forbidden_attribution_in_body()
    test_librarian_build_page()
    test_librarian_ingest_accept_and_reject()
    test_librarian_ingest_text_with_stub_model()
    print("test_wiki_librarian: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
