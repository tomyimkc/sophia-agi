# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for canon promotion — human-gated, re-gated draft → memory elevation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import canon_promote as cp  # noqa: E402


def _store(monkeypatch, tmp_path):
    import agent.wiki_store as ws

    monkeypatch.setattr(ws, "CANONICAL_DIR", tmp_path / "wiki")
    monkeypatch.setattr(ws, "MEMORY_DIR", tmp_path / "mem")
    monkeypatch.setattr(ws, "DRAFT_DIR", tmp_path / "wiki" / "drafts")
    (tmp_path / "wiki").mkdir(parents=True, exist_ok=True)
    return ws


def _make_draft(ws, page_id="penicillin_history"):
    ws.upsert(page_id, tier="draft", body="# Penicillin\n\nAn antibiotic discovered by Fleming.",
              meta={"pageType": "concept", "domain": "science",
                    "attributedAuthor": "alexander_fleming", "authorConfidence": "attributed",
                    "provenance": "librarian_fill", "needsReview": True})


def test_pending_reviews_lists_drafts_with_gate_status(monkeypatch, tmp_path) -> None:
    ws = _store(monkeypatch, tmp_path)
    _make_draft(ws)
    pend = cp.pending_reviews()
    assert len(pend) == 1
    r = pend[0]
    assert r["id"] == "penicillin_history" and r["gatePasses"] is True
    assert r["authorConfidence"] == "attributed"


def test_promote_requires_approver(monkeypatch, tmp_path) -> None:
    ws = _store(monkeypatch, tmp_path)
    _make_draft(ws)
    res = cp.promote("penicillin_history", approver="")
    assert res["ok"] is False and "approver" in res["reason"]


def test_promote_elevates_to_memory_and_clears_review(monkeypatch, tmp_path) -> None:
    ws = _store(monkeypatch, tmp_path)
    _make_draft(ws)
    res = cp.promote("penicillin_history", approver="curator@example.org")
    assert res["ok"] is True and res["promotedTo"] == "memory"

    page = ws.read_page("penicillin_history")
    assert page is not None
    assert page.meta.get("needsReview") is False
    assert page.meta.get("reviewedBy") == "curator@example.org"
    assert page.meta.get("reviewStatus") == "approved"
    # draft file removed (no cross-tier duplication)
    assert not (ws.DRAFT_DIR / "penicillin_history.md").exists()
    # memory file exists
    assert (ws.MEMORY_DIR / "penicillin_history.md").exists()


def test_promoted_page_no_longer_pending(monkeypatch, tmp_path) -> None:
    ws = _store(monkeypatch, tmp_path)
    _make_draft(ws)
    cp.promote("penicillin_history", approver="curator")
    assert cp.pending_reviews() == []


def test_promote_missing_draft_fails(monkeypatch, tmp_path) -> None:
    _store(monkeypatch, tmp_path)
    res = cp.promote("does_not_exist", approver="curator")
    assert res["ok"] is False and "no draft" in res["reason"]


def test_reject_removes_draft(monkeypatch, tmp_path) -> None:
    ws = _store(monkeypatch, tmp_path)
    _make_draft(ws)
    res = cp.reject("penicillin_history", approver="curator", reason="source too thin")
    assert res["ok"] is True and res["existed"] is True
    assert not (ws.DRAFT_DIR / "penicillin_history.md").exists()
    assert cp.pending_reviews() == []


def test_gate_failing_draft_sorts_first_and_cannot_be_promoted(monkeypatch, tmp_path) -> None:
    from okf import frontmatter

    ws = _store(monkeypatch, tmp_path)
    _make_draft(ws, "good_page")
    # Simulate a draft that was HAND-EDITED after creation to introduce a forbidden
    # attribution (upsert would have rejected it, so we write the file directly). Re-gating
    # at review is exactly what catches this.
    ws.DRAFT_DIR.mkdir(parents=True, exist_ok=True)
    bad_meta = {"id": "bad_page", "pageType": "text", "attributedAuthor": "confucius",
                "authorConfidence": "attributed", "doNotAttributeTo": ["confucius"],
                "needsReview": True}
    (ws.DRAFT_DIR / "bad_page.md").write_text(
        frontmatter.serialize(bad_meta, "# Bad\n"), encoding="utf-8")

    pend = cp.pending_reviews()
    assert pend[0]["id"] == "bad_page" and pend[0]["gatePasses"] is False  # failing sorts first
    # Re-gating is fail-closed: a tampered draft cannot be promoted into canon.
    res = cp.promote("bad_page", approver="curator")
    assert res["ok"] is False and "gate rejected" in res["reason"]
