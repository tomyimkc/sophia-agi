# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for sourced fill — promote gap stubs from trusted sources, no fabrication.

Deterministic & offline: extraction is an injected stub (no LLM); the allowlist and the
provenance gate are the real components.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import source_fill as sf  # noqa: E402

SRC = sf.TrustedSource(
    id="raw/penicillin.txt", name="penicillin",
    text="Penicillin was discovered by Alexander Fleming in 1928. It is an antibiotic.",
)


def _good_extractor(_text, _sid):
    return {
        "title": "Penicillin", "pageType": "concept", "domain": "science",
        "attributedAuthor": "alexander_fleming", "authorConfidence": "attributed",
        "summary": "Penicillin is an antibiotic discovered by Alexander Fleming in 1928.",
    }


def test_allowlist_boundary() -> None:
    assert sf.is_allowlisted("raw/penicillin.txt") is True       # operator-curated dir
    assert sf.is_allowlisted("https://arxiv.org/abs/1234") is True  # authority domain
    assert sf.is_allowlisted("model:hallucinated") is False      # model output is not a source
    assert sf.is_allowlisted("") is False


def test_match_source_requires_overlap() -> None:
    assert sf.match_source("penicillin_history", ["Who discovered penicillin?"], [SRC]) is SRC
    assert sf.match_source("quantum_gravity", ["string theory"], [SRC]) is None


def test_load_trusted_sources(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("Alpha source text about alpha.", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    srcs = sf.load_trusted_sources(tmp_path)
    assert [s.name for s in srcs] == ["a"]  # empty skipped
    assert srcs[0].id == "raw/a.txt"


def test_untrusted_source_is_refused_before_extraction(monkeypatch) -> None:
    called = []

    def _spy_extractor(t, s):
        called.append(1)
        return _good_extractor(t, s)

    bad = sf.TrustedSource(id="model:made_up", name="made_up", text="penicillin discovered by someone")
    r = sf.fill_stub("penicillin_history", queries=["penicillin"], sources=[bad],
                     extractor=_spy_extractor, write=False)
    assert r["ok"] is False and r["reason"] == "source not allowlisted"
    assert called == []  # refused BEFORE spending extraction


def test_no_matching_source_is_refused() -> None:
    r = sf.fill_stub("quantum_gravity", queries=["string theory"], sources=[SRC],
                     extractor=_good_extractor, write=False)
    assert r["ok"] is False and "no matching" in r["reason"]


def _with_temp_store(monkeypatch, tmp_path):
    import agent.wiki_store as ws

    monkeypatch.setattr(ws, "CANONICAL_DIR", tmp_path / "wiki")
    monkeypatch.setattr(ws, "MEMORY_DIR", tmp_path / "mem")
    monkeypatch.setattr(ws, "DRAFT_DIR", tmp_path / "wiki" / "drafts")
    (tmp_path / "wiki").mkdir(parents=True, exist_ok=True)
    return ws


def test_fill_promotes_stub_from_none_extant(monkeypatch, tmp_path) -> None:
    ws = _with_temp_store(monkeypatch, tmp_path)
    from agent.gap_ingest import draft_stub

    meta, body = draft_stub("penicillin_history", queries=["Who discovered penicillin?"], gap_hits=2)
    ws.upsert("penicillin_history", meta=meta, body=body, tier="draft")
    assert ws.read_page("penicillin_history").meta["authorConfidence"] == "none_extant"

    r = sf.fill_stub("penicillin_history", queries=["Who discovered penicillin?"], sources=[SRC],
                     extractor=_good_extractor, write=True)
    assert r["ok"] is True
    page = ws.read_page("penicillin_history")
    assert page.meta["authorConfidence"] == "attributed"   # promoted
    assert page.meta["provenance"] == "librarian_fill"
    assert page.meta["needsReview"] is True                 # still awaits human sign-off


def test_fabricated_attribution_is_rejected_by_gate(monkeypatch, tmp_path) -> None:
    _with_temp_store(monkeypatch, tmp_path)

    def _fabricator(_t, _s):
        return {"title": "Penicillin", "pageType": "concept", "attributedAuthor": "confucius",
                "authorConfidence": "attributed", "doNotAttributeTo": ["confucius"],
                "summary": "Confucius discovered penicillin."}

    r = sf.fill_stub("penicillin_history", queries=["penicillin"], sources=[SRC],
                     extractor=_fabricator, write=True)
    assert r["ok"] is False
    assert any("doNotAttributeTo" in str(x) for x in (r.get("reasons") or []))


def test_dry_run_writes_nothing(monkeypatch, tmp_path) -> None:
    ws = _with_temp_store(monkeypatch, tmp_path)
    r = sf.fill_stub("penicillin_history", queries=["penicillin"], sources=[SRC],
                     extractor=_good_extractor, write=False)
    assert r["ok"] is True and r["wouldFill"] is True
    assert ws.read_page("penicillin_history") is None  # nothing written


def test_fill_gaps_only_targets_fillable_stubs(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace

    stub = SimpleNamespace(
        id="penicillin_history",
        meta={"provenance": "knowledge_gap", "authorConfidence": "none_extant"},
        body="## Queries that triggered this gap\n- Who discovered penicillin?\n",
    )
    canonical = SimpleNamespace(id="atomic_bomb", meta={"authorConfidence": "attributed"}, body="")
    report = sf.fill_gaps([stub, canonical], [SRC], extractor=_good_extractor, write=False)
    assert report["stubs"] == 1  # canonical page ignored
    assert report["filledOrWould"] == 1
