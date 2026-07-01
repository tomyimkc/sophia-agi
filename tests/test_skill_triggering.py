#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Hybrid skill router: stemming, synonyms, IDF, fuzzy fallback, ranked, telemetry."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import skills  # noqa: E402


def _write_skills(d: Path) -> None:
    (d / "coding-debugging.json").write_text(json.dumps({
        "name": "coding-debugging", "whenToUse": "Fix a bug or failing test.",
        "triggers": ["bug", "debug", "test", "failing", "error"],
        "requiredTools": ["terminal"], "workflow": ["repro"], "ioSchema": {"input": {}, "output": {}},
        "verification": ["green"], "commonFailures": ["x"], "examples": [{"input": "a", "output": "b"}],
    }), encoding="utf-8")
    (d / "source-verification.json").write_text(json.dumps({
        "name": "source-verification", "whenToUse": "Check who wrote a text and its provenance.",
        "triggers": ["provenance", "attribution", "author", "citation", "wrote"],
        "requiredTools": ["lookup"], "workflow": ["check"], "ioSchema": {"input": {}, "output": {}},
        "verification": ["cited"], "commonFailures": ["x"], "examples": [{"input": "a", "output": "b"}],
    }), encoding="utf-8")


def test_stemming_matches_inflected_goal():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t); _write_skills(d)
        # "debugging" must stem to "debug" and hit coding-debugging
        best = skills.select("I am debugging a failing pytest", skill_dir=d)
        assert best and best["name"] == "coding-debugging", best


def test_synonym_maps_citation_to_provenance():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t); _write_skills(d)
        best = skills.select("is this citation correct", skill_dir=d)
        assert best and best["name"] == "source-verification", best


def test_ranked_topk_is_sorted_and_deterministic():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t); _write_skills(d)
        ranked = skills.select_ranked("debug a failing test and check the author", skill_dir=d, top_k=2)
        assert len(ranked) >= 1
        scores = [r["score"] for r in ranked]
        assert scores == sorted(scores, reverse=True), scores
        # deterministic across calls
        again = skills.select_ranked("debug a failing test and check the author", skill_dir=d, top_k=2)
        assert [r["skill"]["name"] for r in ranked] == [r["skill"]["name"] for r in again]


def test_no_spurious_match_returns_none():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t); _write_skills(d)
        assert skills.select("xyzzy plugh frobnicate", skill_dir=d) is None


def test_fuzzy_fallback_on_typo():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t); _write_skills(d)
        # "debuging" (typo) has no exact token but char-ngrams should still surface the skill
        ranked = skills.select_ranked("debuging the cod", skill_dir=d, top_k=1, min_score=0.1)
        assert ranked and ranked[0]["via"] in ("fuzzy", "keyword")


def test_telemetry_log_written():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t); _write_skills(d)
        log = Path(t) / "log.jsonl"
        skills.select("debug a failing test", skill_dir=d, log_path=log)
        rows = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
        assert rows and rows[0]["skill_id"] == "coding-debugging" and rows[0]["via"] == "keyword"


def test_embed_fn_can_rescue_zero_keyword_match():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t); _write_skills(d)
        # a goal sharing no tokens; a trivial embed_fn that maps everything similar rescues it
        def embed(_text: str):
            return [1.0, 0.0, 0.0]
        ranked = skills.select_ranked("completely unrelated phrasing", skill_dir=d, top_k=1,
                                      min_score=0.5, embed_fn=embed)
        assert ranked and ranked[0]["via"] in ("embed", "fuzzy", "keyword")


def main() -> int:
    test_stemming_matches_inflected_goal()
    test_synonym_maps_citation_to_provenance()
    test_ranked_topk_is_sorted_and_deterministic()
    test_no_spurious_match_returns_none()
    test_fuzzy_fallback_on_typo()
    test_telemetry_log_written()
    test_embed_fn_can_rescue_zero_keyword_match()
    print("test_skill_triggering: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
