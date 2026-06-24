#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the skills registry."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import skills  # noqa: E402

EXPECTED = {
    "coding-debugging",
    "research-rag",
    "terminal-automation",
    "repo-analysis",
    "long-context-summarization",
    "eval-generation",
    "lora-dataset-creation",
}


def test_all_skills_load_and_validate() -> None:
    loaded = skills.load_all()
    assert EXPECTED <= set(loaded)
    for skill in loaded.values():
        assert skills.validate_skill(skill) == []
        assert skill["workflow"] and skill["verification"] and skill["examples"]


def test_validate_catches_missing_fields() -> None:
    errors = skills.validate_skill({"name": "x"})
    assert any("workflow" in e for e in errors)
    assert any("verification" in e for e in errors)


def test_select_matches_intent() -> None:
    assert skills.select("fix a failing pytest bug in the auth module")["name"] == "coding-debugging"
    assert skills.select("who wrote this text, with cited sources")["name"] == "research-rag"
    assert skills.select("build a LoRA SFT dataset from traces")["name"] == "lora-dataset-creation"
    assert skills.select("summarize this very long transcript")["name"] == "long-context-summarization"


def test_select_returns_none_for_unrelated() -> None:
    assert skills.select("xyzzy plugh frobnicate") is None


def test_get_known_and_unknown() -> None:
    assert skills.get("repo-analysis")["name"] == "repo-analysis"
    assert skills.get("nope") is None


def main() -> int:
    test_all_skills_load_and_validate()
    test_validate_catches_missing_fields()
    test_select_matches_intent()
    test_select_returns_none_for_unrelated()
    test_get_known_and_unknown()
    print("test_skills: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
