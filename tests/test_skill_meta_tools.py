#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""The skill-meta tools stay green on the real repo: description linter, unified
index (idempotent + load_all-safe), and the trigger learner's precision gate."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import skills  # noqa: E402
from tools import build_skill_index, lint_skill_descriptions, learn_triggers  # noqa: E402


def test_description_linter_passes_on_repo():
    assert lint_skill_descriptions.main([]) == 0


def test_linter_flags_a_thin_agent_skill():
    thin = {"kind": "agent-skill", "name": "x", "description": "helps with stuff",
            "trigger_tokens": {"helps", "stuff"}, "wants_slash": True}
    errors, _ = lint_skill_descriptions._lint_one(thin, shared=set())
    assert any("too short" in e for e in errors)
    assert any("action/trigger cue" in e for e in errors)


def test_unified_index_idempotent_and_router_safe():
    # regenerating must be a no-op (clean), and the index must NOT break the router
    assert build_skill_index.build(check=False) == 0
    assert build_skill_index.build(check=True) == 0
    loaded = skills.load_all()  # must not raise on index.json / forge_index.json
    assert "coding-debugging" in loaded
    idx = json.loads((ROOT / "skills" / "registry" / "index.json").read_text())
    ids = {s["id"] for s in idx["skills"]}
    assert "skill-author" in ids and "coding-debugging" in ids
    assert {s["layer"] for s in idx["skills"]} <= {"A", "B", "C"}


def test_learn_triggers_precision_gate():
    rows = [
        {"goal": "fix the oauth token bug", "skill_id": "coding-debugging", "accepted": True},
        {"goal": "oauth token refresh fails", "skill_id": "coding-debugging", "accepted": True},
        {"goal": "debug oauth token issue", "skill_id": "coding-debugging", "accepted": True},
        {"goal": "oauth thing", "skill_id": "coding-debugging", "accepted": False},
    ]
    # token 'oauth': support 4, accepts 3 -> precision 0.75 < 0.8 -> not promoted
    p = learn_triggers.propose(rows, min_support=3, min_precision=0.8)
    promoted = {t["token"] for lst in p.values() for t in lst}
    assert "oauth" not in promoted
    # 'token': support 3, accepts 3 -> precision 1.0 -> promoted
    assert "token" in promoted


def main() -> int:
    test_description_linter_passes_on_repo()
    test_linter_flags_a_thin_agent_skill()
    test_unified_index_idempotent_and_router_safe()
    test_learn_triggers_precision_gate()
    print("test_skill_meta_tools: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
