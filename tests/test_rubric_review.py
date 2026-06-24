#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for deterministic strict-pass rubric review."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rubric_review import build_rubric_review, format_rubric_review  # noqa: E402


def test_rubric_review_flags_missing_required_item() -> None:
    case = {
        "id": "logic_001",
        "domain": "logic",
        "scoring": {"mustInclude": ["counterexample"], "mustAvoid": ["certainty"], "semanticChecks": []},
    }
    review = build_rubric_review(
        case,
        "Decision: likely. 中文摘要: 短。",
        {"passed": False, "semanticResults": []},
        {"passed": True},
    )
    assert review["strictPassReady"] is False
    assert any(item["type"] == "required-evidence" for item in review["missing"])
    assert "counterexample" in format_rubric_review(review)


def test_rubric_review_passes_operational_evidence() -> None:
    case = {
        "id": "tool_001",
        "domain": "tool_use",
        "requiresToolLog": True,
        "scoring": {"mustInclude": ["return code"], "mustAvoid": [], "semanticChecks": []},
    }
    review = build_rubric_review(
        case,
        "The command had return code 0.\nDecision: pass.\n中文摘要: 已验证。",
        {"passed": True, "semanticResults": []},
        {"passed": True},
        tool_log={"commands": [{"cmd": "python -m json.tool file.json", "returncode": 0}]},
    )
    assert review["strictPassReady"] is True
    assert review["operationalEvidence"]["passed"] is True


def test_rubric_review_blocks_pending_manual_semantic_review() -> None:
    case = {
        "id": "psych_001",
        "domain": "psychology",
        "scoring": {
            "mustInclude": [],
            "mustAvoid": [],
            "semanticChecks": [{"id": "sem_001", "description": "expert accepts nuance"}],
        },
    }
    review = build_rubric_review(
        case,
        "Decision: answer.\n中文摘要: 简短。",
        {"passed": False, "semanticResults": [{"id": "sem_001", "status": "pending-manual-review"}]},
        {"passed": True},
    )
    assert review["strictPassReady"] is False
    assert review["missing"][0]["type"] == "semantic-pending-manual-review"


def main() -> int:
    test_rubric_review_flags_missing_required_item()
    test_rubric_review_passes_operational_evidence()
    test_rubric_review_blocks_pending_manual_semantic_review()
    print("test_rubric_review: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
