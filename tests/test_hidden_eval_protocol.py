#!/usr/bin/env python3
"""Smoke tests for hidden evaluation pack helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.hidden_eval_commitments import build_commitments  # noqa: E402
from tools.hidden_eval_protocol import score_pack, validate_pack  # noqa: E402


def sample_pack() -> dict:
    return {
        "packId": "unit-hidden-pack",
        "createdAt": "2026-06-19",
        "visibility": "private-hidden",
        "salt": "unit-salt",
        "cases": [
            {
                "id": "logic_001",
                "domain": "logic",
                "prompt": "Return the word valid and avoid the word invalid.",
                "scoring": {
                    "maxPoints": 1,
                    "mustInclude": ["valid"],
                    "mustAvoid": ["invalid"],
                    "rubric": ["Uses required logic verdict."],
                },
            }
        ],
    }


def test_validate_pack() -> None:
    assert validate_pack(sample_pack()) == []


def test_score_pack() -> None:
    report = score_pack(sample_pack(), {"model": "unit", "responses": {"logic_001": "valid"}})
    assert report["passed"] == 1
    assert report["scorePct"] == 100.0


def test_alias_scoring() -> None:
    pack = sample_pack()
    case = pack["cases"][0]
    case["scoring"]["mustInclude"] = ["source discipline"]
    case["scoring"]["aliases"] = {"source discipline": ["source-disciplined", "provenance-aware"]}
    report = score_pack(pack, {"model": "unit", "responses": {"logic_001": "This is provenance-aware."}})
    assert report["passed"] == 1
    assert report["scorePct"] == 100.0


def test_regex_scoring() -> None:
    pack = sample_pack()
    case = pack["cases"][0]
    case["scoring"]["mustInclude"] = ["re:\\bvalid\\s+argument\\b"]
    case["scoring"]["mustAvoid"] = []
    report = score_pack(pack, {"model": "unit", "responses": {"logic_001": "This is a valid argument."}})
    assert report["passed"] == 1
    assert report["scorePct"] == 100.0


def test_invalid_regex_is_validation_error() -> None:
    pack = sample_pack()
    pack["cases"][0]["scoring"]["mustInclude"] = ["re:["]
    errors = validate_pack(pack)
    assert any("invalid regex" in error for error in errors)


def test_empty_response_scores_zero() -> None:
    report = score_pack(sample_pack(), {"model": "unit", "responses": {"logic_001": ""}})
    result = report["results"][0]
    assert report["passed"] == 0
    assert report["scorePct"] == 0.0
    assert result["score"] == 0
    assert result["passedChecks"] == 0
    assert result["emptyResponse"] is True
    assert "empty model response" in result["operationalFailures"]


def test_manual_semantic_review_required() -> None:
    pack = sample_pack()
    case = pack["cases"][0]
    case["scoring"]["mustInclude"] = ["valid"]
    case["scoring"]["mustAvoid"] = []
    case["scoring"]["semanticChecks"] = [
        {"id": "sem_001", "description": "Reviewer confirms the answer distinguishes form from soundness."}
    ]
    pending = score_pack(pack, {"model": "unit", "responses": {"logic_001": "valid"}})
    assert pending["passed"] == 0
    assert pending["results"][0]["requiresManualReview"] is True
    assert "manual semantic review pending" in pending["results"][0]["operationalFailures"][0]

    reviewed = score_pack(
        pack,
        {
            "model": "unit",
            "responses": {"logic_001": "valid"},
            "manualJudgements": {
                "logic_001": {
                    "semanticChecks": {
                        "sem_001": {"passed": True, "judge": "unit-reviewer", "notes": "ok"}
                    }
                }
            },
        },
    )
    assert reviewed["passed"] == 1
    assert reviewed["scorePct"] == 100.0


def test_two_pass_manual_review_requires_adjudication_on_disagreement() -> None:
    pack = sample_pack()
    case = pack["cases"][0]
    case["scoring"]["mustInclude"] = ["valid"]
    case["scoring"]["mustAvoid"] = []
    case["scoring"]["semanticChecks"] = [
        {"id": "sem_001", "description": "Reviewer confirms semantic adequacy."}
    ]
    report = score_pack(
        pack,
        {
            "model": "unit",
            "responses": {"logic_001": "valid"},
            "manualJudgements": {
                "logic_001": {
                    "semanticChecks": {
                        "sem_001": {
                            "reviewers": [
                                {"judge": "reviewer-a", "passed": True, "notes": "ok"},
                                {"judge": "reviewer-b", "passed": False, "notes": "too thin"},
                            ]
                        }
                    }
                }
            },
        },
    )
    result = report["results"][0]
    assert report["passed"] == 0
    assert result["semanticResults"][0]["status"] == "needs-adjudication"
    assert "needs adjudication" in " ".join(result["operationalFailures"])

    adjudicated = score_pack(
        pack,
        {
            "model": "unit",
            "responses": {"logic_001": "valid"},
            "manualJudgements": {
                "logic_001": {
                    "semanticChecks": {
                        "sem_001": {
                            "reviewers": [
                                {"judge": "reviewer-a", "passed": True, "notes": "ok"},
                                {"judge": "reviewer-b", "passed": False, "notes": "too thin"},
                            ],
                            "adjudication": {"judge": "lead", "passed": True, "notes": "accept"},
                        }
                    }
                }
            },
        },
    )
    assert adjudicated["passed"] == 1


def test_missed_rubric_summary_is_reported() -> None:
    report = score_pack(sample_pack(), {"model": "unit", "responses": {"logic_001": "invalid"}})
    result = report["results"][0]
    assert result["missedRubric"]
    assert any(item["type"] == "forbidden-claim-present" for item in result["missedRubric"])


def test_operational_tool_and_memory_checks() -> None:
    pack = sample_pack()
    pack["cases"] = [
        {
            "id": "tool_001",
            "domain": "tool_use",
            "prompt": "Run a tool.",
            "requiresToolLog": True,
            "scoring": {"maxPoints": 1, "mustInclude": ["done"], "rubric": ["tool evidence"]},
        },
        {
            "id": "learning_001",
            "domain": "learning",
            "prompt": "Append memory.",
            "requiresMemoryDiff": True,
            "scoring": {"maxPoints": 1, "mustInclude": ["done"], "rubric": ["memory evidence"]},
        },
    ]
    responses = {
        "model": "unit",
        "responses": {"tool_001": "done", "learning_001": "done"},
        "toolLogs": {"tool_001": {"commands": [{"cmd": "echo ok", "returncode": 0}]}},
        "memoryDiffs": {"learning_001": {"appended": True, "oldHashChanged": False}},
    }
    report = score_pack(pack, responses)
    assert report["passed"] == 2
    assert report["scorePct"] == 100.0


def test_failing_tool_log_fails_operational_check() -> None:
    pack = sample_pack()
    pack["cases"] = [
        {
            "id": "tool_001",
            "domain": "tool_use",
            "prompt": "Run a tool.",
            "requiresToolLog": True,
            "scoring": {"maxPoints": 1, "mustInclude": ["done"], "rubric": ["tool evidence"]},
        }
    ]
    responses = {
        "model": "unit",
        "responses": {"tool_001": "done"},
        "toolLogs": {"tool_001": {"commands": [{"cmd": "false", "returncode": 1}]}},
    }
    report = score_pack(pack, responses)
    assert report["passed"] == 0
    assert report["results"][0]["score"] == 0.5
    assert "missing or failing required tool log" in report["results"][0]["operationalFailures"]


def test_commitments_hide_prompt() -> None:
    commitments = build_commitments(sample_pack())
    assert commitments["caseCount"] == 1
    dumped = str(commitments)
    assert "Return the word" not in dumped
    assert len(commitments["cases"][0]["sha256"]) == 64


def main() -> int:
    test_validate_pack()
    test_score_pack()
    test_alias_scoring()
    test_regex_scoring()
    test_invalid_regex_is_validation_error()
    test_empty_response_scores_zero()
    test_manual_semantic_review_required()
    test_two_pass_manual_review_requires_adjudication_on_disagreement()
    test_missed_rubric_summary_is_reported()
    test_operational_tool_and_memory_checks()
    test_failing_tool_log_fails_operational_check()
    test_commitments_hide_prompt()
    print("test_hidden_eval_protocol: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
