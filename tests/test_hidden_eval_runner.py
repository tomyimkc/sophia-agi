#!/usr/bin/env python3
"""Smoke tests for the full Sophia hidden-eval runner helpers."""

from __future__ import annotations

import sys
from pathlib import Path
import subprocess
import tempfile
import os
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_hidden_eval_sophia as runner  # noqa: E402


def test_backend_preflight_failure() -> None:
    original = runner.call_model

    def fake_call_model(system: str, user: str, *, backend: str, timeout_sec: int, grok_cwd=None) -> dict:
        return {"returncode": 1, "answer": "", "stderrTail": "backend unavailable"}

    runner.call_model = fake_call_model
    try:
        result = runner.backend_preflight(backend="grok", timeout_sec=1)
    finally:
        runner.call_model = original

    assert result["ok"] is False
    assert result["returncode"] == 1
    assert "backend unavailable" in result["stderrTail"]


def test_backend_preflight_success() -> None:
    original = runner.call_model

    def fake_call_model(system: str, user: str, *, backend: str, timeout_sec: int, grok_cwd=None) -> dict:
        return {"returncode": 0, "answer": "SOPHIA_PREFLIGHT_OK\nDecision: ok\n中文摘要: ok", "stderrTail": ""}

    runner.call_model = fake_call_model
    try:
        result = runner.backend_preflight(backend="grok", timeout_sec=1)
    finally:
        runner.call_model = original

    assert result["ok"] is True
    assert result["returncode"] == 0


def test_backend_preflight_accepts_chinese_summary_without_label() -> None:
    original = runner.call_model

    def fake_call_model(system: str, user: str, *, backend: str, timeout_sec: int, grok_cwd=None) -> dict:
        return {"returncode": 0, "answer": "SOPHIA_PREFLIGHT_OK\nDecision: ok\n预检通过。", "stderrTail": ""}

    runner.call_model = fake_call_model
    try:
        result = runner.backend_preflight(backend="deepseek", timeout_sec=1)
    finally:
        runner.call_model = original

    assert result["ok"] is True


def test_learning_probe_skips_append_when_pretest_fails() -> None:
    original_call = runner.call_model
    original_append = runner.append_learning_memory
    append_called = {"value": False}

    def fake_call_model(system: str, user: str, *, backend: str, timeout_sec: int, grok_cwd=None) -> dict:
        return {"returncode": 1, "answer": "", "stderrTail": "pretest failed"}

    def fake_append(case: dict) -> dict:
        append_called["value"] = True
        return {"appended": True, "oldHashChanged": False}

    runner.call_model = fake_call_model
    runner.append_learning_memory = fake_append
    try:
        result = runner.run_learning_probe(
            {"id": "learning_001", "domain": "learning", "prompt": "learn", "requiresMemoryDiff": True},
            "(context)",
            "system",
            backend="grok",
            timeout_sec=1,
        )
    finally:
        runner.call_model = original_call
        runner.append_learning_memory = original_append

    assert append_called["value"] is False
    assert result["skippedAppend"] is True
    assert result["memoryDiff"] == {}


def test_grok_prompt_file_is_removed() -> None:
    original_run = subprocess.run
    captured: dict[str, Path] = {}

    class FakeProc:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(command, cwd, text, capture_output, timeout, check):
        prompt_path = Path(command[command.index("--prompt-file") + 1])
        captured["prompt_path"] = prompt_path
        assert Path(cwd).exists()
        assert prompt_path.exists()
        return FakeProc()

    subprocess.run = fake_run
    try:
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.run_grok_direct("system", "user", timeout_sec=1, grok_cwd=Path(tmp))
    finally:
        subprocess.run = original_run

    assert result["returncode"] == 0
    assert captured["prompt_path"].exists() is False


def test_deepseek_missing_key_fails_cleanly() -> None:
    original = os.environ.pop("DEEPSEEK_API_KEY", None)
    try:
        result = runner.run_deepseek("system", "user", timeout_sec=1)
    finally:
        if original is not None:
            os.environ["DEEPSEEK_API_KEY"] = original

    assert result["returncode"] == 1
    assert "DEEPSEEK_API_KEY" in result["stderrTail"]


def test_deepseek_success_parses_answer() -> None:
    original_key = os.environ.get("DEEPSEEK_API_KEY")
    original_urlopen = urllib.request.urlopen

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"ok answer"}}]}'

    def fake_urlopen(request, timeout):
        assert request.full_url.endswith("/chat/completions")
        return FakeResponse()

    os.environ["DEEPSEEK_API_KEY"] = "test-key"
    urllib.request.urlopen = fake_urlopen
    try:
        result = runner.run_deepseek("system", "user", timeout_sec=1)
    finally:
        urllib.request.urlopen = original_urlopen
        if original_key is None:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        else:
            os.environ["DEEPSEEK_API_KEY"] = original_key

    assert result["returncode"] == 0
    assert result["answer"] == "ok answer"


def test_user_prompt_includes_coding_council_and_operational_evidence() -> None:
    prompt = runner.build_user_prompt(
        {"id": "coding_001", "domain": "coding", "prompt": "Fix Python bug.", "materials": []},
        "(context)",
        coding_council="## Coding Council\n- Python readability seat",
        evidence_context="## Evidence search context\n- [local 1] docs/example.md",
        operational_evidence="### Tool log evidence\n- `pytest` returned 0.",
    )
    assert "Python readability seat" in prompt
    assert "`pytest` returned 0" in prompt
    assert "[local 1]" in prompt
    assert "Rubric Evidence Map" in prompt
    assert "For coding tasks" in prompt


def test_manual_review_template_has_two_reviewers_and_missed_rubric() -> None:
    pack = {
        "packId": "unit",
        "cases": [
            {
                "id": "coding_001",
                "domain": "coding",
                "prompt": "Fix bug.",
                "materials": [],
                "scoring": {
                    "rubric": ["semantic"],
                    "semanticChecks": [{"id": "sem_001", "description": "expert accepts"}],
                },
            }
        ],
    }
    report = {
        "results": [
            {
                "id": "coding_001",
                "domain": "coding",
                "passed": False,
                "score": 0,
                "maxPoints": 1,
                "passedChecks": 0,
                "totalChecks": 1,
                "requiresManualReview": True,
                "manualReview": "required",
                "missedRubric": [{"type": "semantic-pending-manual-review", "label": "sem_001"}],
            }
        ]
    }
    payload = {
        "responses": {"coding_001": "answer"},
        "sources": {"coding_001": ["data/coding_council_figures.json"]},
        "gates": {"coding_001": {"passed": False}},
        "webEvidence": {
            "coding_001": {
                "localSources": [{"url": "docs/example.md"}],
                "web": {"online": False, "sources": []},
                "warnings": ["Online search is disabled"],
            }
        },
        "rubricReviews": {"coding_001": {"strictPassReady": False, "missing": [{"type": "semantic"}]}},
        "codingCouncilRoutes": {"coding_001": {"languageSeats": [{"displayName": "Python readability seat"}]}},
    }
    template = runner.manual_review_template(pack, report, payload)
    check = template["manualJudgements"]["coding_001"]["semanticChecks"]["sem_001"]
    assert len(check["reviewers"]) == 2
    assert "adjudication" in check
    assert template["manualJudgements"]["coding_001"]["missedRubric"]
    assert template["manualJudgements"]["coding_001"]["rubricReview"]["missing"]
    assert template["manualJudgements"]["coding_001"]["webEvidence"]["warnings"]


def test_empty_successful_answer_is_repairable() -> None:
    assert runner.should_attempt_repair(
        enabled=True,
        first={"returncode": 0, "answer": ""},
        provisional={"passed": False},
        gate={"passed": False},
    )


def test_council_public_summary_uses_engineering_label() -> None:
    summary = runner.council_public_summary(
        {
            "codingCouncilRoutes": {
                "tool_001": {
                    "specialistSeats": [
                        {"displayName": "Tool-calling reliability engineer"},
                    ]
                }
            }
        }
    )
    assert summary["engineeringCouncilCasesRouted"] == 1
    assert summary["cases"]["tool_001"]["seats"] == ["Tool-calling reliability engineer"]


def test_sanitized_report_includes_review_and_web_evidence_health() -> None:
    pack = {"packId": "unit", "cases": [{"id": "logic_001"}]}
    private_report = {
        "passed": 0,
        "totalCases": 1,
        "score": 0,
        "maxScore": 1,
        "scorePct": 0,
        "results": [
            {
                "id": "logic_001",
                "domain": "logic",
                "passed": False,
                "score": 0,
                "maxPoints": 1,
                "passedChecks": 0,
                "totalChecks": 1,
                "missedRubric": [{"type": "required-evidence"}],
            }
        ],
    }
    payload = {
        "model": "unit",
        "date": "2026-06-20T00:00:00",
        "responses": {"logic_001": "answer"},
        "logs": {"logic_001": {"returncode": 0}},
        "rubricReviews": {"logic_001": {"strictPassReady": False, "missing": [{"type": "required"}]}},
        "webEvidence": {"logic_001": {"web": {"online": True, "sources": [{"url": "https://arxiv.org"}]}}},
    }
    report = runner.sanitized_report(pack, private_report, payload)
    assert report["rubricReviewHealth"]["totalReviewed"] == 1
    assert report["rubricReviewHealth"]["casesWithMissingItems"]["logic_001"] == 1
    assert report["webEvidenceHealth"]["enabledCases"] == 1
    assert report["caseResults"][0]["rubricReviewMissingCount"] == 1


def main() -> int:
    test_backend_preflight_failure()
    test_backend_preflight_success()
    test_backend_preflight_accepts_chinese_summary_without_label()
    test_learning_probe_skips_append_when_pretest_fails()
    test_grok_prompt_file_is_removed()
    test_deepseek_missing_key_fails_cleanly()
    test_deepseek_success_parses_answer()
    test_user_prompt_includes_coding_council_and_operational_evidence()
    test_manual_review_template_has_two_reviewers_and_missed_rubric()
    test_empty_successful_answer_is_repairable()
    test_council_public_summary_uses_engineering_label()
    test_sanitized_report_includes_review_and_web_evidence_health()
    print("test_hidden_eval_runner: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
