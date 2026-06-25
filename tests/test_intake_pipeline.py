#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Pipeline tests for intake front gate and prompt composition."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.intake import CONTRACT_PROMPT_HEADER, INTAKE_PROMPT_HEADER  # noqa: E402
from tools import run_hidden_eval_sophia as runner  # noqa: E402


CONFIG = runner.RunConfig(backend="fake", timeout_sec=1)
CASE = {
    "id": "intake_001",
    "domain": "tool_use",
    "prompt": "Inspect the repo state.",
    "materials": [],
    "requiresToolLog": True,
    "scoring": {"maxPoints": 1, "rubric": ["x"], "mustInclude": ["Decision"]},
}


def _fake_answer() -> dict:
    return {
        "backend": "fake",
        "returncode": 0,
        "elapsedSec": 0.01,
        "answer": "Decision: inspected. 中文摘要: 好.",
        "stderrTail": "",
    }


def _stub_pipeline():
    saved = {
        "call_model": runner.call_model,
        "retrieve": runner.retrieve,
        "gather_evidence": runner.gather_evidence,
        "run_operational_tools": runner.run_operational_tools,
        "append_learning_memory": runner.append_learning_memory,
        "run_intake": runner.run_intake,
    }

    class FakeChunk:
        path = "data/attributions.json"
        title = "stub"
        excerpt = "stub excerpt"
        score = 0.9

    runner.call_model = lambda system, user, *, backend, timeout_sec, grok_cwd=None: {**_fake_answer(), "_system": system, "_user": user}
    runner.retrieve = lambda query, *, top_k=8: [FakeChunk()]
    runner.gather_evidence = lambda query, **kw: {"localSources": [{"url": "docs/x.md"}], "web": {"online": False, "sources": []}}
    runner.run_operational_tools = lambda case: {"commands": [{"cmd": "git status", "returncode": 0}]}
    runner.append_learning_memory = lambda case: {"appended": True, "oldHashChanged": False, "memoryFile": "stub"}
    return saved


def _restore(saved: dict) -> None:
    for name, value in saved.items():
        setattr(runner, name, value)


def test_original_prompt_is_verbatim_before_contract_metadata() -> None:
    saved = _stub_pipeline()
    try:
        result = runner.run_case(CASE, "unit", config=CONFIG, ablation=runner.SOPHIA_FULL)
    finally:
        _restore(saved)
    user_prompt = result["modelLog"]["_user"]
    assert INTAKE_PROMPT_HEADER in user_prompt
    assert CONTRACT_PROMPT_HEADER in user_prompt
    assert user_prompt.index(CASE["prompt"]) < user_prompt.index(CONTRACT_PROMPT_HEADER)
    assert result["intakeContract"]["original_prompt"] == CASE["prompt"]
    assert result["intakeExecutionGate"]["verdict"] == "accepted"


def test_use_intake_flag_disables_contract_prompt() -> None:
    saved = _stub_pipeline()
    try:
        result = runner.run_case(CASE, "unit", config=CONFIG, ablation=runner.ABLATION_MODES["sophia-no-intake"])
    finally:
        _restore(saved)
    assert result["intakeContract"] == {}
    assert INTAKE_PROMPT_HEADER not in result["modelLog"]["_user"]


def test_invalid_intake_contract_fails_closed_before_model_call() -> None:
    saved = _stub_pipeline()
    calls = {"model": 0}

    def fake_model(system, user, *, backend, timeout_sec, grok_cwd=None):
        calls["model"] += 1
        return {**_fake_answer(), "_system": system, "_user": user}

    runner.call_model = fake_model
    runner.run_intake = lambda *args, **kwargs: {"ok": False, "contract": {}, "errors": ["bad contract"]}
    try:
        result = runner.run_case(CASE, "unit", config=CONFIG, ablation=runner.SOPHIA_FULL)
    finally:
        _restore(saved)
    assert calls["model"] == 0
    assert result["gate"]["verdict"] == "held"
    assert result["gate"]["held_reason"] == "needs_human"


def main() -> int:
    test_original_prompt_is_verbatim_before_contract_metadata()
    test_use_intake_flag_disables_contract_prompt()
    test_invalid_intake_contract_fails_closed_before_model_call()
    print("test_intake_pipeline: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
