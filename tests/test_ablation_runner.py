#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Smoke tests for the Sophia baseline/ablation runner.

These assert that each ablation mode actually suppresses its component, that the
seven README modes exist, and that the delta/falsification math is correct.
Backend calls and IO are stubbed so the tests are hermetic and fast.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_hidden_eval_sophia as runner  # noqa: E402
from tools import run_ablation_sophia as ablation  # noqa: E402


def _fake_answer() -> dict:
    return {
        "backend": "fake",
        "returncode": 0,
        "elapsedSec": 0.01,
        "answer": "Source discipline applies. Decision: yes. 中文摘要: 好.",
        "stderrTail": "",
    }


def _stub_pipeline():
    """Replace IO/network/corpus calls with deterministic stubs."""
    saved = {
        "call_model": runner.call_model,
        "retrieve": runner.retrieve,
        "gather_evidence": runner.gather_evidence,
        "run_operational_tools": runner.run_operational_tools,
        "append_learning_memory": runner.append_learning_memory,
    }

    class FakeChunk:
        path = "data/attributions.json"
        title = "stub"
        excerpt = "stub excerpt"
        score = 0.9

    runner.call_model = lambda system, user, *, backend, timeout_sec, grok_cwd=None: {**_fake_answer(), "_system": system, "_user": user}
    runner.retrieve = lambda query, *, top_k=8: [FakeChunk()]
    runner.gather_evidence = lambda query, **kw: {"localSources": [{"url": "docs/x.md"}], "web": {"online": False, "sources": []}}
    runner.run_operational_tools = lambda case: ({"commands": [{"cmd": "git status", "returncode": 0}]} if case.get("requiresToolLog") else {})
    runner.append_learning_memory = lambda case: {"appended": True, "oldHashChanged": False, "memoryFile": "stub", "entryRecordId": "stub"}
    return saved


def _restore(saved: dict) -> None:
    for name, value in saved.items():
        setattr(runner, name, value)


CONFIG = runner.RunConfig(backend="fake", timeout_sec=1)

CODING_CASE = {
    "id": "coding_001",
    "domain": "coding",
    "prompt": "Fix a Python bug in the retry logic.",
    "materials": [],
    "scoring": {"maxPoints": 1, "rubric": ["x"], "mustInclude": ["Decision"]},
}
TOOL_CASE = {
    "id": "tool_001",
    "domain": "tool_use",
    "prompt": "Inspect the repo state.",
    "materials": [],
    "requiresToolLog": True,
    "scoring": {"maxPoints": 1, "rubric": ["x"], "mustInclude": ["Decision"]},
}
LEARNING_CASE = {
    "id": "learning_001",
    "domain": "learning",
    "prompt": "Learn a new fact and apply it.",
    "materials": [],
    "requiresMemoryDiff": True,
    "scoring": {"maxPoints": 1, "rubric": ["x"], "mustInclude": ["Decision"]},
}


def test_seven_modes_present() -> None:
    # The canonical ablation set consumed by ``--modes all`` (DEFAULT_MODE_ORDER).
    canonical = {
        "raw-model",
        "raw-model-plus-tools",
        "sophia-full",
        "sophia-no-kb",
        "sophia-no-gate",
        "sophia-no-memory",
        "sophia-no-council",
    }
    assert canonical <= set(runner.ABLATION_MODES)
    assert set(ablation.DEFAULT_MODE_ORDER) == canonical
    # W4 opt-in lever: registered for A/B but intentionally NOT in the canonical set.
    assert "sophia-claim-router" in runner.ABLATION_MODES
    assert "sophia-claim-router" not in ablation.DEFAULT_MODE_ORDER


def test_raw_model_suppresses_all_scaffolding() -> None:
    saved = _stub_pipeline()
    try:
        result = runner.run_case(CODING_CASE, "unit", config=CONFIG, ablation=runner.ABLATION_MODES["raw-model"])
    finally:
        _restore(saved)
    assert result["sources"] == []
    assert result["codingCouncilRoute"] == {}
    assert result["toolLog"] == {}
    assert result["webEvidence"] == {}
    assert result["gate"].get("gateApplied") is False
    # raw modes must not leak the Sophia source-discipline contract
    assert "Rubric Evidence Map" not in result["modelLog"].get("_user", "")
    assert "source discipline" not in result["modelLog"].get("_user", "").lower()


def test_raw_model_uses_neutral_system_prompt() -> None:
    saved = _stub_pipeline()
    try:
        result = runner.run_case(CODING_CASE, "unit", config=CONFIG, ablation=runner.ABLATION_MODES["raw-model"])
    finally:
        _restore(saved)
    assert result["modelLog"]["_system"] == runner.RAW_SYSTEM_PROMPT


def test_raw_plus_tools_includes_tool_log_only_for_tool_cases() -> None:
    saved = _stub_pipeline()
    try:
        tool_result = runner.run_case(TOOL_CASE, "unit", config=CONFIG, ablation=runner.ABLATION_MODES["raw-model-plus-tools"])
        coding_result = runner.run_case(CODING_CASE, "unit", config=CONFIG, ablation=runner.ABLATION_MODES["raw-model-plus-tools"])
    finally:
        _restore(saved)
    assert tool_result["toolLog"].get("commands")
    assert "Tool output available" in tool_result["modelLog"]["_user"]
    # a non-tool case yields no tool log even in plus-tools mode
    assert coding_result["toolLog"] == {}


def test_no_kb_drops_sources_but_keeps_council() -> None:
    saved = _stub_pipeline()
    try:
        result = runner.run_case(CODING_CASE, "unit", config=CONFIG, ablation=runner.ABLATION_MODES["sophia-no-kb"])
    finally:
        _restore(saved)
    assert result["sources"] == []
    assert result["webEvidence"] == {}
    assert result["codingCouncilRoute"] != {}  # council still active
    assert result["modelLog"]["_system"] == runner.MODE_PROMPTS["repo"]  # still Sophia prompt


def test_no_council_drops_council_but_keeps_kb() -> None:
    saved = _stub_pipeline()
    try:
        result = runner.run_case(CODING_CASE, "unit", config=CONFIG, ablation=runner.ABLATION_MODES["sophia-no-council"])
    finally:
        _restore(saved)
    assert result["codingCouncilRoute"] == {}
    assert result["sources"] != []  # KB still active
    # council is ablated at the prompt level too, not just the structured route
    assert result["modelLog"]["_system"] == runner.MODE_PROMPTS_NO_COUNCIL["repo"]
    assert result["modelLog"]["_system"] != runner.MODE_PROMPTS["repo"]


def test_no_gate_marks_gate_not_applied() -> None:
    saved = _stub_pipeline()
    try:
        result = runner.run_case(CODING_CASE, "unit", config=CONFIG, ablation=runner.ABLATION_MODES["sophia-no-gate"])
    finally:
        _restore(saved)
    assert result["gate"].get("gateApplied") is False


def test_no_memory_skips_learning_probe() -> None:
    saved = _stub_pipeline()
    try:
        result = runner.run_case(LEARNING_CASE, "unit", config=CONFIG, ablation=runner.ABLATION_MODES["sophia-no-memory"])
    finally:
        _restore(saved)
    assert result["memoryDiff"] == {}
    assert "learningProbe" not in result["modelLog"]


def test_sophia_full_runs_council_for_coding() -> None:
    saved = _stub_pipeline()
    try:
        result = runner.run_case(CODING_CASE, "unit", config=CONFIG, ablation=runner.SOPHIA_FULL)
    finally:
        _restore(saved)
    assert result["codingCouncilRoute"] != {}
    assert result["sources"] != []
    assert result["gate"].get("gateApplied") is not False  # real gate ran


def test_compute_deltas_full_minus_mode() -> None:
    summaries = {
        "sophia-full": {"score": 8.0, "scorePct": 80.0, "passed": 4},
        "raw-model": {"score": 5.0, "scorePct": 50.0, "passed": 1},
    }
    deltas = ablation.compute_deltas(summaries)
    assert deltas["raw-model"]["scoreDelta"] == 3.0
    assert deltas["raw-model"]["scorePctDelta"] == 30.0
    assert deltas["raw-model"]["passedDelta"] == 3
    assert deltas["raw-model"]["fullBeatsMode"] is True
    assert deltas["raw-model"]["meaningfulMargin"] is True


def test_falsification_triggers_when_raw_wins() -> None:
    summaries = {
        "sophia-full": {"score": 5.0, "scorePct": 50.0, "passed": 2},
        "raw-model": {"score": 6.0, "scorePct": 60.0, "passed": 3},
    }
    check = ablation.falsification_check(summaries)
    assert check["evaluable"] is True
    assert check["rawMatchesOrBeatsSophiaFull"] is True


def test_falsification_clear_when_sophia_wins() -> None:
    summaries = {
        "sophia-full": {"score": 9.0, "scorePct": 90.0, "passed": 5},
        "raw-model": {"score": 4.0, "scorePct": 40.0, "passed": 1},
    }
    check = ablation.falsification_check(summaries)
    assert check["rawMatchesOrBeatsSophiaFull"] is False


def test_parse_modes_always_includes_sophia_full() -> None:
    assert ablation.parse_modes("raw-model")[0] == "sophia-full"
    # ``all`` resolves to the canonical DEFAULT_MODE_ORDER, not every registered mode
    # (the opt-in W4 sophia-claim-router lever is excluded from the canonical set).
    assert set(ablation.parse_modes("all")) == set(ablation.DEFAULT_MODE_ORDER)
    # The opt-in lever is still selectable by name.
    assert "sophia-claim-router" in ablation.parse_modes("sophia-claim-router")


def main() -> int:
    test_seven_modes_present()
    test_raw_model_suppresses_all_scaffolding()
    test_raw_model_uses_neutral_system_prompt()
    test_raw_plus_tools_includes_tool_log_only_for_tool_cases()
    test_no_kb_drops_sources_but_keeps_council()
    test_no_council_drops_council_but_keeps_kb()
    test_no_gate_marks_gate_not_applied()
    test_no_memory_skips_learning_probe()
    test_sophia_full_runs_council_for_coding()
    test_compute_deltas_full_minus_mode()
    test_falsification_triggers_when_raw_wins()
    test_falsification_clear_when_sophia_wins()
    test_parse_modes_always_includes_sophia_full()
    print("test_ablation_runner: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
