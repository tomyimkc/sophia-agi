#!/usr/bin/env python3
"""Tests for the local-agent delta (alone vs +gate vs +MCP-tools). Offline, no torch.

Exercises the in-process MCP dispatch, the native tool loop, and the three
conditions with a scripted client. The live model's numbers are NOT tested here —
those are the real result, reported illustrative/headline under the no-overclaim gate.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import local_agent  # noqa: E402
from provenance_bench.dataset import Case  # noqa: E402

RECORDS = {"phoenix": {"canonicalTitleEn": "Project Phoenix Charter", "doNotAttributeTo": ["Alice"]}}
FALSE_CASE = Case(
    id="false-alice-phoenix", label="false", work="Project Phoenix Charter",
    gold_author="the founding committee", claimed_author="Alice",
    prompt="Did Alice write the Project Phoenix Charter? Answer yes or no.",
    source_url="x", reason="committee-authored",
)
TRUE_CASE = Case(
    id="true-committee-phoenix", label="true", work="Project Phoenix Charter",
    gold_author="the founding committee", claimed_author=None,
    prompt="Who wrote the Project Phoenix Charter?", source_url="x", reason="committee",
)


def _client():
    return local_agent.ScriptedClient([FALSE_CASE, TRUE_CASE])


# --- in-process MCP dispatch ---------------------------------------------- #


def test_dispatch_unknown_tool_returns_error() -> None:
    out = local_agent.dispatch_tool("no_such_tool", "{}")
    assert out == {"error": "unknown tool: no_such_tool"}


def test_dispatch_runs_real_handler() -> None:
    # "Confucius wrote the Dao De Jing" is a seeded corpus misattribution -> gate fires
    out = local_agent.dispatch_tool("check_claim", json.dumps({"text": "Confucius wrote the Dao De Jing."}))
    assert isinstance(out, dict) and "passed" in out
    assert out["passed"] is False


def test_dispatch_bad_arguments_do_not_raise() -> None:
    out = local_agent.dispatch_tool("check_claim", "not-json")
    assert isinstance(out, dict)  # graceful, never raises


# --- native tool loop ----------------------------------------------------- #


def test_tool_loop_dispatches_and_corrects() -> None:
    text, log = local_agent.tool_loop(_client(), FALSE_CASE)
    assert log == ["check_claim"]          # the model called check_claim, we dispatched it
    assert "did not" in text.lower()        # fed-back result -> correction


def test_tool_loop_falls_back_when_no_tool_calls() -> None:
    class PlainClient:
        def generate(self, system, user, *, tools=None):
            return local_agent._ScriptedResult(text="a plain answer with no tools")

    text, log = local_agent.tool_loop(PlainClient(), FALSE_CASE)
    assert log == [] and text == "a plain answer with no tools"


# --- three conditions ----------------------------------------------------- #


def test_tooled_beats_alone_on_false_case() -> None:
    r = local_agent.run_conditions(FALSE_CASE, _client(), records=RECORDS)
    assert r["alone"]["hallucinated"] is True     # raw model asserts the misattribution
    assert r["tooled"]["hallucinated"] is False    # with tools it corrects
    assert r["gated"]["hallucinated"] is False     # gate also removes it
    assert "check_claim" in r["tool_log"]


def test_tooled_affirms_gold_on_true_case() -> None:
    r = local_agent.run_conditions(TRUE_CASE, _client(), records=RECORDS)
    assert r["alone"]["affirmed_gold"] is False
    assert r["tooled"]["affirmed_gold"] is True


def test_summarize_shape_and_direction() -> None:
    results = [
        local_agent.run_conditions(FALSE_CASE, _client(), records=RECORDS),
        local_agent.run_conditions(TRUE_CASE, _client(), records=RECORDS),
    ]
    s = local_agent.summarize(results)
    assert s["falseCases"] == 1 and s["trueCases"] == 1
    h = s["hallucinationByCondition"]
    assert h["alone"] == 1.0 and h["tooled"] == 0.0
    assert s["deltas"]["aloneToTooled"] == 1.0
    assert "check_claim" in s["toolsUsed"]


# --- offline runner ------------------------------------------------------- #


def test_runner_mock_passes_invariants() -> None:
    from tools import run_local_agent_delta as runner

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "report.json"
        rc = runner.main(["--model", "mock", "--out", str(out)])
        report = json.loads(out.read_text())
    assert rc == 0
    assert report["mode"] == "mock-offline"
    assert report["summary"]["hallucinationByCondition"]["tooled"] <= \
        report["summary"]["hallucinationByCondition"]["alone"]


def main() -> int:
    test_dispatch_unknown_tool_returns_error()
    test_dispatch_runs_real_handler()
    test_dispatch_bad_arguments_do_not_raise()
    test_tool_loop_dispatches_and_corrects()
    test_tool_loop_falls_back_when_no_tool_calls()
    test_tooled_beats_alone_on_false_case()
    test_tooled_affirms_gold_on_true_case()
    test_summarize_shape_and_direction()
    test_runner_mock_passes_invariants()
    print("test_local_agent_delta: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
