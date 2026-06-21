#!/usr/bin/env python3
"""Tests for the code-execution verifier, code reward, code route, and uplift runner.

Offline (executes tiny snippets in a sandbox — the whole point). Confirms the
interpreter-as-verifier wiring end to end.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import code_exec, code_reward  # noqa: E402

GOOD = "```python\ndef add(a, b):\n    return a + b\n```"
BAD = "```python\ndef add(a, b):\n    return a - b\n```"
SYNTAX_ERR = "```python\ndef add(a, b)\n    return a + b\n```"
TEST = "assert add(2, 3) == 5\nassert add(0, 0) == 0\nassert add(-1, 1) == 0\n"


def test_extract_code_fenced_and_raw() -> None:
    assert "def add" in code_exec.extract_code(GOOD)
    assert "def add" in code_exec.extract_code("def add(a, b):\n    return a + b")
    assert code_exec.extract_code("just prose, no code here") == ""


def test_run_solution_pass_fail() -> None:
    assert code_exec.check_answer(GOOD, TEST)["passed"] is True
    r = code_exec.check_answer(BAD, TEST)
    assert r["passed"] is False and r["executed"] is True


def test_no_code_fails() -> None:
    assert code_exec.check_answer("I cannot help with that.", TEST)["passed"] is False


def test_timeout_guarded() -> None:
    loop = "```python\ndef add(a,b):\n    while True: pass\n```"
    r = code_exec.check_answer(loop, TEST, timeout_sec=2)
    assert r["passed"] is False


# --- reward ---------------------------------------------------------------- #


def test_code_reward_signal() -> None:
    good_r, gd = code_reward.reward_for_task(GOOD, TEST)
    bad_r, bd = code_reward.reward_for_task(BAD, TEST)
    assert good_r == code_reward.REWARD_MAX and gd["passed"] is True
    assert bad_r == code_reward.REWARD_MIN and bd["passed"] is False


def test_grpo_code_reward_routes_by_test_column() -> None:
    fn = code_reward.make_grpo_reward()
    rewards = fn(prompts=["p", "p"], completions=[GOOD, BAD], test=[TEST, TEST])
    assert rewards == [code_reward.REWARD_MAX, code_reward.REWARD_MIN]


# --- claim router code route ----------------------------------------------- #


def test_router_flags_syntax_error_passes_valid() -> None:
    from agent.claim_router import route_and_check

    assert route_and_check(GOOD)["passed"] is True
    bad = route_and_check(SYNTAX_ERR)
    assert not bad["passed"]
    assert any(c["type"] == "code" and not c["passed"] for c in bad["perClaim"])


# --- benchmark dataset integrity ------------------------------------------- #


def test_benchmark_dataset_shape() -> None:
    data = json.loads((ROOT / "benchmark" / "code_tasks.json").read_text())
    tasks = data["tasks"]
    assert len(tasks) >= 20
    for t in tasks:
        assert {"id", "entry_point", "prompt", "test"} <= set(t)
        assert "assert" in t["test"]


def test_uplift_runner_mock() -> None:
    from tools import run_code_uplift as r

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "r.json"
        rc = r.main(["--model", "mock", "--limit", "3", "--max-repairs", "1", "--out", str(out)])
        report = json.loads(out.read_text())
    assert rc == 0
    assert report["benchmark"] == "code-uplift"
    assert "alonePass1" in report and "sophiaPass1" in report


def main() -> int:
    test_extract_code_fenced_and_raw()
    test_run_solution_pass_fail()
    test_no_code_fails()
    test_timeout_guarded()
    test_code_reward_signal()
    test_grpo_code_reward_routes_by_test_column()
    test_router_flags_syntax_error_passes_valid()
    test_benchmark_dataset_shape()
    test_uplift_runner_mock()
    print("test_code_uplift: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
