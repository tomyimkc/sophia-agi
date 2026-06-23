#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.program_induction import compile_program_source, evaluate_program_induction, induce_program  # noqa: E402


def test_program_induction_promotes_affine() -> None:
    task = {"task_id": "add5", "examples": [{"input": i, "output": i + 5} for i in range(20)]}
    res = induce_program(task, seed=1)
    assert res.abstained is False
    assert res.admitted is not None
    assert res.test_stats is not None and res.test_stats.accuracy == 1.0
    assert res.admitted(10) == 15


def test_program_induction_holds_unlearnable() -> None:
    task = {"task_id": "noise", "examples": [{"input": str(i), "output": str((i * 17 + 3) % 11)} for i in range(20)]}
    res = induce_program(task, seed=2)
    assert res.abstained is True
    assert res.admitted is None


def test_program_induction_sandbox_rejects_unsafe() -> None:
    bad = [
        "import os\ndef solve(x):\n    return 1",
        "def solve(x):\n    return __import__('os').system('echo x')",
        "def solve(x):\n    return x.__class__",
        "def solve(x):\n    while True:\n        pass",
        "def solve(x):\n    return [0] * 1000000000",
        "def solve(x):\n    return 'x' * 1000000",
        "def not_solve(x):\n    return x",
    ]
    for src in bad:
        assert compile_program_source(src) is None
    good = compile_program_source("def solve(x):\n    return int(x) * 2 + 1")
    assert good is not None and good("4") == 9


def test_program_induction_demo_invariants() -> None:
    rep = evaluate_program_induction(seed=0)
    assert rep["ok"] is True
    assert rep["metrics"]["falsePromotionRateOnOod"] == 0.0
    assert rep["candidateOnly"] is True and rep["level3Evidence"] is False


def main() -> int:
    test_program_induction_promotes_affine()
    test_program_induction_holds_unlearnable()
    test_program_induction_sandbox_rejects_unsafe()
    test_program_induction_demo_invariants()
    print("test_program_induction: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
