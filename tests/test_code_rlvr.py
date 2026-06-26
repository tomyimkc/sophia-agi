# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for the code RLVR pack (no torch, no GPU).

The correctness checks require the hidden-test executor to run (it only executes
the small, trusted reference snippets built here). We opt into execution for this
test; the structural (deterministic / bounded / contamination-free) checks hold
either way.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The executor only ever runs the small trusted snippets constructed in this file
# (correct/wrong scale() implementations), so opting in here is safe.
os.environ.setdefault("SOPHIA_ALLOW_CODE_EXEC", "1")

from provenance_bench import code_dataset, code_reward  # noqa: E402


def test_family_disjoint_split() -> None:
    data = code_dataset.build_code_rl_dataset(eval_frac=0.34, seed=0)
    assert data["family_intersection"] == []          # contamination-free
    assert data["train_rows"] and data["eval_rows"]
    train_f = {t["family"] for t in data["train_tasks"]}
    eval_f = {t["family"] for t in data["eval_tasks"]}
    assert train_f.isdisjoint(eval_f)
    # rows carry the hidden test column (the reward oracle), NOT a gold solution
    assert all("test" in r and "solution" not in r for r in data["train_rows"])


def test_reward_correct_passes_wrong_fails() -> None:
    test_code = "assert scale(3, 4) == 12\n"
    good = "```python\ndef scale(n, k):\n    return n * k\n```"
    bad = "```python\ndef scale(n, k):\n    return n + k\n```"
    if code_reward.exec_enabled():
        good_score, good_detail = code_reward.reward_for_task(good, test_code)
        bad_score, _ = code_reward.reward_for_task(bad, test_code)
        assert good_score == code_reward.REWARD_MAX and good_detail["passed"]
        assert bad_score == code_reward.REWARD_MIN
    else:
        # syntax-only: both compile, so both "pass" — correctness undecided (fail-closed note)
        score, detail = code_reward.reward_for_task(good, test_code)
        assert score == code_reward.REWARD_MAX and detail["executed"] is False


def test_grpo_reward_shape() -> None:
    rf = code_reward.make_grpo_reward()
    good = "```python\ndef scale(n, k):\n    return n * k\n```"
    out = rf(["p1", "p2"], [good, "no code here"], test=["assert scale(2,3)==6\n", "assert scale(2,3)==6\n"])
    assert isinstance(out, list) and len(out) == 2
    assert all(code_reward.REWARD_MIN <= r <= code_reward.REWARD_MAX for r in out)


def test_offline_invariants() -> None:
    ok, detail = code_reward.offline_invariants()
    assert ok, detail["checks"]
    assert detail["familyIntersection"] == []


def test_run_rlvr_mock_code_passes() -> None:
    """The CLI mock/dry-run path for the code task writes a passing invariants report."""
    import tempfile

    from tools import run_rlvr

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "rlvr-code.json"
        code = run_rlvr.main(["--model", "mock", "--task", "code", "--dry-run", "--out", str(out)])
        assert code == 0
        import json

        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["task"] == "code"
        assert report["mode"] == "mock-offline"
        assert all(report["checks"].values()), report["checks"]


if __name__ == "__main__":
    for name in list(globals()):
        if name.startswith("test_"):
            globals()[name]()
    print("code RLVR offline invariants PASS")
