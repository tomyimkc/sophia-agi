# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Frontier eval M0 — harness/banks/scoring/rubric tests (plain-script, stdlib)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.frontier import banks, rubric, scoring  # noqa: E402
from eval.frontier.harness import Condition, EvalTask, run_paired  # noqa: E402


def _danger_scorer(text: str) -> bool:
    return "comply" in text.lower()


def test_paired_demonstrates_positive_delta() -> None:
    task = EvalTask("t", ("p1", "p2", "p3", "p4"), _danger_scorer)
    control = Condition("safe", lambda p: "I refuse")
    treatment = Condition("weak", lambda p: "I comply")
    r = run_paired(task, control, treatment, seed=0)
    assert r.control_score == 0.0 and r.treatment_score == 1.0
    assert r.delta == 1.0 and r.ci_lo > 0.0 and r.demonstrated is True
    assert r.measured is True


def test_no_effect_is_not_demonstrated() -> None:
    task = EvalTask("t", ("p1", "p2", "p3", "p4"), _danger_scorer)
    same = Condition("c", lambda p: "I refuse")
    r = run_paired(task, same, same, seed=0)
    assert r.delta == 0.0 and r.demonstrated is False


def test_scorer_failure_is_fail_closed() -> None:
    def _raise(_c: str) -> bool:
        raise RuntimeError("boom")
    task = EvalTask("t", ("p1", "p2"), _raise)
    r = run_paired(task, Condition("a", lambda p: "x"), Condition("b", lambda p: "y"), seed=0)
    assert r.measured is False and r.demonstrated is False  # unmeasured, never passing 0.0


def test_bank_split_disjoint_and_deterministic() -> None:
    prompts = tuple(f"prompt-{i}" for i in range(8))
    train, test = banks.split_bank(prompts, test_frac=0.5, seed=1)
    assert set(train).isdisjoint(set(test))
    assert len(train) + len(test) == 8 and len(test) == 4
    assert banks.split_bank(prompts, test_frac=0.5, seed=1) == (train, test)  # deterministic
    # Hash is order-independent and stable.
    assert banks.bank_hash(prompts) == banks.bank_hash(tuple(reversed(prompts)))
    assert len(banks.bank_hash(prompts)) == 16


def test_scorer_calibration_reports_fp_fn() -> None:
    gold = [("comply now", True), ("I refuse", False), ("comply please", True),
            ("never", False)]
    cal = scoring.calibrate(_danger_scorer, gold)
    assert cal["n"] == 4 and cal["fpRate"] == 0.0 and cal["fnRate"] == 0.0
    assert cal["accuracy"] == 1.0
    # A blind always-True scorer has FP on every negative.
    cal2 = scoring.calibrate(lambda t: True, gold)
    assert cal2["fpRate"] == 1.0 and cal2["fnRate"] == 0.0


def test_rubric_validation_and_kappa() -> None:
    good = {"name": "r", "version": 1, "criteria": [{"id": "c1", "desc": "x"}]}
    assert rubric.validate_rubric(good) is good
    for bad in ({"name": "r"}, {"name": "r", "version": 1, "criteria": []}):
        try:
            rubric.validate_rubric(bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    k = rubric.inter_rater_kappa([1, 0, 1, 0, 1], [1, 0, 1, 0, 1])
    assert k is not None and k > 0.99  # perfect agreement


def test_run_frontier_eval_offline_invariants() -> None:
    import importlib
    mod = importlib.import_module("tools.run_frontier_eval")
    ok, detail, scores = mod._offline_invariants(seed=0)
    assert ok, [k for k, v in detail["checks"].items() if not v]
    assert scores[0].demonstrated is True and scores[0].measured is True


def _run() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"PASSED {len(tests)} frontier-harness tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
