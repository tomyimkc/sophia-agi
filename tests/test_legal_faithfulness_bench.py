#!/usr/bin/env python3
"""Tests for the gated semantic-faithfulness benchmark runner (offline).

Uses deterministic SCRIPTED judges (not real models) to verify the aggregation +
no-overclaim gate logic both ways: a clean multi-family run validates; mock /
single-family / low-kappa runs do not. This tests the gate, NOT semantic accuracy.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.legal_faithfulness import Verdict, register_holdings  # noqa: E402


def _load_runner():
    spec = importlib.util.spec_from_file_location("rlfb", ROOT / "tools" / "run_legal_faithfulness_bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cases():
    return json.loads((ROOT / "benchmark" / "legal_holding_faithful.json").read_text("utf-8"))["cases"]


def _perfect_judge(cases):
    answer = {c["proposition"]: c["expectFaithful"] for c in cases}

    def judge(proposition, holding):
        return Verdict(supports=bool(answer.get(proposition)), abstained=False, reason="scripted")

    return judge


def _always_support_judge(proposition, holding):
    return Verdict(supports=True, abstained=False, reason="scripted-yes")


def test_clean_multifamily_run_validates() -> None:
    m = _load_runner()
    cases = _cases()
    holdings = register_holdings()
    judges = [_perfect_judge(cases), _perfect_judge(cases)]
    runs = [m.run_once(cases, holdings, judges) for _ in range(3)]
    result = m.aggregate(runs, judge_specs=["anthropic:stubA", "deepseek:stubB"])
    assert result["consensusAccuracy"] == 1.0
    assert result["meanPairwiseKappa"] == 1.0
    assert result["validated"] is True
    assert all(result["validatedChecks"].values())


def test_mock_never_validates() -> None:
    m = _load_runner()
    cases = _cases()
    holdings = register_holdings()
    judges = [_perfect_judge(cases)]
    runs = [m.run_once(cases, holdings, judges) for _ in range(3)]
    result = m.aggregate(runs, judge_specs=["mock"])
    assert result["validated"] is False
    assert result["validatedChecks"]["notMock"] is False
    assert result["validatedChecks"]["multiFamilyJudges"] is False


def test_single_family_does_not_validate() -> None:
    m = _load_runner()
    cases = _cases()
    holdings = register_holdings()
    judges = [_perfect_judge(cases), _perfect_judge(cases)]
    runs = [m.run_once(cases, holdings, judges) for _ in range(3)]
    # both judges same family -> multiFamilyJudges fails
    result = m.aggregate(runs, judge_specs=["anthropic:a", "anthropic:b"])
    assert result["validatedChecks"]["multiFamilyJudges"] is False
    assert result["validated"] is False


def test_disagreement_lowers_kappa_and_blocks_validation() -> None:
    from provenance_bench.aggregate import KAPPA_FLOOR

    m = _load_runner()
    cases = _cases()
    holdings = register_holdings()
    judges = [_perfect_judge(cases), _always_support_judge]
    runs = [m.run_once(cases, holdings, judges) for _ in range(3)]
    result = m.aggregate(runs, judge_specs=["anthropic:a", "deepseek:b"])
    # perfect judge vs always-yes judge: they disagree on every misstated case, so
    # inter-judge agreement collapses below the floor and the gate refuses to validate.
    assert result["meanPairwiseKappa"] < KAPPA_FLOOR
    assert result["validatedChecks"]["kappaAboveFloor"] is False
    assert result["validated"] is False


def test_abstention_counts_against_accuracy() -> None:
    m = _load_runner()
    cases = _cases()
    holdings = register_holdings()

    def abstainer(_p, _h):
        return Verdict(abstained=True, reason="unsure")

    runs = [m.run_once(cases, holdings, [abstainer]) for _ in range(1)]
    result = m.aggregate(runs, judge_specs=["anthropic:a"])
    # all abstain -> consensus defaults to "not-supports" -> only the misstated (false) cases are right
    expected = sum(1 for c in cases if not c["expectFaithful"]) / len(cases)
    assert result["consensusAccuracy"] == round(expected, 4)


def main() -> int:
    import inspect

    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_legal_faithfulness_bench: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
