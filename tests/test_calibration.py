#!/usr/bin/env python3
"""Tests for calibrated abstention — competence where no verifier exists.

The falsifiable property is "knows what it doesn't know": answering only the
most-confident fraction has lower error than answering everything (selective risk
< base risk), and confidence is calibrated (ECE small when it should be, large
when overconfident).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import calibration as cal  # noqa: E402


def test_ece_zero_when_perfectly_confident_and_correct() -> None:
    confs = [1.0] * 10
    correct = [True] * 10
    assert cal.expected_calibration_error(confs, correct) == 0.0
    # confident-and-wrong is maximally miscalibrated
    assert cal.expected_calibration_error([1.0] * 10, [False] * 10) == 1.0


def test_ece_detects_overconfidence() -> None:
    # claims 100% sure but right only half the time → ECE ~0.5
    confs = [1.0] * 100
    correct = [i % 2 == 0 for i in range(100)]
    assert cal.expected_calibration_error(confs, correct) >= 0.45


def test_base_and_selective_risk() -> None:
    # confidence ranks correctness: top half all correct, bottom half all wrong.
    confs = [0.9] * 50 + [0.1] * 50
    correct = [True] * 50 + [False] * 50
    assert cal.base_risk(correct) == 0.5
    assert cal.selective_risk(confs, correct, 0.5) == 0.0   # answer only the confident half
    assert cal.selective_risk(confs, correct, 1.0) == 0.5   # full coverage = base risk


def test_risk_coverage_full_equals_base() -> None:
    confs = [0.9, 0.8, 0.2, 0.1]
    correct = [True, False, True, False]
    curve = cal.risk_coverage_curve(confs, correct)
    assert curve[-1]["coverage"] == 1.0
    assert curve[-1]["risk"] == cal.base_risk(correct)


def test_self_consistency_majority_and_confidence() -> None:
    ans, conf = cal.self_consistency(["7", "7", "7", "9"])
    assert ans == "7" and conf == 0.75
    assert cal.self_consistency([]) == (None, 0.0)


def test_calibration_report_flags_selective_win() -> None:
    confs = [0.95] * 40 + [0.3] * 60
    correct = [True] * 38 + [False] * 2 + [False] * 40 + [True] * 20
    rep = cal.calibration_report(confs, correct, coverage=0.4)
    assert rep["selectiveBeatsBase"] is True
    assert rep["selectiveRisk"] < rep["baseRisk"]
    assert set(rep) >= {"n", "ece", "aurc", "baseRisk", "selectiveRisk", "selectiveBeatsBase"}


def main() -> int:
    test_ece_zero_when_perfectly_confident_and_correct()
    test_ece_detects_overconfidence()
    test_base_and_selective_risk()
    test_risk_coverage_full_equals_base()
    test_self_consistency_majority_and_confidence()
    test_calibration_report_flags_selective_win()
    print("test_calibration: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
