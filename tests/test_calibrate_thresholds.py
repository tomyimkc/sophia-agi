"""Tests for tools/calibrate_thresholds.py (offline, deterministic, stdlib-only)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.calibrate_thresholds import _balanced_accuracy, _demo_records, calibrate  # noqa: E402


def test_separable_data_finds_perfect_boundary():
    result = calibrate(_demo_records())
    # Correct answers in [0.80,1.0], errors in [0.0,0.40] => a clean split exists.
    assert result["balancedAccuracy"] == 1.0
    assert 0.40 < result["bestHi"] <= 0.80
    assert result["n"] == len(_demo_records())


def test_lo_is_high_recall_floor():
    result = calibrate(_demo_records())
    # lo must not exceed the lowest correct-answer confidence (keep all correct surfaceable).
    assert result["lo"] <= 0.80
    assert result["lo"] <= result["bestHi"]


def test_balanced_accuracy_monotone_edges():
    recs = _demo_records()
    # hi=0 surfaces everything (TNR=0); hi=1 surfaces ~nothing (TPR=0) -> both ~0.5 balanced.
    assert _balanced_accuracy(recs, 0.0) <= 0.6
    assert _balanced_accuracy(recs, 1.0) <= 0.6


def test_empty_is_handled():
    assert calibrate([]).get("error") == "no records"


def _main():
    test_separable_data_finds_perfect_boundary()
    test_lo_is_high_recall_floor()
    test_balanced_accuracy_monotone_edges()
    test_empty_is_handled()
    print("test_calibrate_thresholds: OK")


if __name__ == "__main__":
    _main()
