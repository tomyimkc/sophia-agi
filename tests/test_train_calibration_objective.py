# SPDX-License-Identifier: Apache-2.0
"""Tests for W2 proper-scoring calibration objective. Binds to the REAL agent.calibration
and agent.abstention_scoring (pure-Python, no heavy deps), so this exercises real interfaces."""
import importlib.util
import random
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "train_calibration_objective.py"


def _load():
    sys.modules.pop("w2tool", None)
    spec = importlib.util.spec_from_file_location("w2tool", TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _overconfident(n=200, seed=0):
    rng = random.Random(seed)
    recs = []
    for _ in range(n):
        recs.append({"confidence": round(rng.uniform(0.8, 0.98), 3),
                     "correct": rng.random() < 0.55, "action": "answer"})
    return recs


def test_brier_loss_is_proper():
    m = _load()
    # a proper scoring rule is minimized by reporting the true probability
    assert m.brier_loss(1.0, 1) == 0.0
    assert m.brier_loss(0.0, 0) == 0.0
    assert m.brier_loss(0.5, 1) == 0.25


def test_log_loss_penalizes_confident_wrong():
    m = _load()
    # confident-wrong must cost more than uncertain
    assert m.log_loss(0.99, 0) > m.log_loss(0.6, 0)


def test_abstention_penalty_is_asymmetric():
    m = _load()
    # confident wrong answer penalized; abstention free; correct free
    assert m.asymmetric_abstention_penalty(0.9, 0, "answer", lam=1.0) == 0.9
    assert m.asymmetric_abstention_penalty(0.9, 0, "abstain", lam=1.0) == 0.0
    assert m.asymmetric_abstention_penalty(0.9, 1, "answer", lam=1.0) == 0.0


def test_calibration_improves_ece_on_overconfident_data():
    """The core thesis: minimizing the proper-scoring objective lowers ECE, measured by
    the repo's OWN expected_calibration_error."""
    m = _load()
    if not m._REPO_OK:
        import pytest
        pytest.skip("agent.calibration unavailable")
    r = m.run(_overconfident(), loss="brier", lam=1.0)
    assert r["ok"] is True
    assert r["ece"]["after"] <= r["ece"]["before"]  # improved or equal, never worse
    assert r["ece"]["improved"] is True


def test_empty_records_fail_closed():
    m = _load()
    if not m._REPO_OK:
        import pytest
        pytest.skip("agent.calibration unavailable")
    assert m.run([])["ok"] is False


def test_all_abstain_fails_closed():
    m = _load()
    if not m._REPO_OK:
        import pytest
        pytest.skip("agent.calibration unavailable")
    recs = [{"confidence": 0.0, "correct": False, "action": "abstain"} for _ in range(10)]
    r = m.run(recs)
    assert r["ok"] is False and "abstention" in r["reason"]


def test_unknown_loss_fails_closed():
    m = _load()
    if not m._REPO_OK:
        import pytest
        pytest.skip("agent.calibration unavailable")
    assert m.run(_overconfident(), loss="nonsense")["ok"] is False