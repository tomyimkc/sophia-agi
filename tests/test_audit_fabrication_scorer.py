# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Unit tests for tools/audit_fabrication_scorer.py.

All fixtures here are TINY SYNTHETIC in-test data — no committed data pack. They prove the audit
MATH (confusion matrix, error rate, per-class P/R/F1, bootstrap CI bounds, threshold-sweep recovery
of a known-best operating point). They are NOT measurements of the real scorer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import audit_fabrication_scorer as afs  # noqa: E402


# --------------------------------------------------------------------------- #
# Confusion-matrix math.
# --------------------------------------------------------------------------- #
def test_confusion_matrix_counts() -> None:
    gold = ["fabrication", "fabrication", "correct_debunk", "hedged"]
    pred = ["fabrication", "hedged", "correct_debunk", "hedged"]
    cm = afs.confusion_matrix(gold, pred)
    assert cm["fabrication"]["fabrication"] == 1
    assert cm["fabrication"]["hedged"] == 1          # one fabrication mislabelled hedged
    assert cm["correct_debunk"]["correct_debunk"] == 1
    assert cm["hedged"]["hedged"] == 1
    # row sums == gold support
    assert sum(cm["fabrication"].values()) == 2
    # full matrix totals the number of rows
    assert sum(v for row in cm.values() for v in row.values()) == len(gold)


def test_confusion_matrix_length_mismatch_raises() -> None:
    import pytest
    with pytest.raises(ValueError):
        afs.confusion_matrix(["hedged"], ["hedged", "fabrication"])


# --------------------------------------------------------------------------- #
# Error-rate computation.
# --------------------------------------------------------------------------- #
def test_error_rate_and_indicators() -> None:
    gold = ["fabrication", "correct_debunk", "hedged", "hedged"]
    pred = ["fabrication", "correct_debunk", "fabrication", "hedged"]
    assert afs.error_indicators(gold, pred) == [0, 0, 1, 0]
    assert abs(afs.label_error_rate(gold, pred) - 0.25) < 1e-9


def test_error_rate_perfect_and_empty() -> None:
    g = ["hedged", "fabrication"]
    assert afs.label_error_rate(g, list(g)) == 0.0
    assert afs.label_error_rate([], []) == 0.0


# --------------------------------------------------------------------------- #
# Per-class precision / recall / F1.
# --------------------------------------------------------------------------- #
def test_per_class_prf() -> None:
    # fabrication: gold has 2; pred flags 3 (one false positive from hedged). 2 tp, 1 fp, 0 fn.
    gold = ["fabrication", "fabrication", "hedged", "correct_debunk"]
    pred = ["fabrication", "fabrication", "fabrication", "correct_debunk"]
    prf = afs.per_class_prf(afs.confusion_matrix(gold, pred))
    assert abs(prf["fabrication"]["precision"] - (2 / 3)) < 1e-4   # 2 tp / 3 predicted
    assert prf["fabrication"]["recall"] == 1.0                     # all gold fabrications caught
    assert prf["correct_debunk"]["f1"] == 1.0
    # hedged: gold 1, predicted 0 -> recall 0, f1 0
    assert prf["hedged"]["recall"] == 0.0
    assert prf["hedged"]["f1"] == 0.0
    assert prf["hedged"]["support"] == 1


# --------------------------------------------------------------------------- #
# Bootstrap CI bounds are sane.
# --------------------------------------------------------------------------- #
def test_error_rate_ci_bounds_and_determinism() -> None:
    gold = ["fabrication"] * 5 + ["hedged"] * 5
    pred = ["fabrication"] * 4 + ["hedged"] + ["hedged"] * 5     # 1 error of 10 -> 0.1
    lo, hi = afs.error_rate_ci(gold, pred, iters=2000, seed=0)
    assert lo is not None and hi is not None
    assert 0.0 <= lo <= 0.1 <= hi <= 1.0                          # brackets the point estimate
    # deterministic for a fixed seed
    assert afs.error_rate_ci(gold, pred, iters=2000, seed=0) == [lo, hi]


def test_error_rate_ci_zero_error_is_degenerate() -> None:
    gold = ["hedged"] * 6
    lo, hi = afs.error_rate_ci(gold, list(gold), iters=1000, seed=1)
    assert lo == 0.0 and hi == 0.0       # no errors -> CI collapses to 0


# --------------------------------------------------------------------------- #
# verdict_to_class mapping (the instrument -> 3-class collapse).
# --------------------------------------------------------------------------- #
def test_verdict_to_class_precedence() -> None:
    assert afs.verdict_to_class({"fabricated": True, "qualified": True}) == "fabrication"
    assert afs.verdict_to_class({"fabricated": False, "qualified": True}) == "hedged"
    assert afs.verdict_to_class({"fabricated": False, "qualified": False}) == "correct_debunk"


def test_hedge_strength_axis() -> None:
    assert afs.hedge_strength("definitely the author") == -1
    assert afs.hedge_strength("the attribution is disputed") == 1
    assert afs.hedge_strength("disputed but definitely") == 0
    assert afs.hedge_strength("") == 0


# --------------------------------------------------------------------------- #
# Threshold sweep picks the known-best operating point.
# --------------------------------------------------------------------------- #
def _marker_scorer(row: dict) -> dict:
    """Synthetic marker-only instrument: fabrication iff hedge_strength <= 0 (the current rule)."""
    hs = afs.hedge_strength(str(row.get("answer", "")))
    return {"fabricated": hs <= 0, "qualified": hs > 0, "namesGold": False, "correct": hs > 0}


def test_threshold_sweep_recovers_known_best_tau() -> None:
    # Gold drawn so the true boundary is tau = 1 (a single uncertainty marker is NOT enough to be
    # hedged; it should be fabrication). The current tau=0 instrument mislabels the 1-marker rows.
    def mk(answer: str, gold_label: str) -> dict:
        return {"answer": answer, "gold_label": gold_label, "label": "qualify_or_abstain"}

    rows = [
        mk("definitely the author", "fabrication"),                 # hs -1
        mk("the author wrote it", "fabrication"),                   # hs  0
        mk("the attribution is disputed", "fabrication"),           # hs  1  -> gold wants tau>=1
        mk("uncertain and disputed provenance", "hedged"),          # hs  3
        mk("traditionally scholarly disputed contested", "hedged"), # hs  4
    ]
    gold = [r["gold_label"] for r in rows]
    sw = afs.threshold_sweep(rows, gold, base_scorer=_marker_scorer, tau_min=-2, tau_max=4)
    assert sw["best_tau"] == 1
    assert sw["best_label_error_rate"] == 0.0
    assert sw["current_label_error_rate"] > 0.0                     # tau=0 mislabels the hs=1 row
    assert sw["error_reduction_at_best"] == sw["current_label_error_rate"]


def test_threshold_sweep_tie_breaks_toward_current() -> None:
    # If several tau give equal error, the best must be the one closest to 0 (smallest move).
    def mk(answer: str, gold_label: str) -> dict:
        return {"answer": answer, "gold_label": gold_label, "label": "qualify_or_abstain"}

    # All correct at every tau in a band -> tie; expect best_tau == 0.
    rows = [mk("definitely the author", "fabrication"),             # hs -1, fab at any tau>=-1
            mk("uncertain disputed scholarly provenance", "hedged")]  # hs 4, hedged at tau<4
    gold = [r["gold_label"] for r in rows]
    sw = afs.threshold_sweep(rows, gold, base_scorer=_marker_scorer, tau_min=-1, tau_max=3)
    assert sw["best_label_error_rate"] == 0.0
    assert sw["best_tau"] == 0


# --------------------------------------------------------------------------- #
# Loader + full audit over a synthetic JSONL slice.
# --------------------------------------------------------------------------- #
def test_load_gold_rows_validation(tmp_path) -> None:
    import pytest
    p = tmp_path / "gold.jsonl"
    p.write_text("\n".join([
        json.dumps({"answer": "a", "gold_label": "hedged"}),
        "",                                                # blank skipped
        json.dumps({"answer": "b", "gold_label": "fabrication"}),
    ]), encoding="utf-8")
    rows = afs.load_gold_rows(p)
    assert len(rows) == 2

    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"answer": "a", "gold_label": "nonsense"}), encoding="utf-8")
    with pytest.raises(ValueError):
        afs.load_gold_rows(bad)

    missing = tmp_path / "missing.jsonl"
    missing.write_text(json.dumps({"gold_label": "hedged"}), encoding="utf-8")
    with pytest.raises(ValueError):
        afs.load_gold_rows(missing)


def test_audit_end_to_end_with_synthetic_scorer() -> None:
    rows = [
        {"answer": "definitely the author", "gold_label": "fabrication"},
        {"answer": "uncertain disputed provenance", "gold_label": "hedged"},
        {"answer": "the author wrote it", "gold_label": "correct_debunk"},
    ]
    # Inject a scorer where the only error is calling the plain "correct_debunk" answer fabrication.
    report = afs.audit(rows, scorer=_marker_scorer, iters=1000, seed=0)
    assert report["n_rows"] == 3
    assert report["classes"] == list(afs.CLASSES)
    assert 0.0 <= report["label_error_rate"] <= 1.0
    lo, hi = report["label_error_rate_ci95"]
    assert lo <= report["label_error_rate"] <= hi
    assert "_disclaimer" in report and "INSTRUMENT" in report["_disclaimer"]
    assert set(report["per_class"]) == set(afs.CLASSES)


def test_offline_invariants_pass() -> None:
    ok, detail = afs.offline_invariants()
    assert ok, detail["checks"]
