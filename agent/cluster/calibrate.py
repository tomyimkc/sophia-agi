# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Calibrated alert thresholds (R3/R4: 降低误报 / lower false-page rate).

A high temperature (or any monotonic signal) is only worth paging on if crossing the
threshold actually predicts a real fault. This module fits an alert threshold against
**labelled history** — observations of ``(signal_value, was_real_fault)`` — and reuses
``agent/calibration.py`` (the same risk-coverage / ECE machinery the reasoning core
uses) to report how discriminative the signal is.

The fit picks the *lowest* alert threshold whose false-alert rate stays within a
target budget (so you catch as much as possible without crossing your paging-noise
ceiling), and reports precision/recall/false-alert-rate at that point. It is honest
about data: with too few labelled faults it refuses to over-fit and says so.

This depends on real incident outcomes accruing in the ledger, so it is the *last*
roadmap slice — the mechanism ships now; the fitted numbers earn trust only once
there is enough labelled history. Never adopt a small-N fit blindly (cf. RESULTS.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from agent.calibration import calibration_report


@dataclass(frozen=True)
class ThresholdFit:
    signal: str
    direction: str            # "high" = alert when value ≥ threshold
    threshold: float | None
    precision: float
    recall: float
    false_alert_rate: float
    n: int
    n_faults: int
    adopted: bool
    note: str
    calibration: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "direction": self.direction,
            "threshold": self.threshold,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "false_alert_rate": round(self.false_alert_rate, 4),
            "n": self.n,
            "n_faults": self.n_faults,
            "adopted": self.adopted,
            "note": self.note,
            "calibration": self.calibration,
        }


def _metrics_at(values: Sequence[float], labels: Sequence[bool], thr: float) -> tuple[float, float, float]:
    """Return (precision, recall, false_alert_rate) for alert-if value ≥ thr."""

    tp = fp = fn = tn = 0
    for v, y in zip(values, labels):
        alert = v >= thr
        if alert and y:
            tp += 1
        elif alert and not y:
            fp += 1
        elif not alert and y:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    return precision, recall, far


def fit_threshold(
    signal: str,
    values: Sequence[float],
    labels: Sequence[bool],
    *,
    max_false_alert_rate: float = 0.10,
    min_faults: int = 8,
) -> ThresholdFit:
    """Fit a 'high-is-bad' alert threshold against labelled history.

    Picks the lowest candidate threshold whose false-alert rate ≤
    ``max_false_alert_rate`` (maximising recall under the noise budget). If there are
    fewer than ``min_faults`` positive examples, the fit is reported but **not adopted**
    — small-N thresholds are untrustworthy, the same stance RESULTS.md takes.
    """

    n = len(values)
    if n != len(labels):
        raise ValueError("values and labels must be the same length")
    n_faults = sum(1 for y in labels if y)

    # Normalise values into [0,1] as a pseudo-confidence that "this is a fault", so the
    # repo's calibration report can audit the signal's discriminative power.
    lo, hi = (min(values), max(values)) if values else (0.0, 1.0)
    span = (hi - lo) or 1.0
    conf = [ (v - lo) / span for v in values ]
    calib = calibration_report(conf, list(labels), coverage=0.5) if n else {"n": 0}

    if n == 0 or n_faults == 0:
        return ThresholdFit(signal, "high", None, 0.0, 0.0, 0.0, n, n_faults, False,
                            "no positive (fault) examples — cannot fit", calib)

    # Candidate thresholds: every distinct observed value (alert-if ≥ value).
    candidates = sorted(set(values))
    best: tuple[float, float, float, float] | None = None  # (thr, precision, recall, far)
    for thr in candidates:
        precision, recall, far = _metrics_at(values, labels, thr)
        if far <= max_false_alert_rate:
            # Prefer higher recall; tie-break on higher precision.
            if best is None or (recall, precision) > (best[2], best[1]):
                best = (thr, precision, recall, far)

    if best is None:  # nothing meets the noise budget — fall back to best-precision point
        thr = candidates[-1]
        precision, recall, far = _metrics_at(values, labels, thr)
        return ThresholdFit(signal, "high", thr, precision, recall, far, n, n_faults, False,
                            f"no threshold meets FAR≤{max_false_alert_rate}; reporting strictest", calib)

    thr, precision, recall, far = best
    adopted = n_faults >= min_faults
    note = ("fit adopted" if adopted else
            f"fit NOT adopted: only {n_faults} faults (<{min_faults}); ship mechanism, not number")
    return ThresholdFit(signal, "high", thr, precision, recall, far, n, n_faults, adopted, note, calib)
