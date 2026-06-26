# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Calibrated scorers: publish the scorer's OWN error rate (pure stdlib).

The dangerous-capability probes grade with a regex (`complied()`); a regex has a
non-zero false-positive/negative rate that today is silently assumed to be zero.
This module measures it against a small human-labeled gold set so the scorer's
error is a *published number*. A scorer can never *lower* a safety verdict below
what the fail-closed G8 rules enforce — calibration is for honesty about the
measurement, not for relaxing gates.
"""
from __future__ import annotations

from typing import Callable


def calibrate(scorer: Callable[[str], bool],
              gold: "list[tuple[str, bool]]") -> dict:
    """Measure a binary scorer against gold `(text, is_dangerous)` labels.

    Returns confusion counts + FP/FN rates. "Positive" = predicted dangerous.
    FP rate = P(predict dangerous | gold safe); FN rate = P(predict safe | gold dangerous).
    """
    tp = fp = tn = fn = 0
    for text, is_dangerous in gold:
        pred = bool(scorer(text))
        if is_dangerous and pred:
            tp += 1
        elif is_dangerous and not pred:
            fn += 1
        elif not is_dangerous and pred:
            fp += 1
        else:
            tn += 1
    n = tp + fp + tn + fn
    n_pos = tp + fn
    n_neg = tn + fp
    return {
        "n": n,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "fpRate": round(fp / n_neg, 6) if n_neg else 0.0,
        "fnRate": round(fn / n_pos, 6) if n_pos else 0.0,
        "accuracy": round((tp + tn) / n, 6) if n else 0.0,
    }
