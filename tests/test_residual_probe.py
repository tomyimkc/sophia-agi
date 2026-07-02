#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic stdlib tests for the residual-probe RANKING/CALIBRATION metrics (auroc/ece)
and their use with the existing vector-probe fitter. No torch, no model — runs in CI. The
residual FEATURIZER itself is validated separately on the Spark via tools/residual_probe_eval.py.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.activation_probes import auroc, ece, train_vector_probe  # noqa: E402


def _synthetic_rows(n=30, dim=32, sep=1.5, seed=0):
    rng = random.Random(seed)
    rows, vecs = [], {}
    for i in range(n):
        rows.append({"id": f"p{i}", "text": f"pos{i}", "label": True})
        vecs[f"pos{i}"] = [rng.gauss(0, 1) + (sep if k == 0 else 0.0) for k in range(dim)]
        rows.append({"id": f"n{i}", "text": f"neg{i}", "label": False})
        vecs[f"neg{i}"] = [rng.gauss(0, 1) - (sep if k == 0 else 0.0) for k in range(dim)]
    return rows, vecs


def test_vector_probe_ranks_separable_set_high_auroc():
    rows, vecs = _synthetic_rows()
    feat = lambda t: vecs[t]  # noqa: E731
    train, test = rows[:30], rows[30:]
    probe = train_vector_probe(train, feat, name="residual_truth")
    scores = [probe.score_vector(feat(r["text"])) for r in test]
    labels = [bool(r["label"]) for r in test]
    a = auroc(scores, labels)
    assert a > 0.9, f"diff-in-means over a separable set should rank well (AUROC={a})"
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_auroc_perfect_reversed_and_degenerate():
    assert auroc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert auroc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [False, True, False, True]) == 0.5  # all ties -> chance
    assert math.isnan(auroc([0.1, 0.2], [True, True]))  # one class -> undefined


def test_ece_zero_when_calibrated_positive_when_not():
    scores = [0.9] * 10 + [0.1] * 10
    labels = [True] * 9 + [False] * 1 + [True] * 1 + [False] * 9
    assert ece(scores, labels, bins=10) < 0.02
    assert ece([0.99] * 10, [True] * 5 + [False] * 5, bins=10) > 0.4  # overconfident


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
