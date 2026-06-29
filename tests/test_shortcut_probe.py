# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Representation-level shortcut probe (arXiv:2604.01476) — offline scaffold tests.

The probe is a construct INDEPENDENT of the syntactic scan: a linear direction in
rollout features that separates hack from clean, plus a GRPO advantage modification
that penalizes shortcut-like rollouts. These tests lock the offline behavior on
injected (synthetic) features; the live activation-capture piece is open.
"""

from __future__ import annotations

import pytest

from provenance_bench import shortcut_probe as sp


def test_offline_invariants_pass():
    ok, detail = sp.offline_invariants()
    assert ok, detail


def test_direction_separates_and_is_signed_toward_hack():
    feats, labs = sp._synthetic(seed=1)
    model = sp.fit(feats, labs)
    # hack centroid projects higher than clean centroid (direction points at hacks).
    assert model["proj_hack"] > model["proj_clean"]
    assert sp.auc(model, feats, labs) >= 0.9


def test_advantage_modification_penalizes_only_shortcut_rollouts():
    feats, labs = sp._synthetic(seed=2)
    model = sp.fit(feats[:80], labs[:80])
    adv = [1.0] * len(feats[80:])
    mod = sp.advantage_modification(adv, feats[80:], model, beta=1.0)
    # every modified advantage <= original (penalty is non-negative)...
    assert all(m <= a + 1e-9 for a, m in zip(adv, mod))
    # ...and hack rollouts lose more advantage than clean ones on average.
    drop = [a - m for a, m in zip(adv, mod)]
    dh = [x for x, y in zip(drop, labs[80:]) if y]
    dc = [x for x, y in zip(drop, labs[80:]) if not y]
    assert sum(dh) / len(dh) > sum(dc) / len(dc)


def test_fit_requires_both_classes():
    with pytest.raises(ValueError):
        sp.fit([[1.0, 2.0]], [True])
