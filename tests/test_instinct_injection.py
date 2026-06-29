#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Falsifiable test of the injection (edit-in-place) model.

Pure stdlib, deterministic, offline. Checks: I1 injection cheaper than re-route; I2 it can
dominate re-route at a good strength; I3 brittleness roofline (interior optimum, over-steer
penalty, MC≈closed form); I4 the inject→reroute hybrid beats both.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasoning.instinct_injection import (  # noqa: E402
    best_strength,
    hybrid_correct,
    inject_correct,
    main,
    p_corrupt,
    p_flip,
    run_experiment,
)


def test_self_test_entrypoint():
    assert main(["--self-test"]) == 0


def test_i1_injection_cheaper_than_reroute():
    p = run_experiment()["policies"]
    assert p["inject"]["cost"] < p["reroute"]["cost"]


def test_i2_injection_can_dominate_reroute():
    p = run_experiment()["policies"]
    assert p["inject"]["correct"] > p["reroute"]["correct"]
    assert p["inject"]["cost"] < p["reroute"]["cost"]  # dominance = better on both axes


def test_i3_brittleness_roofline():
    s_star, v_star = best_strength()
    assert 0.0 < s_star < 1.0                       # interior optimum
    assert inject_correct(1.0) < v_star - 0.1       # over-steering hurts
    assert inject_correct(0.0) == 0.0               # no steer, no flip
    # monotone helpers
    assert p_flip(0.2) < p_flip(0.8)
    assert p_corrupt(0.2) < p_corrupt(0.8)


def test_i3_mc_matches_closed_form():
    assert run_experiment()["mc_vs_closed_form_abs_err"] < 0.01


def test_i4_hybrid_beats_both():
    res = run_experiment()
    p = res["policies"]
    assert p["hybrid"]["correct"] >= p["inject"]["correct"] - 1e-9
    assert p["hybrid"]["correct"] > p["reroute"]["correct"]


def test_hybrid_monotone_in_inject_quality():
    # a better single-shot inject yields a better-or-equal hybrid
    assert hybrid_correct(0.5) >= hybrid_correct(0.05)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    print("all instinct_injection tests passed")
