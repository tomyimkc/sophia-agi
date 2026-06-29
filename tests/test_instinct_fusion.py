#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Falsifiable test of reflex fusion (instinct system, two-detector bus).

Pure stdlib + okf, deterministic, offline. Central checks: (U1) neither detector clears
the break-even bar alone but their fusion does; (U2) fusion follows the detection-theory
law d′_fused=(d_A+d_B)/√(2+2ρ) and the gain vanishes as detectors become redundant;
(U3) the okf detector separates exactly the errors self-consistency misses; (U4) fused
d′ exceeds either detector's; plus cross-process determinism (no hash-seed dependence).
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasoning.instinct_fusion import (  # noqa: E402
    _majority,
    _reflex_B,
    complementarity,
    gaussian_fusion_law,
    main,
    max_tolerable_rho,
    run_fusion,
)


def test_self_test_entrypoint():
    assert main(["--self-test"]) == 0


def test_u1_neither_clears_alone_but_fusion_does():
    f = run_fusion(seed=1234, bar=1.0)
    assert not f.A_clears and not f.B_clears
    assert f.fused_clears
    assert f.auc_fused < 0.99  # not a degenerate 'too clean' perfect separation


def test_u4_fused_beats_each_detector():
    f = run_fusion(seed=1234, bar=1.0)
    assert f.d_prime_fused > f.d_prime_A
    assert f.d_prime_fused > f.d_prime_B


def test_u2_fusion_law_matches_closed_form_and_degrades():
    law = [gaussian_fusion_law(0.96, 0.96, rho) for rho in (0.0, 0.3, 0.6, 0.9, 1.0)]
    assert all(abs(r["mc_dprime"] - r["closed_form"]) < 0.05 for r in law)
    assert law[0]["closed_form"] > law[-1]["closed_form"] + 0.3  # redundancy kills the gain
    # quadrature at rho=0: sqrt(2)*0.96 ≈ 1.358
    assert abs(law[0]["closed_form"] - math.sqrt(0.96 ** 2 + 0.96 ** 2)) < 0.01


def test_u3_okf_detector_is_complementary():
    comp = complementarity(seed=1234)
    assert comp["A_missed_errors"] > 0
    assert comp["B_auc_on_A_missed"] > 0.6  # B separates where A is blind


def test_detectors_are_independent_low_correlation():
    f = run_fusion(seed=1234, bar=1.0)
    assert abs(f.correlation) < 0.5


def test_max_tolerable_rho_monotone():
    bar = 1.0
    # stronger detectors tolerate more redundancy before fusion stops clearing
    assert max_tolerable_rho(1.2, 1.2, bar) > max_tolerable_rho(0.9, 0.9, bar)


def test_reflex_B_uses_grounding_not_answer_key():
    # over-inclusion of a still-grounded claim fires; a subset of truth (under-abstention) is blind
    true_set = frozenset({"primary_1", "mid_1", "leaf_1"})
    removed = {"primary_1"}
    assert _reflex_B(frozenset(true_set | {"independent_1"}), true_set, removed) == 1.0  # over -> fires
    assert _reflex_B(frozenset({"primary_1", "mid_1"}), true_set, removed) == 0.0  # under -> blind


def test_majority_is_hash_seed_independent():
    a = frozenset({"x", "y"})
    b = frozenset({"z"})
    samples = [a, a, b]  # 'a' is the clear majority
    assert _majority(samples) == a


def test_cross_process_determinism():
    """Two fresh interpreters with different hash seeds must produce identical output."""
    def once(seed_env: str) -> str:
        env = dict(os.environ, PYTHONHASHSEED=seed_env)
        return subprocess.run(
            [sys.executable, str(ROOT / "reasoning" / "instinct_fusion.py"), "--self-test"],
            capture_output=True, text=True, env=env, check=True,
        ).stdout
    assert once("0") == once("12345")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    print("all instinct_fusion tests passed")
