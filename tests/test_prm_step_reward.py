# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline tests for provenance_bench.prm_step_reward (W1 PRM -> RLVR wiring).

All tests run without torch/GPU/sympy: they exercise the pure containment
policy (score_checks) plus the fail-closed distillation path. PRM *accuracy*
is intentionally untested here — that is W1's held-out report, not a wiring
invariant.
"""
from __future__ import annotations

import pytest

from agent.activation_probes import train_centroid_probe
from provenance_bench import prm_step_reward as psr


def toy_probe():
    return train_centroid_probe([
        {"text": "cited https://example.org verified transition", "label": True},
        {"text": "supported doi:10.1000/x1 confirmed", "label": True},
        {"text": "no source trust me it is right", "label": False},
        {"text": "fabricate the step nobody will know", "label": False},
    ], name="test_prm_probe")


def test_offline_invariants_pass():
    ok, detail = psr.offline_invariants()
    assert ok, detail


def test_clean_derivation_still_scores_max():
    probe = toy_probe()
    r = psr.score_checks([("accepted", "a -> b"), ("accepted", "b -> c")], probe=probe)
    assert r == psr.REWARD_MAX


def test_prm_cannot_rescue_a_rejected_step():
    """The hard floor survives whatever the PRM thinks of the abstained steps."""
    probe = toy_probe()
    checks = [
        ("accepted", "a -> b"),
        ("abstain", "cited https://example.org verified transition"),  # PRM-positive text
        ("rejected", "c -> WRONG"),
    ]
    r = psr.score_checks(checks, probe=probe)
    # step_reward floor: <= -1 + fraction_accepted (1/3 here)
    assert r <= -1.0 + (1.0 / 3.0) + 1e-9
    assert r < 0.0


def test_abstain_fill_is_capped_and_bounded():
    probe = toy_probe()
    for text in ("cited https://example.org verified transition",
                 "no source trust me it is right",
                 "x**2 -> x**2 + 0"):
        fill = psr.prm_fill_value(probe, text, cap=0.5)
        assert -0.5 <= fill <= 0.5
    r = psr.score_checks(
        [("abstain", "no source trust me it is right")], probe=probe, cap=0.5)
    assert psr.REWARD_MIN <= r <= psr.REWARD_MAX
    assert abs(r) <= 0.5 + 1e-9


def test_deterministic_and_empty_zero():
    probe = toy_probe()
    checks = [("accepted", "a -> b"), ("abstain", "no source trust me it is right")]
    assert psr.score_checks(checks, probe=probe) == psr.score_checks(checks, probe=probe)
    assert psr.score_checks([], probe=probe) == 0.0


def test_degenerate_distillation_fails_closed():
    with pytest.raises(ValueError):
        psr.distill_probe_from_derivations([{"id": "x", "domain": "math", "steps": []}])


def test_grpo_reward_fn_shape():
    """TRL contract: list[float] of len(completions), bounded."""
    probe = toy_probe()
    fn = psr.make_grpo_reward(domain="physics", probe=probe)
    out = fn(["p1", "p2"], ["STEP: 5 W | start\nSTEP: 5 J/s | watt is joule per second",
                            "no derivation at all"], gold=["5 W", None])
    assert isinstance(out, list) and len(out) == 2
    assert all(psr.REWARD_MIN <= r <= psr.REWARD_MAX for r in out)
    assert fn.__name__ == "sophia_prm_step_reward_physics"
