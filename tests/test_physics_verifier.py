# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Unit tests for the physics verifier substrate: agent/units.py,
agent.verifiers.physics_equivalent / physics_sound, agent/physics_verifier.py,
and provenance_bench physics_reward / physics_dataset.

All stdlib-only and deterministic — no GPU, no optional backend.
"""
from __future__ import annotations

from agent import physics_verifier as pv
from agent import units
from agent.verifiers import physics_equivalent, physics_sound
from provenance_bench import physics_dataset, physics_reward


# --------------------------------------------------------------------------- #
# Units engine
# --------------------------------------------------------------------------- #
def test_base_and_derived_units_share_dimension() -> None:
    # N (newton) must reduce to kg*m/s^2.
    _, _, dn = units.parse_quantity("1 N")
    _, _, dk = units.parse_quantity("1 kg*m/s^2")
    assert units.same_dim(dn, dk)


def test_energy_is_not_acceleration() -> None:
    _, _, dj = units.parse_quantity("9.8 J")
    _, _, da = units.parse_quantity("9.8 m/s^2")
    assert not units.same_dim(dj, da)


def test_si_prefixes() -> None:
    ok, val, _ = units.parse_quantity("2 km")
    assert ok and abs(val - 2000.0) < 1e-9
    ok, val, _ = units.parse_quantity("5 ms")  # milliseconds, not meter-seconds
    assert ok and abs(val - 0.005) < 1e-12
    ok, val, dim = units.parse_quantity("3 mg")  # milligram
    assert ok and abs(val - 3e-6) < 1e-18 and units.same_dim(dim, units._d(kg=1))


def test_scientific_notation_and_compound() -> None:
    ok, val, dim = units.parse_quantity("3.0 × 10^8 m/s")
    assert ok and abs(val - 3.0e8) < 1.0
    assert units.same_dim(dim, units._d(m=1, s=-1))


def test_unknown_unit_fails_closed() -> None:
    ok, _, _ = units.parse_quantity("5 flibberts")
    assert ok is False


# --------------------------------------------------------------------------- #
# physics_equivalent — the RLVR reward seam
# --------------------------------------------------------------------------- #
def test_equivalent_accepts_correct() -> None:
    r = physics_equivalent("30 N")(r"The force is \boxed{30 N}.", None, {})
    assert r["passed"]


def test_equivalent_accepts_dimensionally_equal_form() -> None:
    r = physics_equivalent("30 N")(r"\boxed{30 kg*m/s^2}", None, {})
    assert r["passed"]


def test_equivalent_rejects_wrong_unit_same_number() -> None:
    # The signature physics failure mode: right number, wrong dimension.
    r = physics_equivalent("30 N")(r"\boxed{30 J}", None, {})
    assert not r["passed"]
    assert "dimension mismatch" in r["reasons"][0]


def test_equivalent_rejects_out_of_tolerance() -> None:
    r = physics_equivalent("30 N")(r"\boxed{33 N}", None, {})  # 10% off, rtol 1%
    assert not r["passed"]


def test_equivalent_within_tolerance() -> None:
    r = physics_equivalent("19.8 m/s")(r"\boxed{19.799 m/s}", None, {})
    assert r["passed"]


def test_equivalent_no_answer_fails() -> None:
    r = physics_equivalent("30 N")("I am not sure.", None, {})
    assert not r["passed"]


# --------------------------------------------------------------------------- #
# physics_sound — gold-free dimensional soundness of stated equalities
# --------------------------------------------------------------------------- #
def test_sound_passes_correct_identity() -> None:
    r = physics_sound()("Since 1 N = 1 kg*m/s^2, the units balance.", None, {})
    assert r["passed"]


def test_sound_flags_dimension_mismatch() -> None:
    r = physics_sound()("Therefore 1 J = 1 kg*m/s^2 here.", None, {})
    assert not r["passed"]
    assert "false physics" in r["reasons"][0]


def test_sound_flags_bad_conversion() -> None:
    r = physics_sound()("Note that 1 km = 1 m in this step.", None, {})
    assert not r["passed"]


def test_sound_passes_good_conversion() -> None:
    r = physics_sound()("We use 1 km = 1000 m throughout.", None, {})
    assert r["passed"]


def test_sound_ignores_pure_prose() -> None:
    r = physics_sound()("The mass equals the sum of the parts.", None, {})
    assert r["passed"]


# --------------------------------------------------------------------------- #
# physics_verifier.verify — accepted/rejected/abstain contract
# --------------------------------------------------------------------------- #
def test_verify_accepted() -> None:
    assert pv.verify(r"\boxed{50 J}", "50 J")["verdict"] == "accepted"


def test_verify_rejected_dimension() -> None:
    assert pv.verify(r"\boxed{50 N}", "50 J")["verdict"] == "rejected"


def test_verify_lean_abstains_when_unavailable() -> None:
    # Opt-in Lean path mirrors the math verifier: abstain fail-closed (no lean-dojo).
    r = pv.verify("", "theorem t : True := by", use_lean=True, lean_proof="trivial")
    assert r["verdict"] == "abstain"
    assert "lean_unavailable" in r["reasons"][0]


def test_verify_lean_requires_proof() -> None:
    r = pv.verify("", "theorem t : True := by", use_lean=True)
    assert r["verdict"] == "abstain"


# --------------------------------------------------------------------------- #
# Reward + dataset
# --------------------------------------------------------------------------- #
def test_reward_positive_and_negative() -> None:
    r_good, _ = physics_reward.reward_for_problem(r"\boxed{30 N}", "30 N")
    r_bad, _ = physics_reward.reward_for_problem(r"\boxed{30 J}", "30 N")
    assert r_good == physics_reward.REWARD_MAX
    assert r_bad == physics_reward.REWARD_MIN


def test_grpo_reward_signature() -> None:
    fn = physics_reward.make_grpo_reward()
    out = fn(["p"], [r"\boxed{30 N}"], gold=["30 N"])
    assert out == [physics_reward.REWARD_MAX]


def test_dataset_contamination_free() -> None:
    data = physics_dataset.build_physics_rl_dataset()
    assert data["family_intersection"] == []
    assert data["train_rows"] and data["eval_rows"]
    assert data["train_sealed"] != data["eval_sealed"]


def test_offline_invariants_pass() -> None:
    ok, detail = physics_reward.offline_invariants()
    assert ok, detail["checks"]


def test_pack_golds_self_consistent() -> None:
    # Every gold in the shipped pack must parse as a physical quantity (no typos
    # that would make the reward un-verifiable).
    for prob in physics_dataset.load_problems():
        ok, _, _ = units.parse_quantity(prob["gold"])
        assert ok, f"gold not parseable: {prob['id']} -> {prob['gold']!r}"
