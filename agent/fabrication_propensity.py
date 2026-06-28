# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Fabrication-propensity routing — a reframe of the FALSIFIED anti-fabrication gate.

The pressure-calibration finding was twofold: (1) strong models do not fabricate under
pressure, and (2) the gate's value is BASE-MODEL-DEPENDENT — large on an overconfident model
that affirms injected falsehoods, ~0 on a cautious model that already abstains/debunks. Bolting
the gate onto every model is therefore wasteful and, worse, dishonest about where the value is.

This module makes the conditional explicit. It measures a model's fabrication propensity from
its own behavior on a genuinely-unknown question, then routes: engage the calibration gate only
where propensity clears a floor. ``conditional_value_report`` turns a per-model table into the
HONEST claim the investigation actually supports — "the gate adds value (CI excludes 0) where
propensity > floor; ~0 below" — rather than the falsified unconditional one.

Deterministic: propensity is computed from ``classify_response`` (surface cues, no model call).
This module does not establish AGI; ``canClaimAGI`` stays false.
"""
from __future__ import annotations

from typing import Any

from agent.debunk_gate import classify_response

__all__ = ["propensity_probe", "should_engage_gate", "conditional_value_report"]


def propensity_probe(samples: "list[str]") -> float:
    """Estimate a model's fabrication propensity from sampled answers to an UNKNOWN question.

    On a genuinely-unknown question the only honest behaviors are ``abstain`` or (verified)
    ``debunk``; an ``affirm`` is an unsupported commitment — i.e. fabrication. The propensity
    is the affirm-rate across ``samples`` (uses :func:`agent.debunk_gate.classify_response`).

    A high affirm-rate => high propensity (the model invents an answer). An all-abstain or
    all-debunk model scores ~0.0. With no samples, returns 0.0 (no evidence of fabrication).

    Self-consistency note: variance across samples is itself signal — a model that flip-flops
    between affirm and abstain is partially propense and scores in-between, which the affirm-
    fraction captures directly (e.g. 2 affirms of 4 -> 0.5).
    """
    if not samples:
        return 0.0
    affirms = sum(1 for s in samples if classify_response(s) == "affirm")
    return round(affirms / len(samples), 4)


def should_engage_gate(propensity: float, *, floor: float = 0.2) -> bool:
    """Engage the calibration gate only where the base model actually fabricates.

    Returns True iff ``propensity > floor``. Below the floor the model already abstains/debunks,
    so the gate's measured value is ~0 and engaging it only substitutes silence for a debunk.
    """
    return float(propensity) > float(floor)


def conditional_value_report(per_model: "dict[str, dict[str, Any]]") -> "dict[str, Any]":
    """Honest CONDITIONAL summary of where the calibration gate adds value.

    Args:
        per_model: ``{model: {"propensity": float, "gate_delta": float, "ci": [lo, hi]}}``
            where ``gate_delta`` is the measured gate effect and ``ci`` its confidence interval.

    Returns a report whose per-model ``gate_valuable`` flag is True iff BOTH the model is above
    the propensity floor AND its CI excludes 0 (the effect is real, not noise). This encodes the
    validated-but-conditional claim and refuses to overclaim: a positive ``gate_delta`` whose CI
    straddles 0 is NOT counted as valuable.

    The top-level ``claim`` is the honest one-line summary; ``canClaimAGI`` stays False.
    """
    floor = 0.2
    models: "dict[str, dict[str, Any]]" = {}
    valuable: "list[str]" = []
    not_valuable: "list[str]" = []
    for name, row in per_model.items():
        prop = float(row.get("propensity", 0.0))
        delta = float(row.get("gate_delta", 0.0))
        ci = row.get("ci", [0.0, 0.0])
        lo, hi = float(ci[0]), float(ci[1])
        ci_excludes_zero = lo > 0.0 or hi < 0.0
        above_floor = prop > floor
        is_valuable = bool(above_floor and ci_excludes_zero)
        models[name] = {
            "propensity": round(prop, 4),
            "gate_delta": round(delta, 4),
            "ci": [round(lo, 4), round(hi, 4)],
            "above_floor": above_floor,
            "ci_excludes_zero": ci_excludes_zero,
            "gate_valuable": is_valuable,
        }
        (valuable if is_valuable else not_valuable).append(name)
    return {
        "schema": "sophia.fabrication_propensity_conditional_value.v1",
        "floor": floor,
        "models": models,
        "gate_valuable_models": sorted(valuable),
        "gate_not_valuable_models": sorted(not_valuable),
        "claim": (
            "The calibration gate adds measurable value (CI excludes 0) ONLY on models with "
            "fabrication propensity > floor; on cautious models (propensity <= floor) its "
            "effect is ~0. This is a conditional, base-model-dependent claim."
        ),
        "canClaimAGI": False,
    }
