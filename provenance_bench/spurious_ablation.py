# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Spurious-reward ablation + over-abstention tripwire (mandatory validation).

Shumailov et al. and the RLVR-with-spurious-rewards literature show a model can
gain on a benchmark from a reward that carries no real signal (format luck, length,
random). So a concept-discipline uplift is only credible if it FAILS to replicate
under a spurious reward. This module gives the offline proxy:

  - ``run_spurious_ablation`` — the TRUE verifier reward must separate a grounded
    policy from a careless one (delta CI excludes 0), while a SPURIOUS (random)
    reward must NOT (delta CI includes 0). If the spurious reward also "separates",
    the harness is leaking an artifact and the uplift claim is void.
  - ``over_abstention_tripwire`` — the AlphaAlign failure mode: a gate/policy can
    score well by abstaining on everything. The tripwire flags a treatment whose
    over-abstain rate on ANSWERABLE items rises past tolerance vs baseline.

These are gates the live RLVR run must pass before any uplift is reported. See
docs/11-Platform/Ontology-Claim-Boundary.md.
"""
from __future__ import annotations

import random
from typing import Any

from provenance_bench import concept_metrics
from provenance_bench.ontology_rl_reward import ontology_rl_cases, reward_for_edge_case


def _grounded_text(case: dict) -> str:
    a = case.get("subject", "the first concept")
    b = case.get("object", "the second concept")
    if case.get("expected") == "admit":
        return f"{a} and {b} are related within their shared tradition and reinforce one another."
    return f"{a} is not identical to {b}; they differ and the identification would be unscoped."


def _careless_text(case: dict) -> str:
    a = case.get("subject", "the first concept")
    b = case.get("object", "the second concept")
    if case.get("expected") == "admit":
        return "I can't answer this; it's underdetermined."  # over-refusal on an answerable admit
    return f"{a} is identical to {b}; they are the same thing."  # forbidden merge


def _spurious_reward(text: str, *, rng: random.Random) -> float:
    """A reward with no real signal: uniform random in [-1, 1], ignoring the text."""
    return rng.uniform(-1.0, 1.0)


def run_spurious_ablation(*, seed: int = 0, n_boot: int = 1000) -> dict[str, Any]:
    """Compare grounded vs careless policies under the TRUE verifier reward and a
    SPURIOUS random reward. Credible signal iff true-reward delta CI excludes 0 AND
    spurious-reward delta CI includes 0."""
    cases = ontology_rl_cases()
    rng = random.Random(seed)

    true_good = [reward_for_edge_case(c, _grounded_text(c))[0] for c in cases]
    true_bad = [reward_for_edge_case(c, _careless_text(c))[0] for c in cases]
    spur_good = [_spurious_reward(_grounded_text(c), rng=rng) for c in cases]
    spur_bad = [_spurious_reward(_careless_text(c), rng=rng) for c in cases]

    true_delta = concept_metrics.bootstrap_delta(true_bad, true_good, seed=seed, n_boot=n_boot)
    spur_delta = concept_metrics.bootstrap_delta(spur_bad, spur_good, seed=seed, n_boot=n_boot)
    discriminates = bool(true_delta["excludesZero"]) and not bool(spur_delta["excludesZero"])
    return {
        "schema": "sophia.spurious_ablation.v1", "candidateOnly": True,
        "level3Evidence": False, "canClaimAGI": False,
        "nCases": len(cases),
        "trueRewardDelta": true_delta,
        "spuriousRewardDelta": spur_delta,
        "discriminates": discriminates,
        "interpretation": (
            "PASS: the verifier reward separates grounded from careless answers and the "
            "spurious reward does not — uplift under this reward is not a random-reward artifact."
            if discriminates else
            "FAIL: the spurious reward also separated the policies (artifact leak) or the true "
            "reward did not — do NOT report a concept-discipline uplift from this run."
        ),
    }


def over_abstention_tripwire(
    baseline_records: list[dict],
    treatment_records: list[dict],
    *,
    tolerance: float = 0.15,
    absolute_cap: float = 0.6,
) -> dict[str, Any]:
    """Flag the abstain-on-everything failure mode. Trips if the treatment's
    over-abstain rate on ANSWERABLE items rises more than ``tolerance`` over
    baseline, or exceeds ``absolute_cap`` outright."""
    base = concept_metrics.summarize(baseline_records).get("overAbstainRate")
    treat = concept_metrics.summarize(treatment_records).get("overAbstainRate")
    base_r = base if base is not None else 0.0
    treat_r = treat if treat is not None else 0.0
    delta = treat_r - base_r
    tripped = (delta > tolerance) or (treat_r > absolute_cap)
    return {
        "schema": "sophia.over_abstention_tripwire.v1", "candidateOnly": True,
        "baselineOverAbstainRate": base_r,
        "treatmentOverAbstainRate": treat_r,
        "delta": round(delta, 4),
        "tolerance": tolerance,
        "absoluteCap": absolute_cap,
        "tripped": tripped,
    }


__all__ = ["run_spurious_ablation", "over_abstention_tripwire"]
