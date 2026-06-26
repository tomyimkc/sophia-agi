# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Generational compounding driver — proof that improvement compounds on real weights.

One adapter promoted (run #3) only shows the gate accepts a weight delta. *Compounding*
needs generation g+1's weights to beat generation g's weights, gated, repeatedly:

    B0 = frozen open base
    for g in 1..G:
        Ag = train LoRA on  B_{g-1}        (the current canonical merged checkpoint)
        eval Ag on the SEALED held-out, 3 seeds  -> aggregate  (agent/ssil_aggregate)
        GATE: Ag.meanAfter must beat canonical(B_{g-1}) by > min_delta, with
              NON-OVERLAPPING confidence intervals, no protected regression, no
              contamination, all 3 seeds promote
        if promote:  B_g = merge(B_{g-1}, Ag)   <- the new canonical the next gen must beat
        else:        loop converged (verifier/task ceiling hit) — stop honestly

The proof artifact is the **gated, monotone-rising canonical curve** across generations
with non-overlapping CIs — that rising-and-gated curve IS compounding on weights.

This module is the deterministic STATE MACHINE (consume per-generation aggregates,
decide promote/converge, build the curve). The merge-then-train itself runs on GPU
(RunPod). Crucially it also runs the **negative control**: the same generations with
the gate disabled — where a contaminated/gamed generation is admitted and the curve
is fake. "Monotone under the gate; collapses without it" is what distinguishes real
compounding from overfitting.

Honesty: a rising curve over a few generations on one task is BOUNDED compounding
within the verifier's reach — it will plateau. NOT open-ended RSI. canClaimAGI=false.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any

from agent.ssil_aggregate import AdapterAggregate


def _sem(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    return round(statistics.stdev(values) / math.sqrt(n), 5)


@dataclass(frozen=True)
class Generation:
    """One generation: a multi-seed aggregate for an adapter trained on the prior canonical.

    Optional real-data metadata (set by the live/mock driver; ``None`` in the demo
    fixtures) is threaded verbatim into the per-generation record so a compounding
    proof can be audited end-to-end: which win-set size it trained on, whether the
    contamination guard was clean, whether the formal-verifier backend was live, and
    the per-generation gate verdict + held-out delta CI.
    """

    gen: int
    adapter_id: str
    trained_on: str            # the base/canonical checkpoint this generation trained on
    aggregate: AdapterAggregate
    win_set_size: int | None = None
    contamination_status: dict[str, Any] | None = None
    solver_checked: bool | None = None
    gate_verdict: str | None = None
    heldout_delta_ci: dict[str, float] | None = None


def evaluate_generations(
    gens: list[Generation],
    *,
    min_delta: float = 0.03,
    ci_k: float = 1.0,
    gated: bool = True,
    base_after: float | None = None,
) -> dict[str, Any]:
    """Walk the generations, gating each against the running canonical. Returns the
    curve + per-generation records + convergence point. ``gated=False`` is the negative
    control (admits on raw mean improvement only — no protected/contamination/CI checks)."""
    canonical_after: float | None = base_after
    canonical_sem = 0.0
    curve: list[float] = []
    records: list[dict[str, Any]] = []
    converged_at: int | None = None

    for g in gens:
        afters = [r.after for r in g.aggregate.runs]
        mean_after = round(statistics.mean(afters), 4) if afters else 0.0
        sem_after = _sem(afters)
        summary = g.aggregate.summary(baseline_after=canonical_after)
        n = summary["n"]

        # Lower bound of this gen must clear the upper bound of the prior canonical.
        ci_separated = (canonical_after is None) or (
            (mean_after - ci_k * sem_after) > (canonical_after + ci_k * canonical_sem)
        )
        beats_floor = (canonical_after is None and mean_after > 0) or (
            canonical_after is not None and (mean_after - canonical_after) >= min_delta
        )

        if gated:
            promoted = bool(
                n >= g.aggregate.canonical_n
                and summary["promotes"] == n
                and beats_floor
                and ci_separated
                and not summary["anyProtectedRegression"]
                and not summary["anyContaminated"]
            )
        else:  # negative control: admit on raw mean improvement, ignore the gate
            promoted = mean_after > (canonical_after if canonical_after is not None else 0.0)

        records.append({
            "gen": g.gen, "adapterId": g.adapter_id, "trainedOn": g.trained_on,
            "meanAfter": mean_after, "semAfter": sem_after,
            "beatFloor": beats_floor, "ciSeparated": ci_separated,
            "promotes": summary["promotes"], "n": n,
            "anyProtectedRegression": summary["anyProtectedRegression"],
            "anyContaminated": summary["anyContaminated"],
            "promoted": promoted,
            **({"winSetSize": g.win_set_size} if g.win_set_size is not None else {}),
            **({"contaminationStatus": g.contamination_status}
               if g.contamination_status is not None else {}),
            **({"solverChecked": g.solver_checked} if g.solver_checked is not None else {}),
            **({"gateVerdict": g.gate_verdict} if g.gate_verdict is not None else {}),
            **({"heldoutDeltaWithCI": g.heldout_delta_ci} if g.heldout_delta_ci is not None else {}),
        })

        if promoted:
            canonical_after, canonical_sem = mean_after, sem_after
            curve.append(mean_after)
        elif gated and converged_at is None:
            converged_at = g.gen  # first generation that fails to compound = ceiling

    monotone = all(b > a for a, b in zip(curve, curve[1:]))
    return {
        "schema": "sophia.ssil_generations.v1",
        "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False,
        "gated": gated,
        "curve": curve,
        "generations": records,
        "promotedGenerations": sum(1 for r in records if r["promoted"]),
        "monotoneRising": monotone,
        "convergedAt": converged_at,
        "finalCanonicalAfter": canonical_after,
        "boundary": ("Bounded compounding within the verifier's reach; it plateaus at the "
                     "task/verifier ceiling. NOT open-ended RSI. canClaimAGI=false."),
    }


def compounding_proof(gens: list[Generation], *, min_delta: float = 0.03, ci_k: float = 1.0,
                      base_after: float | None = None) -> dict[str, Any]:
    """The full proof: the gated curve PLUS the negative control (gate off). Real
    compounding = monotone rising under the gate AND the gate rejecting the
    contaminated/gamed generation the negative control would admit."""
    gated = evaluate_generations(gens, min_delta=min_delta, ci_k=ci_k, gated=True, base_after=base_after)
    ungated = evaluate_generations(gens, min_delta=min_delta, ci_k=ci_k, gated=False, base_after=base_after)
    gate_caught = [r["gen"] for r in gated["generations"] if not r["promoted"]
                   and (r["anyContaminated"] or r["anyProtectedRegression"])]
    ungated_admitted = {r["gen"] for r in ungated["generations"] if r["promoted"]}
    gate_made_a_difference = any(g in ungated_admitted for g in gate_caught)
    return {
        "schema": "sophia.ssil_compounding_proof.v1",
        "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False,
        "gated": gated, "negativeControl": ungated,
        "gateCaughtGenerations": gate_caught,
        "gateMadeADifference": gate_made_a_difference,
        "proves": {
            "compounds_under_gate": gated["monotoneRising"] and len(gated["curve"]) >= 2,
            "negative_control_diverges": gate_made_a_difference,
        },
        "boundary": gated["boundary"],
    }


# --- demo fixtures -----------------------------------------------------------

from agent.ssil_aggregate import SeedRun  # noqa: E402


def _gen(idx: int, trained_on: str, afters: list[float], *, before: float,
         protected: float = 0.79, protected_after: float | None = None,
         contaminated: bool = False) -> Generation:
    prot_after = protected if protected_after is None else protected_after
    runs = [SeedRun(seed=s, before=before, after=a, protected_before=protected,
                    protected_after=prot_after, contaminated=contaminated)
            for s, a in enumerate(afters)]
    agg = AdapterAggregate(adapter_id=f"sophia-rlvr-v{idx}",
                           config={"adapter": f"sophia-rlvr-v{idx}", "kind": "lora_adapter"},
                           runs=runs, canonical_n=3)
    return Generation(gen=idx, adapter_id=f"sophia-rlvr-v{idx}", trained_on=trained_on, aggregate=agg)


def demo_compounding_report() -> dict[str, Any]:
    """3 honest generations rising 0.71 -> 0.79 -> 0.85, then a 4th that hits the
    ceiling (no gain -> converge). Plus a contaminated 'gamed' generation that the
    gate rejects but the negative control would admit."""
    gens = [
        _gen(1, "Qwen2.5-3B(base)", [0.70, 0.71, 0.72], before=0.53),     # beats base
        _gen(2, "B1", [0.78, 0.79, 0.80], before=0.71),                   # beats canonical 0.71
        _gen(3, "B2", [0.84, 0.85, 0.86], before=0.79),                   # beats canonical 0.79
        _gen(4, "B3", [0.84, 0.85, 0.86], before=0.85),                   # no gain -> converge
        _gen(5, "B3", [0.95, 0.96, 0.97], before=0.85, contaminated=True),  # gamed -> gate must reject
    ]
    proof = compounding_proof(gens, min_delta=0.03, ci_k=1.0)
    gated = proof["gated"]
    proof["invariants"] = {
        "gated_curve_monotone_rising": gated["monotoneRising"] and gated["curve"] == [0.71, 0.79, 0.85],
        "converges_at_ceiling": gated["convergedAt"] == 4,
        "gate_rejects_contaminated_gen": 5 in proof["gateCaughtGenerations"],
        "negative_control_would_admit_it": 5 in {r["gen"] for r in proof["negativeControl"]["generations"] if r["promoted"]},
        "gate_made_a_difference": proof["gateMadeADifference"],
        "no_overclaim": proof["canClaimAGI"] is False,
    }
    return proof


if __name__ == "__main__":
    import json
    print(json.dumps(demo_compounding_report(), ensure_ascii=False, indent=2))
