# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Successive-halving / ASHA scheduler (C3) — trial-efficient search over expensive runs.

When every trial costs real GPU money, naive search is wasteful: most configs are bad and
you learn that early. Successive halving runs MANY configs cheaply (a small budget — few
epochs, a small eval slice), keeps the top ``1/eta``, promotes survivors to a larger budget,
and repeats. Total spend is a fraction of running every config to completion, and the best
config still surfaces. This is the scheduler that makes a real RunPod sweep affordable.

Generic and backend-agnostic: ``run_fn(config, budget) -> score`` (lower is better; ``inf``
for a diverged/failed run). Pass a ``CostGovernor`` to make it **fail-closed on budget** —
the sweep stops the moment the next batch would exceed the ceiling, and says so. Pure stdlib,
deterministic.
"""
from __future__ import annotations

from typing import Any, Callable

from pretraining.autopilot.cost_governor import CostGovernor


def successive_halving(configs: "list[dict[str, Any]]",
                       run_fn: "Callable[[dict, float], float]",
                       budgets: "list[float]", *, eta: float = 2.0,
                       governor: CostGovernor | None = None) -> "dict[str, Any]":
    """Run ASHA-style successive halving.

    Args:
        configs: candidate configs (the wider the better — halving prunes them).
        run_fn:  (config, budget) -> score (lower better; inf = failed).
        budgets: increasing per-rung budgets, e.g. epochs [1, 2, 4].
        eta:     reduction factor per rung (keep ceil(n/eta) survivors).
        governor: optional CostGovernor; if set, the sweep is fail-closed on the ceiling.

    Returns a structured report: the per-rung ladder, the surviving best config, the number
    of runs actually executed, and whether the budget truncated the search.
    """
    if not configs or not budgets:
        return {"ok": False, "reason": "need configs and budgets", "best": None,
                "rungs": [], "runs_executed": 0, "truncated": False}

    survivors = list(configs)
    rungs: list[dict] = []
    runs_executed = 0
    truncated = False

    for rung_idx, budget in enumerate(budgets):
        if governor is not None and not governor.can_afford(len(survivors)):
            truncated = True
            rungs.append({"rung": rung_idx, "budget": budget,
                          "skipped": True,
                          "reason": f"budget ceiling: cannot afford {len(survivors)} runs",
                          "affordable": governor.max_affordable_trials()})
            break

        scored = []
        for cfg in survivors:
            score = run_fn(cfg, budget)
            runs_executed += 1
            if governor is not None:
                # book the projected per-trial cost as spend (calibration refines this)
                governor.record(governor.est_hours)
            scored.append({"config": cfg, "score": score})
        scored.sort(key=lambda s: s["score"])

        keep = max(1, int(-(-len(scored) // eta)))   # ceil(n/eta)
        next_survivors = [s["config"] for s in scored[:keep]]
        rungs.append({
            "rung": rung_idx, "budget": budget,
            "n_in": len(survivors), "n_kept": keep,
            "scores": [round(s["score"], 5) if s["score"] != float("inf") else "inf"
                       for s in scored],
            "best_score": scored[0]["score"] if scored[0]["score"] != float("inf") else None,
            "promoted": next_survivors,
        })
        survivors = next_survivors
        if len(survivors) <= 1:
            break

    # the best is the top survivor at the highest rung that actually ran
    best = None
    for r in reversed(rungs):
        if not r.get("skipped") and r.get("promoted"):
            best = r["promoted"][0]
            best_score = r["best_score"]
            break
    return {
        "ok": True,
        "eta": eta,
        "budgets": budgets,
        "n_configs": len(configs),
        "runs_executed": runs_executed,
        "naive_runs": len(configs) * len(budgets),
        "savings_vs_naive": round(1 - runs_executed / (len(configs) * len(budgets)), 4),
        "truncated": truncated,
        "rungs": rungs,
        "best": best,
        "cost": governor.snapshot() if governor is not None else None,
    }


__all__ = ["successive_halving"]
