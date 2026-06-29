# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""The autonomous experiment loop: propose -> run -> read back -> decide -> repeat.

This is the closed loop the role calls "积极地去发现并解决新的问题", grounded in actually-run
experiments. A strategy proposes a config, the backend RUNS it (real nano training), the
loop reads the measured loss, the strategy decides the next config from that result, and it
repeats until the strategy converges or the trial budget is spent.

Honest by construction:
  * **No fabricated results** — every score comes from a real ``backend.run``.
  * **Fail-closed** — a diverged run is recorded as a failure (score=inf) and the strategy
    adapts; the loop never invents a number to keep going.
  * **Provenance** — every trial logs its config, result, and the rationale for the next
    move, so the whole search is auditable. No wall-clock/timestamps (kept deterministic).
  * **Not an AGI agent** — ``canClaimAGI: false``. It is an optimizer over toy experiments.
"""
from __future__ import annotations

from typing import Any, Protocol


class Strategy(Protocol):
    def initial(self) -> "dict[str, Any]":
        pass

    def propose_next(self, history: list) -> "dict[str, Any] | None":
        pass


class Backend(Protocol):
    def run(self, config: "dict[str, Any]") -> "dict[str, Any]":
        pass


def autopilot(strategy: Strategy, backend: Backend, *, max_trials: int = 12,
              patience: int = 4, objective: str = "held_loss") -> "dict[str, Any]":
    """Run the loop. Returns a structured, auditable report of the whole search."""
    history: list[dict] = []
    best: dict | None = None
    no_improve = 0
    stop_reason = "converged"

    config = strategy.initial()
    while config is not None:
        if len(history) >= max_trials:
            stop_reason = "trial_budget_exhausted"
            break
        result = backend.run(config)
        score = result.get(objective, float("inf"))
        if result.get("diverged"):
            score = float("inf")
        trial = {"trial": len(history), "config": config, "result": result,
                 "score": score, "diverged": bool(result.get("diverged"))}

        improved = best is None or score < best["score"]
        if improved:
            best = trial
            no_improve = 0
        else:
            no_improve += 1
        trial["is_best_so_far"] = improved
        trial["rationale"] = _rationale(trial, best, improved)
        history.append(trial)

        if no_improve >= patience:
            stop_reason = "patience_exhausted"
            break
        config = strategy.propose_next(history)

    n_div = sum(1 for h in history if h["diverged"])
    return {
        "agent": "autonomous pretraining-experiment runner",
        "canClaimAGI": False,
        "honesty_note": ("Real closed-loop search over toy nano experiments. Every score is "
                         "measured, diverged runs are recorded as failures, nothing is "
                         "simulated. Not an AGI agent and not a capability claim."),
        "objective": objective,
        "n_trials": len(history),
        "n_diverged": n_div,
        "stop_reason": stop_reason,
        "best": best,
        "history": history,
    }


def _rationale(trial: dict, best: dict | None, improved: bool) -> str:
    if trial["diverged"]:
        return ("run DIVERGED (score=inf) -> recorded as failure; search should back off "
                "(e.g. lower learning rate)")
    if improved:
        return f"new best {trial['score']} at {_short(trial['config'])} -> keep exploring nearby"
    return (f"no improvement over best {best['score'] if best else 'n/a'} -> "
            "narrowing / counting toward patience")


def _short(config: dict) -> str:
    keys = [k for k in ("lr", "hidden", "D", "mix") if k in config]
    return "{" + ", ".join(f"{k}={config[k]}" for k in keys) + "}"


__all__ = ["autopilot", "Strategy", "Backend"]
