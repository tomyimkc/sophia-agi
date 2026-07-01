# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Process world model (#6) — anticipate WHERE a derivation goes wrong before it finishes.

okf_trace.locate_wrong_step finds the first failing step AFTER a trace completes. Error-centric
intelligence wants the world model to anticipate the consequence DURING the process, so you can abstain
or re-route early instead of spending the whole derivation. This maintains a running validity
confidence over reasoning steps and flags the FIRST step whose local check drops below threshold —
before the final answer — turning outcome supervision into a predictive dynamics model of reasoning
trajectories. stdlib-only, deterministic. canClaimAGI false; CANDIDATE prototype.
"""
from __future__ import annotations

from typing import Any


def predict_error_step(step_scores: "list[float]", *, threshold: float = 0.5) -> dict:
    """step_scores[i] = local validity confidence of reasoning step i (1=clearly valid, 0=clearly
    wrong). Returns the FIRST step predicted wrong (below threshold) + how many steps that early flag
    saves vs running the full derivation to the final verifier."""
    n = len(step_scores)
    predicted = next((i for i, s in enumerate(step_scores) if s < threshold), None)
    if predicted is None:
        return {"predicted_error_step": None, "action": "continue", "steps_saved": 0, "n_steps": n}
    return {"predicted_error_step": predicted, "action": "abstain-or-reroute",
            "steps_saved": max(0, n - (predicted + 1)),      # steps not spent after the early flag
            "n_steps": n,
            "note": "flagged during generation, before the final verifier would have caught it."}


def running_validity(step_scores: "list[float]") -> "list[float]":
    """The world-model state: cumulative P(derivation still valid) = product of step confidences.
    Monotone non-increasing; its collapse is the anticipated failure."""
    out, acc = [], 1.0
    for s in step_scores:
        acc *= max(0.0, min(1.0, s))
        out.append(round(acc, 6))
    return out


def offline_invariants() -> "tuple[bool, dict]":
    checks: dict[str, bool] = {}
    err = [0.9, 0.85, 0.2, 0.7, 0.9]            # error at step 2
    clean = [0.9, 0.88, 0.92, 0.8, 0.95]
    r_err = predict_error_step(err)
    r_clean = predict_error_step(clean)
    # 1. Flags the first wrong step (matches locate_wrong_step's first-failure semantics).
    checks["flags_first_wrong_step"] = r_err["predicted_error_step"] == 2
    # 2. It is EARLY: flagged before the last step -> saves work (the predictive advantage).
    checks["flags_before_end"] = r_err["predicted_error_step"] < len(err) - 1
    checks["saves_steps"] = r_err["steps_saved"] == 2
    # 3. A clean derivation is not flagged (no false abstention).
    checks["clean_not_flagged"] = r_clean["predicted_error_step"] is None
    # 4. Running validity collapses at the error and is monotone non-increasing.
    rv = running_validity(err)
    checks["validity_monotone"] = all(rv[i] >= rv[i + 1] - 1e-9 for i in range(len(rv) - 1))
    checks["validity_collapses_at_error"] = rv[2] < 0.5 <= rv[1]
    return all(checks.values()), {"checks": checks, "running_validity_err": rv}


if __name__ == "__main__":
    ok, d = offline_invariants()
    print("process_world_model offline invariants:", "PASS" if ok else "FAIL")
    for k, v in d["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    print(f"  running validity (error traj): {d['running_validity_err']}")
