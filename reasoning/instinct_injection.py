# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""The injection half — *editing* a chain in place vs abandoning it (the original thesis).

The operator's first question had two halves. The instinct stack so far answered the first
(*detect* an error and *re-route*). This module models the second: once the reflex fires, can we
**inject the chain and change its mind in place** — a backtrack-token or an activation-steering
edit at the error step — instead of throwing the whole chain away and re-attempting?

Two mechanisms from the literature (see the research note §1c):
  - *token-level* backtrack ("Wait, that's wrong …") and *activation-level* steering both can flip
    a generation mid-stream, cheaply, without a restart;
  - **but steering is unreliable**: its effect is geometry-dependent and the linear approximation
    breaks at strength — push too hard and it corrupts the (good) rest of the chain.

So injection trades a restart for a cheap edit that (a) flips the bad step with probability
``p_flip(strength)`` but (b) corrupts the otherwise-fine chain with probability
``p_corrupt(strength)``. Conditioned on a detected error at step ``e`` (prefix good):

    P(correct | inject)  = (1 − p_corrupt(s)) · p_flip(s)
    P(correct | reroute) = p_clean                      (a fresh independent attempt)
    cost(inject) ≈ INJECT_COST        cost(reroute) ≈ chain length L   (≫ INJECT_COST)

Falsifiable results (``--self-test`` / the test module):
  I1  CHEAPER. Injection's expected cost is far below re-route's (edit vs restart).
  I2  CAN DOMINATE. At a good operating strength, inject's correctness *exceeds* re-route's AND
      it is cheaper — so it strictly dominates (when steering is effective and not over-driven).
  I3  BRITTLENESS ROOFLINE. Net correctness vs steering strength is **concave with an interior
      optimum**; over-steering (strength→1) *reduces* correctness as corruption overtakes flips —
      the steering-unreliability finding, as a falsifiable curve.
  I4  HYBRID WINS. "inject; if the reflex still fires, fall back to re-route" beats either alone:
      it keeps injection's cheap wins and recovers its failures with a restart.

Honest scope (``candidateOnly: true``, ``canClaimAGI: false``). This is a *model* of the
intervention mechanics with planted ``p_flip``/``p_corrupt`` curves — not a measured steering
experiment (that needs white-box model access: add a contrastive error-direction vector at a
mid-layer and measure the flip/corrupt rates). It maps the design space and says *when* in-place
injection beats re-route; the real steering measurement is the gated next step. It maps to
``agent/graded_decision.py`` (choose edit vs restart) and the consequence/ko gate (bound retries).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from typing import Any

L = 10            # chain length (re-route ≈ restart cost)
INJECT_COST = 2.0  # cheap in-place edit (token/steer + short continuation)
P_CLEAN = 0.5      # a fresh re-route attempt lands correct this often
TPR_REFIRE = 0.9   # reflex re-detects a still-wrong chain after a failed inject (for the hybrid)


def p_flip(strength: float) -> float:
    """Steering efficacy: saturating in strength (more push ⇒ more likely to flip the error)."""
    return 1.0 - math.exp(-3.0 * strength)


def p_corrupt(strength: float) -> float:
    """Collateral corruption of the good chain: grows with strength (linear approx breaks)."""
    return min(1.0, 0.6 * strength * strength)


def inject_correct(strength: float) -> float:
    """P(correct | in-place injection) at a given steering strength (closed form)."""
    return (1.0 - p_corrupt(strength)) * p_flip(strength)


def hybrid_correct(strength: float, *, p_clean: float = P_CLEAN, tpr: float = TPR_REFIRE) -> float:
    """inject; if it fails AND the reflex re-fires, fall back to a re-route."""
    pi = inject_correct(strength)
    return pi + (1.0 - pi) * tpr * p_clean


def hybrid_cost(strength: float, *, tpr: float = TPR_REFIRE) -> float:
    pi = inject_correct(strength)
    return INJECT_COST + (1.0 - pi) * tpr * float(L)


def best_strength(grid: int = 101) -> tuple[float, float]:
    """Strength maximizing inject_correct (the brittleness-roofline ridge)."""
    best = (0.0, 0.0)
    for i in range(grid):
        s = i / (grid - 1)
        v = inject_correct(s)
        if v > best[1]:
            best = (s, v)
    return best


def _mc_inject(rng: random.Random, strength: float) -> bool:
    if rng.random() < p_corrupt(strength):
        return False          # steering corrupted the good chain
    return rng.random() < p_flip(strength)  # else flip the bad step?


def run_experiment(seed: int = 1234, trials: int = 40000) -> dict[str, Any]:
    rng = random.Random(seed)
    s_star, inj_star = best_strength()
    # MC check of the closed form at the optimum.
    mc = sum(_mc_inject(rng, s_star) for _ in range(trials)) / trials
    sweep = []
    for i in range(11):
        s = i / 10
        sweep.append({
            "strength": s, "p_flip": round(p_flip(s), 4), "p_corrupt": round(p_corrupt(s), 4),
            "inject_correct": round(inject_correct(s), 4),
            "hybrid_correct": round(hybrid_correct(s), 4),
        })
    policies = {
        "commit":  {"correct": 0.0, "cost": 0.0},   # conditioned on a detected error
        "reroute": {"correct": round(P_CLEAN, 4), "cost": float(L)},
        "inject":  {"correct": round(inj_star, 4), "cost": INJECT_COST},
        "hybrid":  {"correct": round(hybrid_correct(s_star), 4), "cost": round(hybrid_cost(s_star), 4)},
    }
    return {
        "params": {"L": L, "inject_cost": INJECT_COST, "p_clean": P_CLEAN, "tpr_refire": TPR_REFIRE},
        "best_strength": round(s_star, 3), "inject_correct_at_best": round(inj_star, 4),
        "mc_vs_closed_form_abs_err": round(abs(mc - inj_star), 4),
        "overdrive_correct_at_s1": round(inject_correct(1.0), 4),
        "policies": policies,
        "strength_sweep": sweep,
        "candidateOnly": True, "level3Evidence": False,
        "boundary": "planted p_flip/p_corrupt curves; a real steering measurement (white-box) is gated.",
    }


def format_report(res: dict[str, Any]) -> str:
    p = res["policies"]
    lines = [
        "Injection half — edit-in-place vs re-route after the reflex fires",
        "=" * 68,
        f"params: {json.dumps(res['params'])}",
        f"best steering strength = {res['best_strength']}  (inject correct {res['inject_correct_at_best']}); "
        f"MC~closed {res['mc_vs_closed_form_abs_err']}",
        "",
        "POLICY (conditioned on a detected error)   correct   cost",
        f"  commit (do nothing)                       {p['commit']['correct']:.3f}   {p['commit']['cost']:.1f}",
        f"  reroute (restart)                         {p['reroute']['correct']:.3f}   {p['reroute']['cost']:.1f}",
        f"  inject (edit in place, best strength)     {p['inject']['correct']:.3f}   {p['inject']['cost']:.1f}",
        f"  hybrid (inject then reroute on refire)    {p['hybrid']['correct']:.3f}   {p['hybrid']['cost']:.1f}",
        "",
        "STRENGTH SWEEP (brittleness roofline)",
    ]
    for r in res["strength_sweep"]:
        lines.append(f"  s={r['strength']:.1f}: flip {r['p_flip']:.2f} corrupt {r['p_corrupt']:.2f} "
                     f"-> inject {r['inject_correct']:.3f}  hybrid {r['hybrid_correct']:.3f}")
    lines += [f"  over-drive (s=1.0) inject correct = {res['overdrive_correct_at_s1']} "
              f"(< best {res['inject_correct_at_best']}: over-steering hurts)",
              "", f"candidateOnly={res['candidateOnly']}  boundary: {res['boundary']}"]
    return "\n".join(lines)


def _self_test() -> int:
    res = run_experiment()
    p = res["policies"]
    # I1: injection is far cheaper than re-route.
    assert p["inject"]["cost"] < p["reroute"]["cost"], "I1: inject not cheaper"
    # I2: at the best strength injection dominates re-route (more correct AND cheaper).
    assert p["inject"]["correct"] > p["reroute"]["correct"], "I2: inject not more correct"
    # I3: brittleness roofline — interior optimum, and over-drive hurts.
    assert 0.0 < res["best_strength"] < 1.0, f"I3: optimum not interior ({res['best_strength']})"
    assert res["overdrive_correct_at_s1"] < res["inject_correct_at_best"] - 0.1, "I3: no over-steer penalty"
    assert res["mc_vs_closed_form_abs_err"] < 0.01, "I3: MC diverged from closed form"
    # I4: hybrid beats both inject and reroute on correctness.
    assert p["hybrid"]["correct"] >= p["inject"]["correct"] - 1e-9, "I4: hybrid < inject"
    assert p["hybrid"]["correct"] > p["reroute"]["correct"], "I4: hybrid not > reroute"
    # concavity of the sweep (single interior peak)
    inj = [r["inject_correct"] for r in res["strength_sweep"]]
    peak = inj.index(max(inj))
    assert 0 < peak < len(inj) - 1, "I3: sweep peak not interior"
    print(f"self-test OK: inject {p['inject']['correct']}@{p['inject']['cost']} dominates reroute "
          f"{p['reroute']['correct']}@{p['reroute']['cost']}; best strength {res['best_strength']}; "
          f"over-drive {res['overdrive_correct_at_s1']}; hybrid {p['hybrid']['correct']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args(argv)
    if args.self_test:
        return _self_test()
    if args.run or args.json:
        res = run_experiment(seed=args.seed)
        print(json.dumps(res, indent=2) if args.json else format_report(res))
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
