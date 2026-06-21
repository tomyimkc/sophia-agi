"""Aggregate multiple runs into point estimates with bootstrap 95% CIs.

Turns the single-run, illustrative numbers into something citable: each model is
run N times; we report the pooled hallucination rates (alone vs gated) and a
paired bootstrap confidence interval on the delta, plus per-run deltas so
variance is visible.
"""

from __future__ import annotations

import random

from provenance_bench import score


def _ci(xs: list[float], alpha: float = 0.05) -> list[float]:
    xs = sorted(xs)
    n = len(xs)
    lo = xs[max(0, int((alpha / 2) * n))]
    hi = xs[min(n - 1, int((1 - alpha / 2) * n))]
    return [round(lo, 4), round(hi, 4)]


def aggregate_runs(runs: list[list[dict]], *, n_boot: int = 2000, seed: int = 0) -> dict:
    """``runs``: a list of per-run result lists (each from ``runner.run_cases``)."""
    per_run = [score.score(r) for r in runs]

    # paired false-case observations across all runs (raw vs gated hallucination)
    false_pairs = [
        (int(res["raw"]["hallucinated"]), int(res["gated"]["hallucinated"]))
        for r in runs for res in r if res["label"] == "false"
    ]
    # true-case false-positive observations (correct-alone -> broken-by-gate)
    correct_alone, broke = 0, 0
    bad_alone, fixed = 0, 0
    for r in runs:
        for res in r:
            if res["label"] == "true" and res["raw"]["affirmed_gold"]:
                correct_alone += 1
                broke += int(not res["gated"]["affirmed_gold"])
            if res["label"] == "false" and res["raw"]["hallucinated"]:
                bad_alone += 1
                fixed += int(not res["gated"]["hallucinated"])

    def stat(sample):
        a = sum(p[0] for p in sample) / len(sample)
        g = sum(p[1] for p in sample) / len(sample)
        return a, g, a - g

    rng = random.Random(seed)
    n = len(false_pairs)
    boot_a, boot_g, boot_d = [], [], []
    if n:
        for _ in range(n_boot):
            sample = [false_pairs[rng.randrange(n)] for _ in range(n)]
            a, g, d = stat(sample)
            boot_a.append(a); boot_g.append(g); boot_d.append(d)
    pt = stat(false_pairs) if n else (0.0, 0.0, 0.0)

    return {
        "runs": len(runs),
        "falseObs": n,
        "hallucinationRateAlone": round(pt[0], 4),
        "ciAlone": _ci(boot_a) if boot_a else [0.0, 0.0],
        "hallucinationRateGated": round(pt[1], 4),
        "ciGated": _ci(boot_g) if boot_g else [0.0, 0.0],
        "delta": round(pt[2], 4),
        "ciDelta": _ci(boot_d) if boot_d else [0.0, 0.0],
        "perRunDelta": [s["delta"] for s in per_run],
        "falsePositiveCost": round(broke / correct_alone, 4) if correct_alone else 0.0,
        "coverageRecall": round(fixed / bad_alone, 4) if bad_alone else 0.0,
        "coverageDetail": {"hallucinatedAlone": bad_alone, "fixedByGate": fixed},
    }
