"""Aggregate multiple runs into point estimates with bootstrap 95% CIs.

Turns the single-run, illustrative numbers into something citable: each model is
run N times; we report the pooled hallucination rates (alone vs gated) and a
paired bootstrap confidence interval on the delta, plus per-run deltas so
variance is visible.
"""

from __future__ import annotations

import random

from provenance_bench import consensus, score


def _ci(xs: list[float], alpha: float = 0.05) -> list[float]:
    xs = sorted(xs)
    n = len(xs)
    lo = xs[max(0, int((alpha / 2) * n))]
    hi = xs[min(n - 1, int((1 - alpha / 2) * n))]
    return [round(lo, 4), round(hi, 4)]


KAPPA_FLOOR = 0.40   # "moderate" agreement, the minimum for a validated headline


def _distinct_families(judges: "list[str] | None") -> int:
    """Count distinct provider families among judge specs ('anthropic:..' -> 'anthropic')."""
    if not judges:
        return 0
    return len({j.split(":", 1)[0].strip().lower() for j in judges if j and j.strip()})


def aggregate_runs(
    runs: list[list[dict]], *, n_boot: int = 2000, seed: int = 0,
    model_spec: "str | None" = None, judges: "list[str] | None" = None,
) -> dict:
    """``runs``: a list of per-run result lists (each from ``runner.run_cases``).

    ``model_spec``/``judges`` let the validated-gate check that the run was a real
    multi-family-judge run (not mock/lexical/single-judge).
    """
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

    # inter-judge agreement, when a consensus judge attached per-judge votes
    vote_lists = [res["raw"]["votes"] for r in runs for res in r if res["raw"].get("votes")]
    agreement = consensus.percent_agreement(vote_lists)

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
        "judgeAgreement": agreement,
        "validated": _is_validated(runs, agreement, model_spec, judges, _ci(boot_d) if boot_d else [0.0, 0.0]),
        "validatedChecks": _validated_checks(runs, agreement, model_spec, judges, _ci(boot_d) if boot_d else [0.0, 0.0]),
    }


def _validated_checks(runs, agreement, model_spec, judges, ci_delta) -> dict:
    kappa = (agreement or {}).get("meanPairwiseKappa")
    return {
        "notMock": bool(model_spec) and "mock" not in (model_spec or ""),
        "multiFamilyJudges": _distinct_families(judges) >= 2,
        "kappaAboveFloor": kappa is not None and kappa >= KAPPA_FLOOR,
        "atLeast3Runs": len(runs) >= 3,
        "ciExcludesZero": bool(ci_delta) and (ci_delta[0] > 0 or ci_delta[1] < 0),
    }


def _is_validated(runs, agreement, model_spec, judges, ci_delta) -> bool:
    return all(_validated_checks(runs, agreement, model_spec, judges, ci_delta).values())
