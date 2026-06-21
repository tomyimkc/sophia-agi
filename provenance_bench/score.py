"""Aggregate per-case results into the three honest Provenance Delta metrics.

1. hallucinated-attribution rate (false cases): asserted a false lineage /
   total false cases — reported ALONE vs BEHIND GATE. The delta is the headline.
2. false-positive cost (true cases): of true answers the model got right ALONE,
   the fraction the gate then broke (no longer affirmed gold).
3. coverage / recall (false cases): of the false cases hallucinated ALONE, the
   fraction the gate fixed (no longer hallucinated when gated).
"""

from __future__ import annotations


def _rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def score(results: list[dict]) -> dict:
    false_cases = [r for r in results if r["label"] == "false"]
    true_cases = [r for r in results if r["label"] == "true"]

    halluc_alone = sum(1 for r in false_cases if r["raw"]["hallucinated"])
    halluc_gated = sum(1 for r in false_cases if r["gated"]["hallucinated"])

    # false-positive cost: correct-alone true answers the gate then broke
    correct_alone = [r for r in true_cases if r["raw"]["affirmed_gold"]]
    broke = sum(1 for r in correct_alone if not r["gated"]["affirmed_gold"])

    # coverage / recall: hallucinated-alone false cases the gate fixed
    bad_alone = [r for r in false_cases if r["raw"]["hallucinated"]]
    fixed = sum(1 for r in bad_alone if not r["gated"]["hallucinated"])

    halluc_alone_rate = _rate(halluc_alone, len(false_cases))
    halluc_gated_rate = _rate(halluc_gated, len(false_cases))

    return {
        "falseCases": len(false_cases),
        "trueCases": len(true_cases),
        "hallucinationRateAlone": halluc_alone_rate,
        "hallucinationRateGated": halluc_gated_rate,
        "delta": round(halluc_alone_rate - halluc_gated_rate, 4),
        "falsePositiveCost": _rate(broke, len(correct_alone)),
        "falsePositiveDetail": {"correctAlone": len(correct_alone), "brokenByGate": broke},
        "coverageRecall": _rate(fixed, len(bad_alone)),
        "coverageDetail": {"hallucinatedAlone": len(bad_alone), "fixedByGate": fixed},
    }
