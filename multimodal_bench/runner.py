# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the visual-trap suite and aggregate runs into CI'd, no-overclaim numbers.

Mirrors ``provenance_bench/runner.py`` + ``aggregate.py``: a single run scores
every trap with a (possibly consensus) judge; ``aggregate_runs`` pools N runs and
attaches paired bootstrap 95% CIs plus the validated-headline checklist so no
number is published unless it clears the no-overclaim bar.
"""

from __future__ import annotations

import json
import os
import random

from multimodal_bench import judge as judge_mod

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "visual_traps.json")
SYNTH_PATH = os.path.join(os.path.dirname(__file__), "data", "visual_traps_synth.json")
KAPPA_FLOOR = 0.40  # "moderate" agreement — the minimum for a validated headline


def load_traps(path: str = DATA_PATH) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["traps"]


def load_all_traps() -> list:
    """Hand-authored base traps + verifier-checked synthetic chart/table/doc traps."""
    traps = load_traps(DATA_PATH)
    if os.path.exists(SYNTH_PATH):
        traps = traps + load_traps(SYNTH_PATH)
    return traps


# The physical / 2.5D axes a VLM is weakest on (depth, occlusion, real size,
# distance) + their discrimination controls + the numeric `measure` rows. Used to
# scope a real-VLM run to the physical-understanding pre-registration.
PHYSICAL_CATEGORIES = (
    "depth_order", "depth_control", "occlusion", "occlusion_control",
    "size_illusion", "size_control", "distance", "distance_control", "distance_measure",
)


def filter_by_category(traps: list, categories) -> list:
    """Keep only traps whose ``category`` is in ``categories`` (order preserved)."""
    cats = set(categories)
    return [t for t in traps if t.get("category") in cats]


def run_cases(traps: list, answer_fn, judge_fn=None) -> list:
    """Score each trap: get the model's answer, judge it, record the verdict."""
    judge_fn = judge_fn or judge_mod.lexical_judge
    out = []
    for t in traps:
        verdict = judge_fn(answer_fn(t), t)
        out.append({
            "id": t["id"], "category": t["category"],
            "hallucinated": bool(verdict.hallucinated),
            "affirmed_gold": bool(verdict.affirmed_gold),
            "abstained": bool(verdict.abstained),
            "votes": verdict.votes,
        })
    return out


def _ci(xs: list, alpha: float = 0.05) -> list:
    xs = sorted(xs)
    n = len(xs)
    lo = xs[max(0, int((alpha / 2) * n))]
    hi = xs[min(n - 1, int((1 - alpha / 2) * n))]
    return [round(lo, 4), round(hi, 4)]


def _rate(rows: list, key: str) -> float:
    return sum(int(r[key]) for r in rows) / len(rows) if rows else 0.0


def _distinct_families(judges: "list[str] | None") -> int:
    """Distinct provider families among judge specs ('anthropic:..' -> 'anthropic')."""
    if not judges:
        return 0
    fams = set()
    for j in judges:
        if not j or not j.strip():
            continue
        prov, _, model = j.partition(":")
        prov = prov.strip().lower()
        if prov in ("openrouter", "openai") and "/" in model:
            fams.add(model.strip().split("/", 1)[0].lower())
        else:
            fams.add(prov)
    return len(fams)


def aggregate_runs(runs: list, *, n_boot: int = 2000, seed: int = 0,
                   model_spec: "str | None" = None, judges: "list[str] | None" = None) -> dict:
    """Pool N runs (each from ``run_cases``) into point estimates with bootstrap CIs.

    ``model_spec``/``judges`` feed the validated-headline checklist (real model,
    multi-family judges, kappa floor, >=3 runs, CI computed).
    """
    flat = [r for run in runs for r in run]
    # bootstrap over the pooled per-case hallucination / grounding indicators
    halluc = [int(r["hallucinated"]) for r in flat]
    gold = [int(r["affirmed_gold"]) for r in flat]
    abst = [int(r["abstained"]) for r in flat]

    rng = random.Random(seed)
    n = len(flat)
    boot_h, boot_g = [], []
    if n:
        for _ in range(n_boot):
            idx = [rng.randrange(n) for _ in range(n)]
            boot_h.append(sum(halluc[i] for i in idx) / n)
            boot_g.append(sum(gold[i] for i in idx) / n)

    # per-category breakdown (point estimates only)
    cats: dict = {}
    for r in flat:
        c = cats.setdefault(r["category"], {"n": 0, "hallucinated": 0, "affirmed_gold": 0, "abstained": 0})
        c["n"] += 1
        c["hallucinated"] += int(r["hallucinated"])
        c["affirmed_gold"] += int(r["affirmed_gold"])
        c["abstained"] += int(r["abstained"])
    by_category = {k: {"n": v["n"],
                       "hallucinationRate": round(v["hallucinated"] / v["n"], 4),
                       "groundingRate": round(v["affirmed_gold"] / v["n"], 4),
                       "abstentionRate": round(v["abstained"] / v["n"], 4)} for k, v in sorted(cats.items())}

    vote_lists = [r["votes"] for r in flat if r.get("votes")]
    agreement = judge_mod.percent_agreement(vote_lists)
    ci_h = _ci(boot_h) if boot_h else [0.0, 0.0]

    result = {
        "runs": len(runs),
        "cases": n,
        "hallucinationRate": round(sum(halluc) / n, 4) if n else 0.0,
        "ciHallucination": ci_h,
        "groundingRate": round(sum(gold) / n, 4) if n else 0.0,
        "ciGrounding": _ci(boot_g) if boot_g else [0.0, 0.0],
        "abstentionRate": round(sum(abst) / n, 4) if n else 0.0,
        "byCategory": by_category,
        "judgeAgreement": agreement,
    }
    result["validatedChecks"] = _validated_checks(runs, agreement, model_spec, judges, ci_h)
    result["validated"] = all(result["validatedChecks"].values())
    return result


def _validated_checks(runs, agreement, model_spec, judges, ci_h) -> dict:
    kappa = (agreement or {}).get("meanPairwiseKappa")
    return {
        "notMock": bool(model_spec) and "mock" not in (model_spec or ""),
        "multiFamilyJudges": _distinct_families(judges) >= 2,
        "kappaAboveFloor": kappa is not None and kappa >= KAPPA_FLOOR,
        "atLeast3Runs": len(runs) >= 3,
        "ciComputed": bool(ci_h) and ci_h != [0.0, 0.0],
    }
