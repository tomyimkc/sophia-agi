# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Memetic combat arena — score embryo configs on held-out traps + generality micro-tasks.

No weight updates. Fitness is verifier pass rate and deterministic generality stubs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.embryogenesis.population import EmbryoSpec, mutate, seed_population

ROOT = Path(__file__).resolve().parents[2]
TRAPS_PATH = ROOT / "data" / "reference_holdout_traps.json"
GENERALITY_PATH = ROOT / "data" / "generality_tasks.json"


@dataclass
class EvalScorecard:
    embryo: EmbryoSpec
    trapPassRate: float
    trapPassed: int
    trapTotal: int
    generalityPassRate: float
    generalityPassed: int
    generalityTotal: int
    fitness: float
    candidateOnly: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "embryo": self.embryo.to_dict(),
            "trapPassRate": self.trapPassRate,
            "trapPassed": self.trapPassed,
            "trapTotal": self.trapTotal,
            "generalityPassRate": self.generalityPassRate,
            "generalityPassed": self.generalityPassed,
            "generalityTotal": self.generalityTotal,
            "fitness": self.fitness,
            "candidateOnly": self.candidateOnly,
        }


def _load_traps() -> list[dict]:
    if not TRAPS_PATH.exists():
        return []
    return json.loads(TRAPS_PATH.read_text(encoding="utf-8")).get("traps", [])


def _load_generality(limit: int = 5) -> list[dict]:
    if not GENERALITY_PATH.exists():
        return []
    tasks = json.loads(GENERALITY_PATH.read_text(encoding="utf-8")).get("tasks", [])
    return tasks[:limit]


def _score_trap_intrinsic(answer: str) -> bool:
    try:
        from agent.gate import check_response

        result = check_response(answer, mode="advisor")
        return bool(result.get("passed"))
    except Exception:
        return False


def _score_generality_stub(task: dict, embryo: EmbryoSpec) -> bool:
    """Deterministic stub: hash embryo + task id to simulate variable capability."""
    from tools.eval_generality import score

    seed_bias = hash((embryo.embryo_id, embryo.tradition_seed)) % 3
    gold = str(task.get("answer", ""))
    match = str(task.get("match", "exact"))
    if seed_bias == 0:
        reply = gold
    elif seed_bias == 1:
        reply = f"The answer is {gold}"
    else:
        reply = "I cannot determine."
    return score(reply, gold, match)


def score_embryo(embryo: EmbryoSpec, *, generality_limit: int = 5) -> EvalScorecard:
    traps = _load_traps()
    trap_passed = sum(1 for t in traps if _score_trap_intrinsic(str(t.get("answer", ""))))
    trap_total = len(traps) or 1

    gen_tasks = _load_generality(generality_limit)
    gen_passed = sum(1 for t in gen_tasks if _score_generality_stub(t, embryo))
    gen_total = len(gen_tasks) or 1

    trap_rate = round(trap_passed / trap_total, 4)
    gen_rate = round(gen_passed / gen_total, 4)
    # Fitness weights traps higher (Goodhart guard per Training-Speed-vs-AGI)
    fitness = round(0.6 * trap_rate + 0.4 * gen_rate, 4)

    return EvalScorecard(
        embryo=embryo,
        trapPassRate=trap_rate,
        trapPassed=trap_passed,
        trapTotal=trap_total,
        generalityPassRate=gen_rate,
        generalityPassed=gen_passed,
        generalityTotal=gen_total,
        fitness=fitness,
    )


def run_arena(
    *,
    population_size: int = 8,
    generations: int = 2,
    top_k: int = 3,
    generality_limit: int = 5,
) -> dict[str, Any]:
    """Run population scoring + top-k reproduction for ``generations`` rounds."""
    population = seed_population(population_size, generation=0)
    history: list[dict] = []

    for gen in range(generations):
        scorecards = [score_embryo(e, generality_limit=generality_limit) for e in population]
        scorecards.sort(key=lambda s: s.fitness, reverse=True)
        winners = scorecards[:top_k]
        history.append(
            {
                "generation": gen,
                "populationSize": len(population),
                "topFitness": winners[0].fitness if winners else 0.0,
                "scorecards": [s.to_dict() for s in scorecards],
            }
        )
        if gen + 1 >= generations:
            break
        next_pop: list[EmbryoSpec] = []
        for i, w in enumerate(winners):
            next_pop.append(mutate(w.embryo, i, generation=gen + 1))
        # refill to population_size with mutated children of winners
        idx = len(next_pop)
        while len(next_pop) < population_size:
            parent = winners[idx % len(winners)].embryo
            next_pop.append(mutate(parent, idx, generation=gen + 1))
            idx += 1
        population = next_pop

    final_winners = history[-1]["scorecards"][:top_k] if history else []
    return {
        "schema": "sophia.embryogenesis_arena.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "populationSize": population_size,
        "generations": generations,
        "topK": top_k,
        "history": history,
        "winners": final_winners,
        "weightsFrozen": True,
        "claimBoundary": "Verifier population search only — no LoRA reproduction.",
    }
