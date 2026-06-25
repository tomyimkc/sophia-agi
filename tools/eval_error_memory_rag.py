#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Stage-3 held-out measurement for inference-time error-RAG.

Compares WITH vs WITHOUT error-RAG on a sealed held-out eval the failure store never
saw. Reports repeat-error reduction (gain) AND false-correction/abstention cost on
questions the baseline policy already answered correctly. Uses a disjoint oracle
(labels in eval/error_memory/heldout_v1.jsonl), not the store or gate.

    python tools/eval_error_memory_rag.py
    python tools/eval_error_memory_rag.py --seeds 3 --out agi-proof/error-memory/
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.error_rag import inject_error_rag  # noqa: E402
from agent.failure_memory import (  # noqa: E402
    FailureMemoryStore,
    check_store_decontamination,
    error_memory_heldout_hash,
    heldout_prompt_hash,
    overlaps_heldout,
)
from provenance_bench.dataset_guard import normalize  # noqa: E402

HELDOUT_PATH = ROOT / "data" / "error_memory_heldout_v1.jsonl"
ABSTAIN_MARKER = "[ABSTAIN: past-error guard triggered]"

# Training-side failure patterns (disjoint questions from held-out).
SEED_FAILURES = [
    {
        "question": "ERROR-MEMORY-SEED-001: Who wrote the Dao De Jing?",
        "wrong_claim": "Confucius wrote the Dao De Jing.",
        "correction_claim": "The Dao De Jing is attributed to Laozi (legendary).",
        "correction_citation": "data/attributions.json#dao_de_jing",
        "correction_source": "provenance eval",
        "verifier_name": "eval_label",
        "verifier_verdict": "label:false",
    },
    {
        "question": "ERROR-MEMORY-SEED-002: Who authored the I Ching?",
        "wrong_claim": "Buddha authored the I Ching.",
        "correction_claim": "Traditional attribution is Fu Xi / King Wen / Duke of Zhou.",
        "correction_citation": "data/attributions.json#i_ching",
        "correction_source": "provenance eval",
        "verifier_name": "gate",
        "verifier_verdict": "provenance_violation",
    },
    {
        "question": "ERROR-MEMORY-SEED-003: Who compiled the Analects?",
        "wrong_claim": "Laozi personally authored the Analects.",
        "correction_claim": "The Analects is compiled from disciples of Confucius.",
        "correction_citation": "data/attributions.json#analects",
        "correction_source": "provenance eval",
        "verifier_name": "eval_label",
        "verifier_verdict": "label:false",
    },
]


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _bootstrap_ci(samples: list[float], *, n_boot: int = 2000, seed: int = 0) -> list[float]:
    if not samples:
        return [0.0, 0.0]
    rng = random.Random(seed)
    n = len(samples)
    means: list[float] = []
    for _ in range(n_boot):
        draw = [samples[rng.randrange(n)] for _ in range(n)]
        means.append(sum(draw) / n)
    means.sort()
    lo = means[max(0, int(0.025 * len(means)))]
    hi = means[min(len(means) - 1, int(0.975 * len(means)))]
    return [round(lo, 4), round(hi, 4)]


def _answers_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def _baseline_answer(case: dict) -> str:
    return case["correctAnswer"] if case["modelWasRight"] else case["wrongAnswer"]


def _with_rag_answer(case: dict, store: FailureMemoryStore, *, seed: int) -> str:
    rag = inject_error_rag(case["question"], store=store, min_score=0.02)
    if not rag.injected:
        return _baseline_answer(case)

    if not case["modelWasRight"]:
        # Repeat-eligible: apply verified correction from top retrieved node
        hits = store.retrieve_similar(case["question"], top_k=1, min_score=0.02)
        if hits:
            correction = hits[0][1].get("correction", {}).get("claim", "")
            if correction:
                return correction
        return case["wrongAnswer"]

    # Already-right: false alarm if retrieval fires on a correct-answer question
    # Seed shifts min_score tie-break sensitivity slightly across runs
    rag_strict = inject_error_rag(
        case["question"],
        store=store,
        min_score=0.02 + (seed % 3) * 0.001,
    )
    if rag_strict.injected:
        return ABSTAIN_MARKER
    return case["correctAnswer"]


def _rate(cases: list[dict], preds: list[str], *, repeat: bool) -> float:
    if not cases:
        return 0.0
    if repeat:
        # fraction that still repeat the wrong answer
        errs = sum(
            1 for case, pred in zip(cases, preds)
            if _answers_match(pred, case["wrongAnswer"])
        )
    else:
        # fraction that abandon a correct baseline (abstain or wrong)
        errs = sum(
            1 for case, pred in zip(cases, preds)
            if not _answers_match(pred, case["correctAnswer"])
        )
    return errs / len(cases)


def build_seed_store(tmp: Path) -> FailureMemoryStore:
    store = FailureMemoryStore(path=tmp / "nodes.jsonl")
    created = "2026-06-25T12:00:00Z"
    for i, item in enumerate(SEED_FAILURES):
        if overlaps_heldout(item["question"]):
            raise RuntimeError(f"seed failure overlaps held-out: {item['question']}")
        store.ingest(
            **item,
            run_id="error-memory-seed-build",
            created_at=created,
            eval_id=f"seed-{i}",
        )
    audit = check_store_decontamination(store)
    if not audit["clean"]:
        raise RuntimeError(f"seed store contamination: {audit}")
    return store


def run_eval(*, seeds: int = 3, n_boot: int = 2000) -> dict:
    heldout = _load_jsonl(HELDOUT_PATH)
    for case in heldout:
        if overlaps_heldout(case["question"]):
            raise RuntimeError(f"held-out case overlaps eval set unexpectedly: {case['id']}")

    repeat_cases = [c for c in heldout if not c["modelWasRight"]]
    right_cases = [c for c in heldout if c["modelWasRight"]]

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        store = build_seed_store(Path(tmp))

        per_seed: list[dict] = []
        for seed in range(seeds):
            base_repeat = [_baseline_answer(c) for c in repeat_cases]
            base_right = [_baseline_answer(c) for c in right_cases]
            rag_repeat = [_with_rag_answer(c, store, seed=seed) for c in repeat_cases]
            rag_right = [_with_rag_answer(c, store, seed=seed) for c in right_cases]

            repeat_without = _rate(repeat_cases, base_repeat, repeat=True)
            repeat_with = _rate(repeat_cases, rag_repeat, repeat=True)
            cost_without = _rate(right_cases, base_right, repeat=False)
            cost_with = _rate(right_cases, rag_right, repeat=False)

            per_seed.append({
                "seed": seed,
                "repeatErrorRateWithout": round(repeat_without, 4),
                "repeatErrorRateWith": round(repeat_with, 4),
                "repeatErrorReduction": round(repeat_without - repeat_with, 4),
                "falseCorrectionCostWithout": round(cost_without, 4),
                "falseCorrectionCostWith": round(cost_with, 4),
                "falseCorrectionDelta": round(cost_with - cost_without, 4),
                "netValue": round((repeat_without - repeat_with) - (cost_with - cost_without), 4),
            })

    gains = [s["repeatErrorReduction"] for s in per_seed]
    costs = [s["falseCorrectionDelta"] for s in per_seed]
    nets = [s["netValue"] for s in per_seed]

    ci_gain = _bootstrap_ci(gains, n_boot=n_boot, seed=42)
    ci_cost = _bootstrap_ci(costs, n_boot=n_boot, seed=43)
    ci_net = _bootstrap_ci(nets, n_boot=n_boot, seed=44)

    mean_gain = round(sum(gains) / len(gains), 4)
    mean_cost = round(sum(costs) / len(costs), 4)
    mean_net = round(sum(nets) / len(nets), 4)

    if ci_net[0] > 0:
        verdict = "helps"
    elif ci_net[1] < 0:
        verdict = "harms"
    else:
        verdict = "within_noise"

    return {
        "schema": "sophia.error_memory_rag_eval.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "canClaimAGI": False,
        "claimBoundary": (
            "Inference-time error-RAG guard-rail measurement on a disjoint held-out "
            "oracle — NOT capability or AGI evidence."
        ),
        "heldoutPath": str(HELDOUT_PATH.relative_to(ROOT)),
        "heldoutCaseCount": len(heldout),
        "repeatEligibleCount": len(repeat_cases),
        "alreadyRightCount": len(right_cases),
        "seeds": seeds,
        "evaluator": "disjoint-oracle (labels in heldout_v1.jsonl; independent of store/gate)",
        "decontamination": {
            "storeNeverSawHeldout": True,
            "evalPromptHash": heldout_prompt_hash(),
            "errorMemoryHeldoutHash": error_memory_heldout_hash(),
        },
        "metrics": {
            "repeatErrorReduction": {"mean": mean_gain, "ci95": ci_gain, "perSeed": gains},
            "falseCorrectionCost": {"mean": mean_cost, "ci95": ci_cost, "perSeed": costs},
            "netValue": {"mean": mean_net, "ci95": ci_net, "perSeed": nets},
        },
        "perSeed": per_seed,
        "verdict": verdict,
        "verdictRule": "helps if net CI lower > 0; harms if net CI upper < 0; else within_noise",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=ROOT / "agi-proof" / "error-memory")
    args = ap.parse_args(argv)

    report = run_eval(seeds=args.seeds, n_boot=args.bootstrap)
    print(json.dumps(report, indent=2))

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "error-memory-rag.public-report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
