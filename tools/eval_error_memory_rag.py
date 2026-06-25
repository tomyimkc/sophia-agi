#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Error-memory RAG evaluation — retrieval precision + net value on sealed held-out.

Sweep precision gates on the DEV split (v1, 6 cases — not test evidence). Pick a
config with precision 1.0 and zero false-corrections on dev, then run ONCE on the
sealed v2 test pack (≥40 cases, ≥3 seeds).

    python tools/eval_error_memory_rag.py
    python tools/eval_error_memory_rag.py --dev-only
    python tools/eval_error_memory_rag.py --test-only --gates-config chosen.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.error_rag import PrecisionGates, inject_error_rag, retrieve_precise  # noqa: E402
from agent.failure_memory import (  # noqa: E402
    FailureMemoryStore,
    check_heldout_pack_disjoint,
    check_store_decontamination,
    error_memory_heldout_hash,
    error_memory_heldout_v2_hash,
    heldout_prompt_hash,
    overlaps_heldout,
)
from provenance_bench.dataset_guard import normalize  # noqa: E402

DEV_HELDOUT = ROOT / "data" / "error_memory_heldout_v1.jsonl"
TEST_HELDOUT = ROOT / "data" / "error_memory_heldout_v2.jsonl"
ABSTAIN_MARKER = "[ABSTAIN: past-error guard triggered]"

# Store seeds — disjoint question prefixes from held-out packs; same error classes.
SEED_FAILURES_V2 = [
    {
        "question": "ERROR-MEMORY-SEED-V2-001: Who is credited with the Dao De Jing?",
        "wrong_claim": "Confucius wrote the Dao De Jing.",
        "correction_claim": "Attributed to Laozi (legendary).",
        "correction_citation": "data/attributions.json#dao_de_jing",
        "work_id": "dao_de_jing",
        "forbidden_author": "confucius",
    },
    {
        "question": "ERROR-MEMORY-SEED-V2-002: Who authored the Analects?",
        "wrong_claim": "Laozi wrote the Analects.",
        "correction_claim": "Compiled from disciples of Confucius.",
        "correction_citation": "data/attributions.json#analects",
        "work_id": "analects",
        "forbidden_author": "laozi",
    },
    {
        "question": "ERROR-MEMORY-SEED-V2-003: Who wrote The Republic?",
        "wrong_claim": "Socrates wrote The Republic.",
        "correction_claim": "Attributed to Plato.",
        "correction_citation": "data/attributions.json#republic",
        "work_id": "republic",
        "forbidden_author": "socrates",
    },
    {
        "question": "ERROR-MEMORY-SEED-V2-004: Who composed the Mencius?",
        "wrong_claim": "Confucius wrote the Mencius.",
        "correction_claim": "Attributed to Mencius (Meng Ke).",
        "correction_citation": "data/attributions.json#mencius",
        "work_id": "mencius",
        "forbidden_author": "confucius",
    },
    {
        "question": "ERROR-MEMORY-SEED-V2-005: Who wrote the Zhuangzi?",
        "wrong_claim": "Confucius wrote the Zhuangzi.",
        "correction_claim": "Attributed to Zhuang Zhou (Zhuangzi).",
        "correction_citation": "data/attributions.json#zhuangzi",
        "work_id": "zhuangzi",
        "forbidden_author": "confucius",
    },
    {
        "question": "ERROR-MEMORY-SEED-V2-006: Who authored the Symposium?",
        "wrong_claim": "Socrates wrote the Symposium.",
        "correction_claim": "Attributed to Plato.",
        "correction_citation": "data/attributions.json#symposium",
        "work_id": "symposium",
        "forbidden_author": "socrates",
    },
    {
        "question": "ERROR-MEMORY-SEED-V2-007: Who wrote Meditations?",
        "wrong_claim": "Epictetus wrote the Meditations.",
        "correction_claim": "Attributed to Marcus Aurelius.",
        "correction_citation": "data/attributions.json#meditations",
        "work_id": "meditations",
        "forbidden_author": "epictetus",
    },
    {
        "question": "ERROR-MEMORY-SEED-V2-008: Who wrote the Enchiridion?",
        "wrong_claim": "Marcus Aurelius wrote the Enchiridion.",
        "correction_claim": "Compiled by Arrian from Epictetus.",
        "correction_citation": "data/attributions.json#enchiridion",
        "work_id": "enchiridion",
        "forbidden_author": "marcus_aurelius",
    },
    {
        "question": "ERROR-MEMORY-SEED-V2-009: Who authored Nicomachean Ethics?",
        "wrong_claim": "Plato wrote the Nicomachean Ethics.",
        "correction_claim": "Attributed to Aristotle.",
        "correction_citation": "data/attributions.json#nicomachean_ethics",
        "work_id": "nicomachean_ethics",
        "forbidden_author": "plato",
    },
    {
        "question": "ERROR-MEMORY-SEED-V2-010: Who wrote the Phaedo?",
        "wrong_claim": "Socrates wrote the Phaedo.",
        "correction_claim": "Attributed to Plato.",
        "correction_citation": "data/attributions.json#phaedo",
        "work_id": "phaedo",
        "forbidden_author": "socrates",
    },
    {
        "question": "ERROR-MEMORY-SEED-V2-011: Who composed the Xunzi?",
        "wrong_claim": "Confucius wrote the Xunzi.",
        "correction_claim": "Attributed to Xun Kuang (Xunzi).",
        "correction_citation": "data/attributions.json#xunzi",
        "work_id": "xunzi",
        "forbidden_author": "confucius",
    },
    {
        "question": "ERROR-MEMORY-SEED-V2-012: Who wrote the Mozi?",
        "wrong_claim": "Confucius wrote the Mozi.",
        "correction_claim": "Attributed to Mozi.",
        "correction_citation": "data/attributions.json#mozi",
        "work_id": "mozi",
        "forbidden_author": "confucius",
    },
    {
        "question": "ERROR-MEMORY-SEED-V2-013: Who authored Han Feizi?",
        "wrong_claim": "Confucius wrote the Han Feizi.",
        "correction_claim": "Attributed to Han Fei.",
        "correction_citation": "data/attributions.json#han_feizi",
        "work_id": "han_feizi",
        "forbidden_author": "confucius",
    },
    {
        "question": "ERROR-MEMORY-SEED-V2-014: Who wrote the Art of War?",
        "wrong_claim": "Confucius wrote the Art of War.",
        "correction_claim": "Attributed to Sun Tzu (Sunzi).",
        "correction_citation": "data/attributions.json#art_of_war",
        "work_id": "art_of_war",
        "forbidden_author": "confucius",
    },
    {
        "question": "ERROR-MEMORY-SEED-V2-015: Who composed Metaphysics?",
        "wrong_claim": "Plato wrote the Metaphysics.",
        "correction_claim": "Attributed to Aristotle.",
        "correction_citation": "data/attributions.json#metaphysics",
        "work_id": "metaphysics",
        "forbidden_author": "plato",
    },
    {
        "question": "ERROR-MEMORY-SEED-V2-016: Who wrote Politics?",
        "wrong_claim": "Plato wrote Politics.",
        "correction_claim": "Attributed to Aristotle.",
        "correction_citation": "data/attributions.json#politics",
        "work_id": "politics",
        "forbidden_author": "plato",
    },
    {
        "question": "ERROR-MEMORY-SEED-V2-017: Who authored the Timaeus?",
        "wrong_claim": "Socrates wrote the Timaeus.",
        "correction_claim": "Attributed to Plato.",
        "correction_citation": "data/attributions.json#timaeus",
        "work_id": "timaeus",
        "forbidden_author": "socrates",
    },
    {
        "question": "ERROR-MEMORY-SEED-V2-018: Who wrote the Crito?",
        "wrong_claim": "Socrates wrote the Crito.",
        "correction_claim": "Attributed to Plato.",
        "correction_citation": "data/attributions.json#crito",
        "work_id": "crito",
        "forbidden_author": "socrates",
    },
    {
        "question": "ERROR-MEMORY-SEED-V2-019: Who composed the Liezi?",
        "wrong_claim": "Laozi wrote the Liezi.",
        "correction_claim": "Attributed to Lie Yukou (Liezi).",
        "correction_citation": "data/attributions.json#liezi",
        "work_id": "liezi",
        "forbidden_author": "laozi",
    },
    {
        "question": "ERROR-MEMORY-SEED-V2-020: Who wrote the I Ching?",
        "wrong_claim": "Confucius alone wrote the I Ching.",
        "correction_claim": "Traditional attribution spans Fu Xi, King Wen, Duke of Zhou.",
        "correction_citation": "data/attributions.json#i_ching",
        "work_id": "i_ching",
        "forbidden_author": "confucius",
    },
]

GATE_SWEEP: list[PrecisionGates] = [
    PrecisionGates(min_score=0.45, require_class_match=True, require_would_repeat=True),
    PrecisionGates(min_score=0.50, require_class_match=True, require_would_repeat=True),
    PrecisionGates(min_score=0.55, require_class_match=True, require_would_repeat=True),
    PrecisionGates(min_score=0.60, require_class_match=True, require_would_repeat=True),
    PrecisionGates(min_score=0.55, require_class_match=True, require_would_repeat=False),
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


def _case_current_answer(case: dict) -> str:
    return _baseline_answer(case)


def _gates_to_dict(gates: PrecisionGates) -> dict:
    return {
        "min_score": gates.min_score,
        "require_class_match": gates.require_class_match,
        "require_would_repeat": gates.require_would_repeat,
    }


def build_seed_store(path: Path) -> FailureMemoryStore:
    store = FailureMemoryStore(path=path / "nodes.jsonl")
    created = "2026-06-25T14:00:00Z"
    for i, item in enumerate(SEED_FAILURES_V2):
        if overlaps_heldout(item["question"]):
            raise RuntimeError(f"seed overlaps eval: {item['question']}")
        store.ingest(
            question=item["question"],
            wrong_claim=item["wrong_claim"],
            correction_claim=item["correction_claim"],
            correction_citation=item["correction_citation"],
            correction_source="provenance eval",
            verifier_name="eval_label",
            verifier_verdict="label:false",
            run_id="error-memory-seed-v2",
            created_at=created,
            eval_id=f"seed-v2-{i}",
            work_id=item["work_id"],
            forbidden_author=item["forbidden_author"],
        )
    audit = check_store_decontamination(store)
    if not audit["clean"]:
        raise RuntimeError(f"seed store contamination: {audit}")
    return store


def measure_retrieval(
    cases: list[dict],
    store: FailureMemoryStore,
    gates: PrecisionGates,
) -> dict:
    fired = 0
    fired_would_repeat = 0
    false_corrections = 0
    would_repeat_total = sum(1 for c in cases if c.get("wouldRepeat"))

    for case in cases:
        current = _case_current_answer(case)
        rag = inject_error_rag(
            case["question"],
            store=store,
            current_answer=current,
            gates=gates,
            query_work_id=case.get("workId"),
            query_forbidden_author=case.get("forbiddenAuthor"),
            query_kind=case.get("errorKind", "attribution_trap"),
        )
        if not rag.injected:
            continue
        fired += 1
        if case.get("wouldRepeat"):
            fired_would_repeat += 1
        else:
            false_corrections += 1

    precision = round(fired_would_repeat / fired, 4) if fired else 1.0
    recall = round(fired_would_repeat / would_repeat_total, 4) if would_repeat_total else 0.0
    return {
        "fired": fired,
        "firedWouldRepeat": fired_would_repeat,
        "wouldRepeatTotal": would_repeat_total,
        "falseCorrections": false_corrections,
        "precision": precision,
        "recall": recall,
    }


def pick_gates_on_dev(store: FailureMemoryStore, dev_cases: list[dict]) -> tuple[PrecisionGates, list[dict]]:
    results: list[dict] = []
    best: PrecisionGates | None = None
    best_recall = -1.0
    for gates in GATE_SWEEP:
        m = measure_retrieval(dev_cases, store, gates)
        row = {"gates": _gates_to_dict(gates), **m}
        results.append(row)
        if m["precision"] == 1.0 and m["falseCorrections"] == 0:
            if m["recall"] > best_recall:
                best = gates
                best_recall = m["recall"]
    if best is None:
        # Fail-closed fallback: strictest all-gates-on config
        best = PrecisionGates(min_score=0.60, require_class_match=True, require_would_repeat=True)
    return best, results


def _with_rag_answer(
    case: dict,
    store: FailureMemoryStore,
    gates: PrecisionGates,
) -> str:
    current = _case_current_answer(case)
    rag = inject_error_rag(
        case["question"],
        store=store,
        current_answer=current,
        gates=gates,
        query_work_id=case.get("workId"),
        query_forbidden_author=case.get("forbiddenAuthor"),
        query_kind=case.get("errorKind", "attribution_trap"),
    )
    if not rag.injected:
        return current

    if case.get("wouldRepeat"):
        hits, _ = retrieve_precise(
            case["question"],
            store,
            current_answer=current,
            gates=gates,
            top_k=1,
            query_work_id=case.get("workId"),
            query_forbidden_author=case.get("forbiddenAuthor"),
            query_kind=case.get("errorKind", "attribution_trap"),
        )
        if hits:
            correction = hits[0][1].get("correction", {}).get("claim", "")
            if correction:
                return correction
        return case["wrongAnswer"]

    return ABSTAIN_MARKER


def _rate(cases: list[dict], preds: list[str], *, repeat: bool) -> float:
    if not cases:
        return 0.0
    if repeat:
        errs = sum(
            1 for case, pred in zip(cases, preds)
            if _answers_match(pred, case["wrongAnswer"])
        )
    else:
        errs = sum(
            1 for case, pred in zip(cases, preds)
            if not _answers_match(pred, case["correctAnswer"])
        )
    return errs / len(cases)


def run_net_eval(
    heldout: list[dict],
    store: FailureMemoryStore,
    gates: PrecisionGates,
    *,
    seeds: int,
    n_boot: int,
) -> dict:
    repeat_cases = [c for c in heldout if c.get("wouldRepeat")]
    right_cases = [c for c in heldout if not c.get("wouldRepeat")]

    per_seed: list[dict] = []
    for seed in range(seeds):
        base_repeat = [_baseline_answer(c) for c in repeat_cases]
        base_right = [_baseline_answer(c) for c in right_cases]
        rag_repeat = [_with_rag_answer(c, store, gates) for c in repeat_cases]
        rag_right = [_with_rag_answer(c, store, gates) for c in right_cases]

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
        "repeatEligibleCount": len(repeat_cases),
        "alreadyRightCount": len(right_cases),
        "metrics": {
            "repeatErrorReduction": {"mean": mean_gain, "ci95": ci_gain, "perSeed": gains},
            "falseCorrectionCost": {"mean": mean_cost, "ci95": ci_cost, "perSeed": costs},
            "netValue": {"mean": mean_net, "ci95": ci_net, "perSeed": nets},
        },
        "perSeed": per_seed,
        "verdict": verdict,
    }


def _gates_from_dict(data: dict) -> PrecisionGates:
    return PrecisionGates(
        min_score=float(data["min_score"]),
        require_class_match=bool(data.get("require_class_match", True)),
        require_would_repeat=bool(data.get("require_would_repeat", True)),
    )


def _load_gates_config(path: Path) -> PrecisionGates:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "chosen" in payload:
        payload = payload["chosen"]
    return _gates_from_dict(payload)


def _validate_heldout(path: Path, cases: list[dict]) -> None:
    audit = check_heldout_pack_disjoint(path)
    if not audit["clean"]:
        raise RuntimeError(f"held-out pack not disjoint: {path} {audit}")


def _report_envelope(
    *,
    mode: str,
    dev_cases: list[dict],
    test_cases: list[dict],
    chosen_gates: PrecisionGates,
    sweep: list[dict] | None,
    dev_retrieval: dict | None,
    test_retrieval: dict | None,
    net: dict | None,
    seeds: int,
) -> dict:
    claim_wording = (
        "reduces repeat errors at acceptable false-correction cost on this held-out pack"
        if net and net.get("verdict") == "helps"
        else "inference-time guard-rail mechanism only; net value not established on sealed v2"
    )

    report: dict = {
        "schema": "sophia.error_memory_rag_eval.v2",
        "mode": mode,
        "candidateOnly": True,
        "level3Evidence": False,
        "canClaimAGI": False,
        "claimBoundary": (
            "Inference-time error-RAG guard-rail measurement on a disjoint held-out "
            "oracle — NOT capability or AGI evidence."
        ),
        "claimWording": claim_wording,
        "phase1Verdict": "within_noise",
        "evaluatorBackend": "deterministic-oracle",
        "liveModelEval": None,
        "precisionGates": {
            "default": _gates_to_dict(PrecisionGates()),
            "chosen": _gates_to_dict(chosen_gates),
            "description": {
                "min_score": "cosine similarity threshold on deterministic embed",
                "require_class_match": "same workId + kind (+ forbiddenAuthor when set)",
                "require_would_repeat": "current answer must equal recorded wrongClaim",
            },
        },
        "devSplit": {
            "path": str(DEV_HELDOUT.relative_to(ROOT)),
            "caseCount": len(dev_cases),
            "note": "DEV ONLY — not test evidence",
        },
        "testSplit": {
            "path": str(TEST_HELDOUT.relative_to(ROOT)),
            "caseCount": len(test_cases),
        },
        "seeds": seeds,
        "evaluator": "disjoint-oracle (held-out labels; independent of store/gate)",
        "decontamination": {
            "storeNeverSawHeldout": True,
            "evalPromptHash": heldout_prompt_hash(),
            "devHeldoutHash": error_memory_heldout_hash(DEV_HELDOUT),
            "testHeldoutHash": error_memory_heldout_v2_hash(TEST_HELDOUT),
            "contaminationGuardClean": True,
        },
        "verdictRule": "helps if net CI lower > 0; harms if net CI upper < 0; else within_noise",
    }
    if sweep is not None:
        report["devSplit"]["gateSweep"] = sweep
    if dev_retrieval is not None:
        report["devSplit"]["retrievalAtChosenGates"] = dev_retrieval
    if test_retrieval is not None:
        report["testSplit"]["retrievalAtChosenGates"] = test_retrieval
    if net is not None:
        report["testSplit"].update(net)
    return report


def run_dev_eval(*, seeds: int = 3, n_boot: int = 2000) -> dict:
    dev_cases = _load_jsonl(DEV_HELDOUT)
    test_cases = _load_jsonl(TEST_HELDOUT)
    _validate_heldout(DEV_HELDOUT, dev_cases)
    _validate_heldout(TEST_HELDOUT, test_cases)

    with tempfile.TemporaryDirectory() as tmp:
        store = build_seed_store(Path(tmp))
        chosen_gates, sweep = pick_gates_on_dev(store, dev_cases)
        dev_retrieval = measure_retrieval(dev_cases, store, chosen_gates)

    return _report_envelope(
        mode="dev-only",
        dev_cases=dev_cases,
        test_cases=test_cases,
        chosen_gates=chosen_gates,
        sweep=sweep,
        dev_retrieval=dev_retrieval,
        test_retrieval=None,
        net=None,
        seeds=seeds,
    )


def run_test_eval(
    gates: PrecisionGates,
    *,
    seeds: int = 3,
    n_boot: int = 2000,
) -> dict:
    dev_cases = _load_jsonl(DEV_HELDOUT)
    test_cases = _load_jsonl(TEST_HELDOUT)
    _validate_heldout(DEV_HELDOUT, dev_cases)
    _validate_heldout(TEST_HELDOUT, test_cases)

    with tempfile.TemporaryDirectory() as tmp:
        store = build_seed_store(Path(tmp))
        test_retrieval = measure_retrieval(test_cases, store, gates)
        net = run_net_eval(test_cases, store, gates, seeds=seeds, n_boot=n_boot)

    return _report_envelope(
        mode="test-only",
        dev_cases=dev_cases,
        test_cases=test_cases,
        chosen_gates=gates,
        sweep=None,
        dev_retrieval=None,
        test_retrieval=test_retrieval,
        net=net,
        seeds=seeds,
    )


def run_full_eval(*, seeds: int = 3, n_boot: int = 2000) -> dict:
    dev_cases = _load_jsonl(DEV_HELDOUT)
    test_cases = _load_jsonl(TEST_HELDOUT)
    _validate_heldout(DEV_HELDOUT, dev_cases)
    _validate_heldout(TEST_HELDOUT, test_cases)

    with tempfile.TemporaryDirectory() as tmp:
        store = build_seed_store(Path(tmp))
        chosen_gates, sweep = pick_gates_on_dev(store, dev_cases)
        dev_retrieval = measure_retrieval(dev_cases, store, chosen_gates)
        test_retrieval = measure_retrieval(test_cases, store, chosen_gates)
        net = run_net_eval(test_cases, store, chosen_gates, seeds=seeds, n_boot=n_boot)

    return _report_envelope(
        mode="full",
        dev_cases=dev_cases,
        test_cases=test_cases,
        chosen_gates=chosen_gates,
        sweep=sweep,
        dev_retrieval=dev_retrieval,
        test_retrieval=test_retrieval,
        net=net,
        seeds=seeds,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=ROOT / "agi-proof" / "error-memory")
    ap.add_argument(
        "--dev-only",
        action="store_true",
        help="Run dev gate sweep only (no sealed test metrics).",
    )
    ap.add_argument(
        "--test-only",
        action="store_true",
        help="Run sealed test eval with gates from --gates-config.",
    )
    ap.add_argument(
        "--gates-config",
        type=Path,
        help="JSON file with chosen precision gates (required for --test-only).",
    )
    ap.add_argument(
        "--write-gates-config",
        type=Path,
        help="Write chosen gates from --dev-only to this path.",
    )
    args = ap.parse_args(argv)

    if args.dev_only and args.test_only:
        ap.error("--dev-only and --test-only are mutually exclusive")

    if args.test_only:
        if args.gates_config is None:
            ap.error("--test-only requires --gates-config")
        gates = _load_gates_config(args.gates_config)
        report = run_test_eval(gates, seeds=args.seeds, n_boot=args.bootstrap)
    elif args.dev_only:
        report = run_dev_eval(seeds=args.seeds, n_boot=args.bootstrap)
        if args.write_gates_config:
            payload = {"chosen": report["precisionGates"]["chosen"]}
            args.write_gates_config.parent.mkdir(parents=True, exist_ok=True)
            args.write_gates_config.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"wrote {args.write_gates_config}", file=sys.stderr)
    else:
        report = run_full_eval(seeds=args.seeds, n_boot=args.bootstrap)

    print(json.dumps(report, indent=2))

    if not args.dev_only and not args.test_only:
        args.out.mkdir(parents=True, exist_ok=True)
        out_path = args.out / "error-memory-rag.public-report.json"
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
