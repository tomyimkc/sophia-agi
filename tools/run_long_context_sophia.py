#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline long-context benchmark harness for Sophia.

The default run uses a deterministic synthetic corpus and the shared
``run_case()`` pipeline with the offline ``mock`` backend. Results are candidate
measurements only: the corpus is self-authored, no hidden prompts are published,
and no AGI or model-capability claim is promoted from this report.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from tools.run_hidden_eval_sophia import ABLATION_MODES, RunConfig, SOPHIA_FULL, run_case  # noqa: E402

CARD_SCHEMA = ROOT / "schema" / "context-pack-card-1.0.0.json"
REPORT_DIR = ROOT / "agi-proof" / "long-context"
FULL_CONTEXT_SIZES = [4096, 16384, 65536, 131072]
QUICK_CONTEXT_SIZES = [4096, 16384]
DEPTHS = [0, 25, 50, 75, 100]
NEEDLE_COUNTS = [1, 3]
PASSAGE_TOKENS = 1024
RAW_BASELINE_MODE = "matrix-g0-r0-p0-raw-long-context"
GATED_PACKED_MODE = "matrix-g1-r1-p1-gated-packed"
MATRIX_MODE_ORDER = [
    RAW_BASELINE_MODE,
    "matrix-g0-r0-p1-packing-only",
    "matrix-g0-r1-p0-retrieval-only",
    "matrix-g0-r1-p1-retrieval-packing",
    "matrix-g1-r0-p0-gate-only",
    "matrix-g1-r0-p1-gate-packing",
    "matrix-g1-r1-p0-gate-retrieval",
    GATED_PACKED_MODE,
]
CONTROL_MODE_ORDER = ["control-broken-packer", "control-oracle-packer"]


def parse_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise SystemExit("expected at least one integer value")
    return values


def estimate_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\S+", text))) if text.strip() else 0


def target_indices(passage_count: int, depth_pct: int, needle_count: int) -> list[int]:
    center = round((passage_count - 1) * (depth_pct / 100))
    if needle_count <= 1:
        return [min(max(center, 0), passage_count - 1)]
    start = min(max(center - (needle_count // 2), 0), max(passage_count - needle_count, 0))
    return [start + offset for offset in range(min(needle_count, passage_count))]


def filler_text(case_id: str, passage_index: int, tokens: int) -> str:
    stem = f"background_{case_id}_{passage_index}"
    words = [
        stem,
        "archive",
        "governed",
        "retrieval",
        "memory",
        "proof",
        "council",
        "source",
    ]
    repeated = [words[(passage_index + i) % len(words)] for i in range(max(tokens - 12, 8))]
    return " ".join(repeated)


def build_synthetic_case(
    *,
    context_size: int,
    depth_pct: int,
    needle_count: int,
    seed: int,
    budget_tokens: int,
) -> dict[str, Any]:
    passage_count = max(4, context_size // PASSAGE_TOKENS)
    case_id = f"lc_{context_size}_{depth_pct}_{needle_count}_{seed}"
    answer_tokens = [f"LC_NEEDLE_TRUE_{context_size}_{depth_pct}_{needle_count}_{idx}_{seed}" for idx in range(needle_count)]
    distractor_tokens = [f"LC_NEEDLE_WRONG_{context_size}_{depth_pct}_{idx}_{seed}" for idx in range(2)]
    target_by_index = dict(zip(target_indices(passage_count, depth_pct, needle_count), answer_tokens, strict=False))
    distractor_positions = {
        min(passage_count - 1, max(0, round((passage_count - 1) * (depth_pct / 100)) - 1)): distractor_tokens[0],
        min(passage_count - 1, max(0, round((passage_count - 1) * (depth_pct / 100)) + 1)): distractor_tokens[1],
    }

    passages: list[dict[str, Any]] = []
    for index in range(passage_count):
        passage_id = f"{case_id}_p{index:04d}"
        depth = round((index / max(passage_count - 1, 1)) * 100)
        text = filler_text(case_id, index, PASSAGE_TOKENS)
        relevance = 0.15 + ((index % 7) * 0.01)
        verifier = 0.45
        answer_span_tokens: list[str] = []
        source_id = f"synthetic://{case_id}/{index}"
        if index in target_by_index:
            token = target_by_index[index]
            answer_span_tokens = [token]
            source_id = f"synthetic://{case_id}/ANSWER/{index}"
            text = (
                f"Verified answer-bearing span for {case_id}. "
                f"Entity {case_id} has verified effective date 2037-04-{(index % 27) + 1:02d}. "
                f"The only accepted answer token is {token}. "
                "This passage cites the synthetic source ledger and rejects nearby distractors. "
                f"{text}"
            )
            relevance = 0.90
            verifier = 1.0
        elif index in distractor_positions:
            token = distractor_positions[index]
            source_id = f"synthetic://{case_id}/DISTRACTOR/{index}"
            text = (
                f"Plausible near-duplicate distractor for Entity {case_id}. "
                f"It repeats the same entity but gives the wrong date 2038-04-{(index % 27) + 1:02d} "
                f"and the negated candidate answer token {token}. "
                f"Do not answer with {token}; it conflicts with the verifier. "
                f"{text}"
            )
            relevance = 0.95
            verifier = 0.10
        passages.append(
            {
                "id": passage_id,
                "sourceId": source_id,
                "depthPct": depth,
                "text": text,
                "tokenEstimate": PASSAGE_TOKENS,
                "relevanceScore": relevance,
                "verifierScore": verifier,
                "answerTokens": answer_span_tokens,
            }
        )

    return {
        "id": case_id,
        "domain": "logic",
        "prompt": (
            "Return every verified LC_NEEDLE token from the synthetic long-context corpus. "
            "Ignore LC_DISTRACTOR tokens even when they look relevant."
        ),
        "materials": [],
        "longContextPassages": passages,
        "contextBudgetTokens": budget_tokens,
        "needleDepthPct": depth_pct,
        "contextSizeTokens": context_size,
        "needleCount": needle_count,
        "answerTokens": answer_tokens,
        "distractorTokens": distractor_tokens,
        "scoring": {
            "maxPoints": float(max(needle_count, 1)),
            "mustInclude": answer_tokens,
            "mustAvoid": distractor_tokens,
            "rubric": [
                "Recover only answer-bearing synthetic needles included by the context packer.",
                "Avoid plausible-but-wrong distractor tokens.",
            ],
        },
    }


def validate_context_pack_card(card: dict[str, Any], schema_path: Path = CARD_SCHEMA) -> list[str]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for key in schema.get("required", []):
        if key not in card:
            errors.append(f"missing required field: {key}")
    if card.get("schema") != "sophia.context_pack_card.v1":
        errors.append("schema mismatch")
    if card.get("schemaVersion") != "1.0.0":
        errors.append("schemaVersion mismatch")
    if card.get("status") != "candidate" or card.get("claimStatus") != "candidate":
        errors.append("card must be candidate/candidate")
    if card.get("candidateOnly") is not True or card.get("canClaimAGI") is not False:
        errors.append("candidateOnly=true and canClaimAGI=false are required")
    if not isinstance(card.get("budgetTokens"), int) or card.get("budgetTokens", 0) <= 0:
        errors.append("budgetTokens must be a positive integer")
    if card.get("budget_tokens") != card.get("budgetTokens"):
        errors.append("budget_tokens must mirror budgetTokens")
    if not isinstance(card.get("tokens_used"), int) or card.get("tokens_used", -1) < 0:
        errors.append("tokens_used must be a non-negative integer")
    if card.get("tokensUsed") != card.get("tokens_used"):
        errors.append("tokensUsed must mirror tokens_used")
    if not isinstance(card.get("tokens_of_answer_span"), int) or card.get("tokens_of_answer_span", -1) < 0:
        errors.append("tokens_of_answer_span must be a non-negative integer")
    if card.get("tokensOfAnswerSpan") != card.get("tokens_of_answer_span"):
        errors.append("tokensOfAnswerSpan must mirror tokens_of_answer_span")
    if not isinstance(card.get("answer_span_present_in_corpus"), bool):
        errors.append("answer_span_present_in_corpus must be boolean")
    if not isinstance(card.get("answer_span_present_in_pack"), bool):
        errors.append("answer_span_present_in_pack must be boolean")
    if card.get("answer_span_present_in_pack") != card.get("answerBearingSpanIncluded"):
        errors.append("answer_span_present_in_pack must mirror answerBearingSpanIncluded")
    needle_position = card.get("needle_position")
    if not isinstance(needle_position, dict):
        errors.append("needle_position must be an object")
    else:
        if not isinstance(needle_position.get("passage_indexes"), list):
            errors.append("needle_position.passage_indexes must be a list")
        if not isinstance(needle_position.get("depth_pct"), int):
            errors.append("needle_position.depth_pct must be an integer")
    considered = card.get("candidatePassagesConsidered")
    if not isinstance(considered, list) or not considered:
        errors.append("candidatePassagesConsidered must be a non-empty list")
    else:
        for index, item in enumerate(considered):
            prefix = f"candidatePassagesConsidered[{index}]"
            for key in (
                "passageId",
                "sourceId",
                "depthPct",
                "tokenEstimate",
                "relevanceScore",
                "verifierScore",
                "containsAnswerBearingSpan",
                "selected",
                "selectionReason",
            ):
                if key not in item:
                    errors.append(f"{prefix}: missing {key}")
            if not isinstance(item.get("selected"), bool):
                errors.append(f"{prefix}: selected must be boolean")
            if not isinstance(item.get("containsAnswerBearingSpan"), bool):
                errors.append(f"{prefix}: containsAnswerBearingSpan must be boolean")
    candidates_considered = card.get("candidates_considered")
    if not isinstance(candidates_considered, list) or not candidates_considered:
        errors.append("candidates_considered must be a non-empty list")
    else:
        for index, item in enumerate(candidates_considered):
            prefix = f"candidates_considered[{index}]"
            for key in (
                "passage_id",
                "relevance_score",
                "verifier_score",
                "included",
                "eviction_reason",
            ):
                if key not in item:
                    errors.append(f"{prefix}: missing {key}")
            if not isinstance(item.get("included"), bool):
                errors.append(f"{prefix}: included must be boolean")
            if item.get("eviction_reason") not in {"budget", "low_score", "dedup", None}:
                errors.append(f"{prefix}: eviction_reason must be budget|low_score|dedup|null")
    packed = card.get("packedPassages")
    if not isinstance(packed, list):
        errors.append("packedPassages must be a list")
    if not isinstance(card.get("answerBearingSpanIncluded"), bool):
        errors.append("answerBearingSpanIncluded must be boolean")
    flags = card.get("ablationFlags")
    if not isinstance(flags, dict):
        errors.append("ablationFlags must be an object")
    else:
        for flag in ("use_kb", "use_gate", "use_context_packing"):
            if not isinstance(flags.get(flag), bool):
                errors.append(f"ablationFlags.{flag} must be boolean")
        if not isinstance(flags.get("context_packing_policy"), str):
            errors.append("ablationFlags.context_packing_policy must be a string")
    return errors


def long_context_modes(raw_modes: str) -> dict[str, Any]:
    raw_baseline = replace(
        SOPHIA_FULL,
        label=RAW_BASELINE_MODE,
        raw_system=True,
        use_kb=False,
        use_evidence=False,
        use_council=False,
        use_gate=False,
        use_memory=False,
        use_tools=False,
        allow_repair=False,
        use_context_packing=False,
        use_intake=False,
    )
    matrix_modes: dict[str, Any] = {RAW_BASELINE_MODE: raw_baseline}
    matrix_specs = [
        ("matrix-g0-r0-p1-packing-only", False, False, True),
        ("matrix-g0-r1-p0-retrieval-only", False, True, False),
        ("matrix-g0-r1-p1-retrieval-packing", False, True, True),
        ("matrix-g1-r0-p0-gate-only", True, False, False),
        ("matrix-g1-r0-p1-gate-packing", True, False, True),
        ("matrix-g1-r1-p0-gate-retrieval", True, True, False),
        (GATED_PACKED_MODE, True, True, True),
    ]
    for label, use_gate, use_kb, use_context_packing in matrix_specs:
        matrix_modes[label] = replace(
            raw_baseline,
            label=label,
            use_gate=use_gate,
            use_kb=use_kb,
            use_context_packing=use_context_packing,
            context_packing_policy="score",
        )
    controls = {
        "control-broken-packer": replace(
            raw_baseline,
            label="control-broken-packer",
            use_gate=True,
            use_kb=True,
            use_context_packing=True,
            context_packing_policy="broken",
        ),
        "control-oracle-packer": replace(
            raw_baseline,
            label="control-oracle-packer",
            use_gate=True,
            use_kb=True,
            use_context_packing=True,
            context_packing_policy="oracle",
        ),
    }
    available = {
        **matrix_modes,
        **controls,
        "raw-long-context": raw_baseline,
        "sophia-full": SOPHIA_FULL,
        "sophia-no-kb": ABLATION_MODES["sophia-no-kb"],
        "sophia-no-gate": ABLATION_MODES["sophia-no-gate"],
        "sophia-no-context-packing": replace(
            SOPHIA_FULL,
            label="sophia-no-context-packing",
            use_context_packing=False,
        ),
    }
    if raw_modes.strip().lower() == "all":
        return {mode: available[mode] for mode in [*MATRIX_MODE_ORDER, *CONTROL_MODE_ORDER]}
    wanted = [mode.strip() for mode in raw_modes.split(",") if mode.strip()]
    unknown = [mode for mode in wanted if mode not in available]
    if unknown:
        raise SystemExit(f"unknown mode(s): {', '.join(unknown)}; valid: {', '.join(available)}")
    return {mode: available[mode] for mode in wanted}


def run_matrix(
    *,
    context_sizes: list[int],
    depths: list[int],
    needle_counts: list[int],
    modes: dict[str, Any],
    seed: int,
    budget_tokens: int,
) -> dict[str, Any]:
    config = RunConfig(backend="mock", timeout_sec=5)
    pack_id = f"long-context-synthetic-seed-{seed}"
    case_results: list[dict[str, Any]] = []
    card_errors: dict[str, list[str]] = {}

    for context_size in context_sizes:
        for depth in depths:
            for needle_count in needle_counts:
                case = build_synthetic_case(
                    context_size=context_size,
                    depth_pct=depth,
                    needle_count=needle_count,
                    seed=seed,
                    budget_tokens=budget_tokens,
                )
                for mode_name, ablation in modes.items():
                    started = time.time()
                    result = run_case(case, pack_id, config=config, ablation=ablation)
                    wall_time = round(time.time() - started, 6)
                    answer = result["answer"]
                    recalled = [token for token in case["answerTokens"] if token in answer]
                    wrong = [token for token in case["distractorTokens"] if token in answer]
                    card = result.get("contextPackCard", {})
                    errors = validate_context_pack_card(card)
                    if errors:
                        card_errors[f"{case['id']}:{mode_name}"] = errors
                    selected_distractors = [
                        item["passageId"]
                        for item in card.get("candidatePassagesConsidered", [])
                        if item.get("selected") and "DISTRACTOR" in str(item.get("sourceId", ""))
                    ]
                    row = {
                        "caseId": case["id"],
                        "mode": mode_name,
                        "ablationFlags": asdict(ablation),
                        "matrixAxes": matrix_axes_for_mode(mode_name),
                        "contextSizeTokens": context_size,
                        "needleDepthPct": depth,
                        "needleCount": needle_count,
                        "tokenBudget": budget_tokens,
                        "answerTokenCount": len(case["answerTokens"]),
                        "recalledTokenCount": len(recalled),
                        "recall": round(len(recalled) / len(case["answerTokens"]), 6),
                        "scoringMethod": "span-containment/exact-match over answerTokens; no LLM judge",
                        "wrongDistractorTokenCount": len(wrong),
                        "selectedDistractorPassageCount": len(selected_distractors),
                        "answerBearingSpanIncluded": bool(card.get("answerBearingSpanIncluded")),
                        "answerSpanPresentInCorpus": bool(card.get("answer_span_present_in_corpus")),
                        "answerSpanPresentInPack": bool(card.get("answer_span_present_in_pack")),
                        "contextPackCard": card,
                        "answer": answer,
                        "costLatency": {
                            "promptTokens": result.get("modelLog", {}).get("promptTokens"),
                            "completionTokens": result.get("modelLog", {}).get("completionTokens"),
                            "calls": 1 + int(result.get("repairAttempts", 0)),
                            "modelElapsedSec": result.get("elapsedSec"),
                            "wallTimeSec": wall_time,
                            "costUsd": result.get("modelLog", {}).get("costUsd", 0.0),
                        },
                    }
                    row["failureTaxonomy"] = classify_failure(row)
                    case_results.append(row)

    return {
        "packId": pack_id,
        "caseResults": case_results,
        "cardValidation": {
            "schema": str(CARD_SCHEMA.relative_to(ROOT)),
            "valid": not card_errors,
            "errors": card_errors,
        },
    }


def average(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def ci95(values: list[float]) -> list[float]:
    if not values:
        return [0.0, 0.0]
    mean = sum(values) / len(values)
    if len(values) == 1:
        return [round(mean, 6), round(mean, 6)]
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    margin = 1.96 * math.sqrt(variance / len(values))
    return [round(max(0.0, mean - margin), 6), round(min(1.0, mean + margin), 6)]


def classify_failure(row: dict[str, Any]) -> str | None:
    if row["recall"] >= 1.0:
        return None
    card = row.get("contextPackCard", {})
    if not card.get("answer_span_present_in_corpus", card.get("answerBearingSpanIds")):
        return "retrieval_miss"
    if not card.get("answer_span_present_in_pack", card.get("answerBearingSpanIncluded")):
        return "packer_eviction"
    if row.get("ablationFlags", {}).get("use_gate") and not str(row.get("answer", "")).strip():
        return "gate_suppressed"
    return "model_ignored_packed_span"


def matrix_axes_for_mode(mode: str) -> dict[str, bool] | None:
    if not mode.startswith("matrix-"):
        return None
    return {
        "use_gate": "-g1-" in mode,
        "use_retrieval": "-r1-" in mode,
        "use_context_packing": "-p1-" in mode,
    }


def build_headline_metric(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_case_mode = {(row["caseId"], row["mode"]): row for row in results}
    paired_deltas: list[float] = []
    for case_id in sorted({row["caseId"] for row in results}):
        raw = by_case_mode.get((case_id, RAW_BASELINE_MODE))
        gated = by_case_mode.get((case_id, GATED_PACKED_MODE))
        if raw is None or gated is None:
            continue
        paired_deltas.append(float(gated["recall"]) - float(raw["recall"]))
    raw_rows = [row for row in results if row["mode"] == RAW_BASELINE_MODE]
    gated_rows = [row for row in results if row["mode"] == GATED_PACKED_MODE]
    delta = average(paired_deltas)
    return {
        "metric": "gated_recall - raw_recall",
        "scoring": "deterministic span-containment/exact-match over synthetic answer tokens; no LLM judge",
        "rawBaselineMode": RAW_BASELINE_MODE,
        "gatedMode": GATED_PACKED_MODE,
        "rawRecall": average([float(row["recall"]) for row in raw_rows]),
        "gatedRecall": average([float(row["recall"]) for row in gated_rows]),
        "delta": delta,
        "ci95": ci95(paired_deltas),
        "pairedCases": len(paired_deltas),
    }


def build_ablation_matrix(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in MATRIX_MODE_ORDER:
        mode_rows = [row for row in results if row["mode"] == mode]
        axes = matrix_axes_for_mode(mode)
        if not mode_rows or axes is None:
            continue
        rows.append(
            {
                "mode": mode,
                **axes,
                "recall": average([float(row["recall"]) for row in mode_rows]),
                "wrongDistractorTokenRate": average([1.0 if row["wrongDistractorTokenCount"] else 0.0 for row in mode_rows]),
                "selectedDistractorPassageRate": average(
                    [1.0 if row["selectedDistractorPassageCount"] else 0.0 for row in mode_rows]
                ),
                "tokenBudget": sorted({int(row["tokenBudget"]) for row in mode_rows}),
                "allOffEqualsRawLongContextBaseline": mode == RAW_BASELINE_MODE,
            }
        )
    return rows


def build_position_length_cross_tab(results: list[dict[str, Any]]) -> dict[str, Any]:
    tab: dict[str, dict[str, float]] = {}
    for row in results:
        if row["mode"] != GATED_PACKED_MODE:
            continue
        key = f"depth_{row['needleDepthPct']}"
        bucket = tab.setdefault(key, {})
        bucket[str(row["contextSizeTokens"])] = average(
            [
                float(other["recall"])
                for other in results
                if other["mode"] == GATED_PACKED_MODE
                and other["needleDepthPct"] == row["needleDepthPct"]
                and other["contextSizeTokens"] == row["contextSizeTokens"]
            ]
        )
    return tab


def build_token_budget_summary(results: list[dict[str, Any]], budget_tokens: int) -> dict[str, Any]:
    by_mode = {
        mode: sorted({int(row["tokenBudget"]) for row in rows})
        for mode, rows in ((mode, [row for row in results if row["mode"] == mode]) for mode in sorted({row["mode"] for row in results}))
    }
    observed = sorted({budget for budgets in by_mode.values() for budget in budgets})
    return {
        "requestedBudgetTokens": budget_tokens,
        "observedBudgetTokens": observed,
        "identicalAcrossArms": observed == [budget_tokens],
        "byMode": by_mode,
    }


def build_control_sanity(results: list[dict[str, Any]]) -> dict[str, Any]:
    broken = [row for row in results if row["mode"] == "control-broken-packer"]
    oracle = [row for row in results if row["mode"] == "control-oracle-packer"]
    broken_recall = average([float(row["recall"]) for row in broken])
    oracle_recall = average([float(row["recall"]) for row in oracle])
    return {
        "brokenPackerPresent": bool(broken),
        "brokenPackerRecall": broken_recall,
        "brokenPackerAssertApproxZero": broken_recall <= 0.05 if broken else None,
        "oraclePackerPresent": bool(oracle),
        "oraclePackerRecall": oracle_recall,
        "oraclePackerAssertHigh": oracle_recall >= 0.95 if oracle else None,
        "note": "Controls use the same mock backend and token budget; broken excludes answer spans, oracle promotes them first.",
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_mode.setdefault(result["mode"], []).append(result)

    per_mode: dict[str, Any] = {}
    for mode, rows in sorted(by_mode.items()):
        single = [row for row in rows if row["needleCount"] == 1]
        multi = [row for row in rows if row["needleCount"] > 1]
        distractor_clean = [
            1.0 if row["wrongDistractorTokenCount"] == 0 and row["selectedDistractorPassageCount"] == 0 else 0.0
            for row in rows
        ]
        per_depth = {
            str(depth): average([row["recall"] for row in rows if row["needleDepthPct"] == depth])
            for depth in sorted({row["needleDepthPct"] for row in rows})
        }
        per_size = {
            str(size): average([row["recall"] for row in rows if row["contextSizeTokens"] == size])
            for size in sorted({row["contextSizeTokens"] for row in rows})
        }
        middle = [row["recall"] for row in rows if row["needleDepthPct"] in {25, 50, 75}]
        edges = [row["recall"] for row in rows if row["needleDepthPct"] in {0, 100}]
        prompt_tokens = [float(row["costLatency"].get("promptTokens") or 0) for row in rows]
        wall = [float(row["costLatency"].get("wallTimeSec") or 0) for row in rows]
        per_mode[mode] = {
            "multiNeedleRecall": {
                "singleNeedle": average([row["recall"] for row in single]),
                "multipleNeedles": average([row["recall"] for row in multi]),
                "overall": average([row["recall"] for row in rows]),
            },
            "positionSensitivity": {
                "recallByDepthPct": per_depth,
                "lostInMiddleDelta": round(average(edges) - average(middle), 6),
            },
            "distractorRobustness": {
                "cleanRate": average(distractor_clean),
                "wrongTokenRate": average([1.0 if row["wrongDistractorTokenCount"] else 0.0 for row in rows]),
                "selectedDistractorPassageRate": average(
                    [1.0 if row["selectedDistractorPassageCount"] else 0.0 for row in rows]
                ),
            },
            "costLatencyVsRecall": {
                "recallByContextSize": per_size,
                "meanPromptTokens": average(prompt_tokens),
                "meanWallTimeSec": average(wall),
                "totalCalls": sum(int(row["costLatency"]["calls"]) for row in rows),
                "costUsd": 0.0,
            },
        }
    return per_mode


def build_report(
    matrix: dict[str, Any],
    *,
    context_sizes: list[int],
    depths: list[int],
    needle_counts: list[int],
    full_matrix_available: bool,
    budget_tokens: int,
) -> dict[str, Any]:
    results = matrix["caseResults"]
    headline = build_headline_metric(results)
    ablation_matrix = build_ablation_matrix(results)
    control_sanity = build_control_sanity(results)
    return {
        "schema": "sophia.long_context.public_report.v1",
        "reportStatus": "candidate",
        "claimStatus": "candidate_not_validated",
        "candidateOnly": True,
        "canClaimAGI": False,
        "claimBoundary": (
            "CANDIDATE only: deterministic synthetic corpus, mock backend, no hidden non-synthetic eval, "
            "and no model-ability or AGI claim."
        ),
        "packId": matrix["packId"],
        "runAt": datetime.now().isoformat(timespec="seconds"),
        "visibility": "public-aggregate-no-prompts",
        "backend": "mock",
        "backendClaimBoundary": (
            "Recall came from the offline MOCK backend. It validates harness wiring, context packing, "
            "and deterministic scoring only; it is not evidence of model long-context ability."
        ),
        "corpus": {
            "kind": "deterministic synthetic",
            "seedRegeneratable": True,
            "selfAuthored": True,
            "hiddenPromptsPublished": False,
            "distractors": (
                "Plausible near-duplicates reuse the same synthetic entity with wrong dates, negation, "
                "and wrong LC_NEEDLE candidate tokens."
            ),
        },
        "matrix": {
            "label": "full long-context measurement" if full_matrix_available else "smoke matrix",
            "contextSizesTokensMeasured": context_sizes,
            "fullContextSizesTokensAvailableWithFullFlag": FULL_CONTEXT_SIZES,
            "depthsPct": depths,
            "needleCounts": needle_counts,
            "fullMatrixRun": full_matrix_available,
            "ablationCells": len(ablation_matrix),
            "allOffCellEqualsRawLongContextBaseline": any(
                row["mode"] == RAW_BASELINE_MODE and row["allOffEqualsRawLongContextBaseline"]
                for row in ablation_matrix
            ),
        },
        "tokenBudget": build_token_budget_summary(results, budget_tokens),
        "headlineMetric": headline,
        "ablationMatrix": ablation_matrix,
        "controls": control_sanity,
        "positionLengthCrossTab": build_position_length_cross_tab(results),
        "metricFamilies": {
            "multiNeedleRecall": "single and multiple verified synthetic needles recovered by span containment/exact match",
            "positionSensitivity": "recall by needle depth, including lost-in-the-middle delta",
            "distractorRobustness": "plausible-but-wrong passage selection and wrong-token output",
            "costLatencyVsRecall": "token estimates, call counts, wall-time, and recall by context size",
        },
        "summaryByMode": summarize(results),
        "contextPackCards": [row["contextPackCard"] for row in results],
        "cardValidation": matrix["cardValidation"],
        "failureTaxonomy": {
            "allowedValues": [
                "retrieval_miss",
                "packer_eviction",
                "model_ignored_packed_span",
                "gate_suppressed",
            ],
            "byCaseMode": [
                {
                    "caseId": row["caseId"],
                    "mode": row["mode"],
                    "failure": row["failureTaxonomy"],
                }
                for row in results
                if row["failureTaxonomy"]
            ],
        },
        "distractorSignal": {
            "rawBaselineWrongTokenRate": next(
                (
                    row["wrongDistractorTokenRate"]
                    for row in ablation_matrix
                    if row["mode"] == RAW_BASELINE_MODE
                ),
                0.0,
            ),
            "rawBaselineSelectedDistractorPassageRate": next(
                (
                    row["selectedDistractorPassageRate"]
                    for row in ablation_matrix
                    if row["mode"] == RAW_BASELINE_MODE
                ),
                0.0,
            ),
            "interpretation": (
                "The mock backend is span-extractive from accepted packed spans, so wrong-token output is not "
                "a model distractor-robustness signal. Distractors are measured as budget/ranking pressure "
                "and recall delta, not as validated adversarial robustness."
            ),
        },
        "measuredVsAsserted": {
            "measured": [
                "Synthetic needle recall through the shared run_case() pipeline.",
                "Context-pack card inclusion of answer-bearing spans.",
                "Raw-baseline vs gated-packed recall delta at the same token budget.",
                "Token/call/wall-time accounting for the mock offline backend.",
            ],
            "stillAssertedOrBlocked": [
                "Third-party long-context benchmark performance is not measured here.",
                "Live embedding retrieval replacement is not completed by this runner.",
                "The graded router remains an architecture bet unless separately wired and tested.",
                "Distractor robustness against real model behavior is not measured by this mock run.",
                "No trained checkpoint or AGI capability is claimed.",
            ],
        },
        "trainingFirebreak": {
            "eligibleAsTrainingTarget": False,
            "why": [
                "real retrieval is scaffolded rather than proven as the live default for this harness",
                "there is no held-out non-synthetic long-context evaluation",
                "any reward target must optimize recall and trap-abstention jointly, not recall alone",
            ],
        },
        "notes": [
            "Candidate only: self-authored synthetic corpus and mock backend.",
            "Every case result carries a schema-validated context-pack card.",
            "Quick 4k/16k runs are labeled smoke matrix; use --full for the separate 4k/16k/64k/128k measurement.",
            "Mock backend reads only the context block assembled by the packer/raw-budget arm.",
            "No hidden prompts, private answers, API keys, or external gold data are published.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Sophia long-context candidate benchmark offline")
    parser.add_argument("--context-sizes", default="", help="Comma-separated token sizes; default quick matrix")
    parser.add_argument("--depths", default=",".join(str(depth) for depth in DEPTHS))
    parser.add_argument("--needle-counts", default=",".join(str(count) for count in NEEDLE_COUNTS))
    parser.add_argument("--modes", default="all", help="'all' or comma-separated long-context ablation modes")
    parser.add_argument("--budget-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--full", action="store_true", help="Run 4k, 16k, 64k, and 128k+ context sizes")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    context_sizes = parse_ints(args.context_sizes) if args.context_sizes else (FULL_CONTEXT_SIZES if args.full else QUICK_CONTEXT_SIZES)
    depths = parse_ints(args.depths)
    needle_counts = parse_ints(args.needle_counts)
    modes = long_context_modes(args.modes)
    matrix = run_matrix(
        context_sizes=context_sizes,
        depths=depths,
        needle_counts=needle_counts,
        modes=modes,
        seed=args.seed,
        budget_tokens=args.budget_tokens,
    )
    report = build_report(
        matrix,
        context_sizes=context_sizes,
        depths=depths,
        needle_counts=needle_counts,
        full_matrix_available=args.full,
        budget_tokens=args.budget_tokens,
    )
    if not report["cardValidation"]["valid"]:
        print(json.dumps({"ok": False, "stage": "context-card-validation", "errors": report["cardValidation"]["errors"]}, indent=2))
        return 1
    controls = report["controls"]
    if controls["brokenPackerPresent"] and not controls["brokenPackerAssertApproxZero"]:
        print(json.dumps({"ok": False, "stage": "broken-packer-control", "controls": controls}, indent=2))
        return 1
    if controls["oraclePackerPresent"] and not controls["oraclePackerAssertHigh"]:
        print(json.dumps({"ok": False, "stage": "oracle-packer-control", "controls": controls}, indent=2))
        return 1

    out = args.out or (REPORT_DIR / f"long-context-candidate-{datetime.now().date().isoformat()}.public-report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print(json.dumps({k: report[k] for k in ("reportStatus", "claimStatus", "packId", "cardValidation")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
