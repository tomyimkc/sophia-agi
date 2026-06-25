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
    answer_tokens = [f"LC_NEEDLE_{context_size}_{depth_pct}_{needle_count}_{idx}_{seed}" for idx in range(needle_count)]
    distractor_tokens = [f"LC_DISTRACTOR_{context_size}_{depth_pct}_{idx}_{seed}" for idx in range(2)]
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
                f"Plausible but wrong distractor for {case_id}. "
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
    return errors


def long_context_modes(raw_modes: str) -> dict[str, Any]:
    available = {
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
        return available
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
                    case_results.append(
                        {
                            "caseId": case["id"],
                            "mode": mode_name,
                            "ablationFlags": asdict(ablation),
                            "contextSizeTokens": context_size,
                            "needleDepthPct": depth,
                            "needleCount": needle_count,
                            "answerTokenCount": len(case["answerTokens"]),
                            "recalledTokenCount": len(recalled),
                            "recall": round(len(recalled) / len(case["answerTokens"]), 6),
                            "wrongDistractorTokenCount": len(wrong),
                            "selectedDistractorPassageCount": len(selected_distractors),
                            "answerBearingSpanIncluded": bool(card.get("answerBearingSpanIncluded")),
                            "contextPackCard": card,
                            "costLatency": {
                                "promptTokens": result.get("modelLog", {}).get("promptTokens"),
                                "completionTokens": result.get("modelLog", {}).get("completionTokens"),
                                "calls": 1 + int(result.get("repairAttempts", 0)),
                                "modelElapsedSec": result.get("elapsedSec"),
                                "wallTimeSec": wall_time,
                                "costUsd": result.get("modelLog", {}).get("costUsd", 0.0),
                            },
                        }
                    )

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
) -> dict[str, Any]:
    results = matrix["caseResults"]
    return {
        "schema": "sophia.long_context.public_report.v1",
        "reportStatus": "candidate",
        "claimStatus": "candidate_not_validated",
        "candidateOnly": True,
        "canClaimAGI": False,
        "packId": matrix["packId"],
        "runAt": datetime.now().isoformat(timespec="seconds"),
        "visibility": "public-aggregate-no-prompts",
        "backend": "mock",
        "corpus": {
            "kind": "deterministic synthetic",
            "seedRegeneratable": True,
            "selfAuthored": True,
            "hiddenPromptsPublished": False,
        },
        "matrix": {
            "contextSizesTokensMeasured": context_sizes,
            "fullContextSizesTokensAvailableWithFullFlag": FULL_CONTEXT_SIZES,
            "depthsPct": depths,
            "needleCounts": needle_counts,
            "fullMatrixRun": full_matrix_available,
        },
        "metricFamilies": {
            "multiNeedleRecall": "single and multiple verified synthetic needles recovered from packed context",
            "positionSensitivity": "recall by needle depth, including lost-in-the-middle delta",
            "distractorRobustness": "plausible-but-wrong passage selection and wrong-token output",
            "costLatencyVsRecall": "token estimates, call counts, wall-time, and recall by context size",
        },
        "summaryByMode": summarize(results),
        "contextPackCards": [row["contextPackCard"] for row in results],
        "cardValidation": matrix["cardValidation"],
        "measuredVsAsserted": {
            "measured": [
                "Synthetic needle recall through the shared run_case() pipeline.",
                "Context-pack card inclusion of answer-bearing spans.",
                "Distractor selection/output on deterministic plausible-wrong passages.",
                "Token/call/wall-time accounting for the mock offline backend.",
            ],
            "stillAssertedOrBlocked": [
                "Third-party long-context benchmark performance is not measured here.",
                "Live embedding retrieval replacement is not completed by this runner.",
                "The graded router remains an architecture bet unless separately wired and tested.",
                "No trained checkpoint or AGI capability is claimed.",
            ],
        },
        "notes": [
            "Candidate only: self-authored synthetic corpus and mock backend.",
            "Every case result carries a schema-validated context-pack card.",
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
    )
    if not report["cardValidation"]["valid"]:
        print(json.dumps({"ok": False, "stage": "context-card-validation", "errors": report["cardValidation"]["errors"]}, indent=2))
        return 1

    out = args.out or (REPORT_DIR / f"long-context-candidate-{datetime.now().date().isoformat()}.public-report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print(json.dumps({k: report[k] for k in ("reportStatus", "claimStatus", "packId", "cardValidation")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
