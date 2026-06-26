# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Held-out task pack validation for real-cognition scaffolding.

This module does not implement program induction, ARC solving, table reasoning,
code repair, or a world model. It defines the boundary that a future evaluator
must clear before those tasks can be treated as real held-out evidence instead
of toy reference demos.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PACK_SCHEMA = "sophia.real_cognition.heldout_tasks.v1"
SPECS_SCHEMA = "sophia.real_cognition.benchmark_specs.v1"
REAL_HELDOUT_TIER = "real-heldout-spec"
TASK_FAMILIES = ("arc_like", "table_transform", "code_transform")
ALLOWED_VERIFIERS = {"exact_match", "execution", "official_grader"}
SPLITS = ("fit", "validation", "test")
TOY_MARKERS = ("toy", "demo", "smoke", "style sample", "not real", "example/")


def _load_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def _stable_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _iter_values(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from _iter_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_values(child)
    else:
        yield value


def _contains_toy_marker(value: Any) -> bool:
    texts = [str(v).lower() for v in _iter_values(value)]
    return any(marker in text for text in texts for marker in TOY_MARKERS)


def _validate_common_boundary(data: dict[str, Any], errors: list[str]) -> None:
    if data.get("candidateOnly") is not True:
        errors.append("candidateOnly must be true; real-cognition scaffolds are not public capability evidence")
    if data.get("level3Evidence") is not False:
        errors.append("level3Evidence must be false; these specs do not prove AGI or Level-3 capability")


def _split_examples(task: dict[str, Any], errors: list[str]) -> dict[str, list[dict[str, Any]]]:
    splits = task.get("splits")
    task_id = str(task.get("taskId", "<missing-taskId>"))
    if not isinstance(splits, dict):
        errors.append(f"{task_id}: splits must contain fit/validation/test examples")
        return {}

    out: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        rows = splits.get(split)
        if not isinstance(rows, list) or not rows:
            errors.append(f"{task_id}: split {split!r} must be a non-empty list")
            continue
        clean_rows: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                errors.append(f"{task_id}: {split}[{idx}] must be an object")
                continue
            if "input" not in row or "output" not in row:
                errors.append(f"{task_id}: {split}[{idx}] must include input and output")
                continue
            clean_rows.append(row)
        out[split] = clean_rows
    return out


def validate_real_cognition_pack(data: dict[str, Any], *, require_real_heldout: bool = False) -> list[str]:
    """Return validation errors for a real-cognition held-out task pack."""
    errors: list[str] = []
    if data.get("schema") != PACK_SCHEMA:
        errors.append(f"schema must be {PACK_SCHEMA}")
    _validate_common_boundary(data, errors)

    if require_real_heldout:
        if data.get("benchmarkTier") != REAL_HELDOUT_TIER or data.get("isToyReference") is not False:
            errors.append("toy/reference packs cannot satisfy require_real_heldout")
        if data.get("sealed") is not True:
            errors.append("real held-out packs must be sealed before scoring")
        policy = data.get("contaminationPolicy", {})
        if not isinstance(policy, dict) or policy.get("trainingUseForbidden") is not True:
            errors.append("contaminationPolicy.trainingUseForbidden must be true")
        if not isinstance(policy, dict) or policy.get("requiresLeakageAudit") is not True:
            errors.append("contaminationPolicy.requiresLeakageAudit must be true")
        if _contains_toy_marker(data):
            errors.append("toy/reference marker found in pack content")

    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        errors.append("tasks must be a non-empty list")
        return errors

    for task in tasks:
        if not isinstance(task, dict):
            errors.append("each task must be an object")
            continue
        task_id = str(task.get("taskId", "<missing-taskId>"))
        if not task.get("taskId"):
            errors.append("taskId is required")
        if task.get("family") not in TASK_FAMILIES:
            errors.append(f"{task_id}: family must be one of {', '.join(TASK_FAMILIES)}")
        if task.get("verifier") not in ALLOWED_VERIFIERS:
            errors.append(f"{task_id}: verifier must be one of {', '.join(sorted(ALLOWED_VERIFIERS))}")

        splits = _split_examples(task, errors)
        if not all(split in splits for split in SPLITS):
            continue
        split_inputs = {
            split: {_stable_key(row["input"]) for row in rows}
            for split, rows in splits.items()
        }
        trainish = split_inputs["fit"] | split_inputs["validation"]
        leaked = trainish & split_inputs["test"]
        if leaked:
            errors.append(f"{task_id}: held-out overlap between fit/validation and test inputs")
        if split_inputs["fit"] & split_inputs["validation"]:
            errors.append(f"{task_id}: fit/validation overlap found")

    return errors


def summarize_real_cognition_pack(data: dict[str, Any]) -> dict[str, Any]:
    tasks = data.get("tasks") if isinstance(data.get("tasks"), list) else []
    return {
        "nTasks": len(tasks),
        "families": sorted({str(t.get("family")) for t in tasks if isinstance(t, dict) and t.get("family")}),
        "nFitExamples": sum(len(t.get("splits", {}).get("fit", [])) for t in tasks if isinstance(t, dict)),
        "nValidationExamples": sum(len(t.get("splits", {}).get("validation", [])) for t in tasks if isinstance(t, dict)),
        "nHeldoutExamples": sum(len(t.get("splits", {}).get("test", [])) for t in tasks if isinstance(t, dict)),
    }


def load_real_cognition_pack(path: str | Path, *, require_real_heldout: bool = False) -> dict[str, Any]:
    """Load a held-out task pack or raise ``ValueError`` with boundary errors."""
    data = _load_json(path)
    errors = validate_real_cognition_pack(data, require_real_heldout=require_real_heldout)
    if errors:
        raise ValueError("; ".join(errors))
    return {**data, "summary": summarize_real_cognition_pack(data)}


def validate_benchmark_specs(data: dict[str, Any]) -> list[str]:
    """Return validation errors for the public benchmark-family specification."""
    errors: list[str] = []
    if data.get("schema") != SPECS_SCHEMA:
        errors.append(f"schema must be {SPECS_SCHEMA}")
    _validate_common_boundary(data, errors)
    if data.get("canClaimAGI") is not False:
        errors.append("canClaimAGI must be false")
    if tuple(data.get("families", ())) != TASK_FAMILIES:
        errors.append(f"families must be {list(TASK_FAMILIES)}")
    benchmarks = data.get("benchmarks")
    if not isinstance(benchmarks, list) or not benchmarks:
        errors.append("benchmarks must be a non-empty list")
        return errors
    seen: set[str] = set()
    for item in benchmarks:
        if not isinstance(item, dict):
            errors.append("each benchmark spec must be an object")
            continue
        family = item.get("family")
        if family not in TASK_FAMILIES:
            errors.append(f"{item.get('benchmarkId', '<missing>')}: unknown family {family!r}")
        seen.add(str(family))
        if item.get("heldOutRequired") is not True:
            errors.append(f"{item.get('benchmarkId', '<missing>')}: heldOutRequired must be true")
        if item.get("verifier") not in ALLOWED_VERIFIERS:
            errors.append(f"{item.get('benchmarkId', '<missing>')}: unsupported verifier")
        acceptance = item.get("acceptanceGate", {})
        if not isinstance(acceptance, dict) or acceptance.get("beatsMemorizationBaseline") is not True:
            errors.append(f"{item.get('benchmarkId', '<missing>')}: must require a memorization baseline")
        if not isinstance(acceptance, dict) or acceptance.get("oodDetectionRequired") is not True:
            errors.append(f"{item.get('benchmarkId', '<missing>')}: must require OOD detection")
    missing = set(TASK_FAMILIES) - seen
    if missing:
        errors.append(f"missing benchmark families: {', '.join(sorted(missing))}")
    return errors


def load_benchmark_specs(path: str | Path) -> dict[str, Any]:
    data = _load_json(path)
    errors = validate_benchmark_specs(data)
    if errors:
        raise ValueError("; ".join(errors))
    return data


__all__ = [
    "PACK_SCHEMA",
    "SPECS_SCHEMA",
    "REAL_HELDOUT_TIER",
    "TASK_FAMILIES",
    "load_benchmark_specs",
    "load_real_cognition_pack",
    "summarize_real_cognition_pack",
    "validate_benchmark_specs",
    "validate_real_cognition_pack",
]
