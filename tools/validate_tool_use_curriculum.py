#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Validate the candidate-only tool-use/MCP curriculum metadata and DPO JSONL.

This is an offline shape/contamination check only. It does not train a model and does
not promote any proof-facing result.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench.dataset_guard import (  # noqa: E402
    eval_prompt_set,
    normalize,
    prompt_of,
    tool_use_benchmark_prompt_set,
)

CURRICULUM_DIR = ROOT / "training" / "tool_use"
MANIFEST = CURRICULUM_DIR / "manifest.json"
DPO_PAIRS = CURRICULUM_DIR / "dpo_pairs.jsonl"
MCP_TRACE_SCHEMA = CURRICULUM_DIR / "mcp_trace_schema.json"
HOLDOUT_SRC = ROOT / "training" / "lora" / "holdout.jsonl"

ALLOWED_REJECTED_TYPES = {
    "ignored_error",
    "mis_ground",
    "over_call",
    "schema_invalid",
    "spurious_extra",
    "wrong_tool",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{idx}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{idx}: row must be a JSON object")
        rows.append(row)
    return rows


def _holdout_prompt_set() -> set[str]:
    if not HOLDOUT_SRC.exists():
        return set()
    return {
        normalize(prompt)
        for row in _read_jsonl(HOLDOUT_SRC)
        for prompt in [prompt_of(row)]
        if prompt
    }


def _validate_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema") != "sophia.tool_use_curriculum.v1":
        errors.append("manifest schema must be sophia.tool_use_curriculum.v1")
    if manifest.get("candidateOnly") is not True:
        errors.append("manifest candidateOnly must be true")
    if manifest.get("canClaimAGI") is not False:
        errors.append("manifest canClaimAGI must be false")
    boundary = str(manifest.get("claimBoundary", ""))
    if "canClaimAGI=false" not in boundary:
        errors.append("manifest claimBoundary must state canClaimAGI=false")
    if manifest.get("mcpTraceSchema") != "training/tool_use/mcp_trace_schema.json":
        errors.append("manifest mcpTraceSchema must point at training/tool_use/mcp_trace_schema.json")
    return errors


def _validate_trace_schema(schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if schema.get("schema") != "sophia.mcp_trace_row.v1":
        errors.append("mcp trace schema id mismatch")
    if schema.get("candidateOnly") is not True:
        errors.append("mcp trace schema candidateOnly must be true")
    if schema.get("canClaimAGI") is not False:
        errors.append("mcp trace schema canClaimAGI must be false")
    required = set(schema.get("requiredTopLevelFields", []))
    if not {"messages", "toolTrace", "metadata"}.issubset(required):
        errors.append("mcp trace schema must require messages, toolTrace, and metadata")
    verdicts = set(schema.get("toolTrace", {}).get("verdictValues", []))
    if not {"accepted", "rejected", "held", "abstain"}.issubset(verdicts):
        errors.append("mcp trace schema must define accepted/rejected/held/abstain verdicts")
    return errors


def _validate_dpo_rows(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    rejected_types: Counter[str] = Counter()
    prompts: list[str] = []

    for idx, row in enumerate(rows, start=1):
        for key in ("prompt", "chosen", "rejected"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                errors.append(f"dpo_pairs.jsonl:{idx}: {key} must be a non-empty string")
        if row.get("chosen") == row.get("rejected"):
            errors.append(f"dpo_pairs.jsonl:{idx}: chosen and rejected must differ")
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            errors.append(f"dpo_pairs.jsonl:{idx}: metadata must be an object")
            continue
        rejected_type = metadata.get("rejected_type")
        if rejected_type not in ALLOWED_REJECTED_TYPES:
            errors.append(f"dpo_pairs.jsonl:{idx}: unsupported rejected_type {rejected_type!r}")
        else:
            rejected_types[str(rejected_type)] += 1
        if not isinstance(metadata.get("caseId"), str) or not metadata["caseId"].strip():
            errors.append(f"dpo_pairs.jsonl:{idx}: metadata.caseId must be a non-empty string")
        prompt = prompt_of(row)
        if prompt:
            prompts.append(prompt)

    forbidden = eval_prompt_set(root=ROOT) | tool_use_benchmark_prompt_set(root=ROOT) | _holdout_prompt_set()
    overlaps = [prompt for prompt in prompts if normalize(prompt) in forbidden]
    if overlaps:
        errors.append(f"tool-use curriculum overlaps held-out prompts: {len(overlaps)}")

    summary = {
        "rows": len(rows),
        "uniquePrompts": len({normalize(prompt) for prompt in prompts}),
        "rejectedTypes": dict(sorted(rejected_types.items())),
        "heldoutOverlapCount": len(overlaps),
    }
    return errors, summary


def validate() -> tuple[int, dict[str, Any]]:
    manifest = _read_json(MANIFEST)
    trace_schema = _read_json(MCP_TRACE_SCHEMA)
    rows = _read_jsonl(DPO_PAIRS)

    errors: list[str] = []
    errors.extend(_validate_manifest(manifest))
    errors.extend(_validate_trace_schema(trace_schema))
    row_errors, summary = _validate_dpo_rows(rows)
    errors.extend(row_errors)

    pack = manifest.get("packs", {}).get("dpo_pairs.jsonl", {})
    if pack.get("rows") != summary["rows"]:
        errors.append("manifest dpo_pairs rows does not match JSONL")
    if pack.get("uniquePrompts") != summary["uniquePrompts"]:
        errors.append("manifest dpo_pairs uniquePrompts does not match JSONL")
    if pack.get("rejectedTypes") != summary["rejectedTypes"]:
        errors.append("manifest dpo_pairs rejectedTypes does not match JSONL")

    result = {
        "ok": not errors,
        "errors": errors,
        "summary": summary,
        "claimBoundary": manifest.get("claimBoundary"),
    }
    return (0 if not errors else 1), result


def main() -> int:
    code, result = validate()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
