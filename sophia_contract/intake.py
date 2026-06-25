# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Request Triage + Intake Contract wire model.

The intake contract extends the existing Sophia contract conventions: stable
enums, deterministic ids via ``claim_id_for()``, optional signatures via the
same ``build_claim()`` signing path, explicit BLP labels, and fail-closed
validation. It stores references and checkable routing metadata, never answers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from sophia_contract import blp
from sophia_contract.models import HELD_REASONS, SOURCE_STATUSES, build_claim, claim_id_for
from sophia_contract.scopes import OPS, Scope

INTAKE_CONTRACT_VERSION = "1.0.0"
INTAKE_SCHEMA_ID = "https://sophia.aihk/schema/intake-contract-1.0.0.json"
RECOMMENDED_ACTIONS = ("allow", "retrieve", "clarify", "escalate", "abstain")
INTENT_LABELS = (
    "answer_question",
    "coding_task",
    "tool_use",
    "planning",
    "learning",
    "advice",
    "unknown",
)
RISK_LEVELS = ("low", "medium", "high", "critical")
SIGNING_CREATED_AT = "1970-01-01T00:00:00+00:00"


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def mechanical_normalize_prompt(prompt: str) -> tuple[str, list[dict[str, str]]]:
    """Normalize only encoding/whitespace artifacts and record each change."""
    normalized = prompt.replace("\r\n", "\n").replace("\r", "\n")
    changes: list[dict[str, str]] = []
    if normalized != prompt:
        changes.append({"kind": "line_endings", "description": "Converted CRLF/CR line endings to LF."})
    stripped_lines = "\n".join(line.rstrip() for line in normalized.split("\n"))
    if stripped_lines != normalized:
        changes.append({"kind": "trailing_whitespace", "description": "Removed trailing spaces from line ends."})
        normalized = stripped_lines
    return normalized, changes


@dataclass(frozen=True)
class IntakeContract:
    """Typed wrapper around the stable intake contract dictionary."""

    data: dict[str, Any]

    @property
    def contract_id(self) -> str:
        return str(self.data["contract_id"])

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)


def _stable_payload(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in contract.items()
        if key not in {"contract_id", "signature"}
    }


def contract_id_for(contract_without_id: dict[str, Any]) -> str:
    stable = _canonical(_stable_payload(contract_without_id))
    return claim_id_for(f"intake:{stable}")


def _list_of_objects(value: Any, field: str, errors: list[str]) -> None:
    if not isinstance(value, list):
        errors.append(f"{field} must be an array")
        return
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            errors.append(f"{field}[{index}] must be an object")


def validate_intake_contract(contract: dict[str, Any]) -> list[str]:
    """Return validation errors. An empty list means the contract is executable."""
    errors: list[str] = []
    if not isinstance(contract, dict):
        return ["contract must be an object"]
    required = (
        "schema_version",
        "contract_id",
        "original_prompt",
        "original_prompt_sha256",
        "normalized_prompt",
        "normalization_changes",
        "intent_label",
        "ambiguities",
        "constraints",
        "success_criteria",
        "failure_criteria",
        "risk_level",
        "blp_level",
        "provenance_requirements",
        "tool_policy",
        "recommended_action",
        "material_rewrites",
        "confidence",
    )
    for field in required:
        if field not in contract:
            errors.append(f"missing required field: {field}")
    if errors:
        return errors
    if contract["schema_version"] != INTAKE_CONTRACT_VERSION:
        errors.append(f"schema_version must be {INTAKE_CONTRACT_VERSION}")
    if not isinstance(contract["original_prompt"], str) or not contract["original_prompt"]:
        errors.append("original_prompt must be a non-empty string")
    if contract["original_prompt_sha256"] != prompt_sha256(contract.get("original_prompt", "")):
        errors.append("original_prompt_sha256 does not match original_prompt")
    expected_id = contract_id_for(contract)
    if contract["contract_id"] != expected_id:
        errors.append(f"contract_id must be deterministic id {expected_id}")
    if contract["intent_label"] not in INTENT_LABELS:
        errors.append(f"intent_label must be one of {INTENT_LABELS}")
    if contract["risk_level"] not in RISK_LEVELS:
        errors.append(f"risk_level must be one of {RISK_LEVELS}")
    if not blp.is_level(contract["blp_level"]):
        errors.append(f"blp_level must be one of {blp.BLP_LEVELS}")
    if contract["recommended_action"] not in RECOMMENDED_ACTIONS:
        errors.append(f"recommended_action must be one of {RECOMMENDED_ACTIONS}")
    try:
        confidence = float(contract["confidence"])
        if not 0.0 <= confidence <= 1.0:
            errors.append("confidence must be in [0,1]")
    except (TypeError, ValueError):
        errors.append("confidence must be a number")
    for field in (
        "normalization_changes",
        "ambiguities",
        "constraints",
        "success_criteria",
        "failure_criteria",
        "material_rewrites",
    ):
        _list_of_objects(contract[field], field, errors)
    for index, ambiguity in enumerate(contract.get("ambiguities", [])):
        if not ambiguity.get("clarifying_question"):
            errors.append(f"ambiguities[{index}] must include clarifying_question")
    provenance = contract.get("provenance_requirements", {})
    if not isinstance(provenance, dict):
        errors.append("provenance_requirements must be an object")
    else:
        refs = provenance.get("source_refs", [])
        if not isinstance(refs, list):
            errors.append("provenance_requirements.source_refs must be an array")
        for index, ref in enumerate(refs):
            if not isinstance(ref, dict) or "id" not in ref or "status" not in ref:
                errors.append(f"source_refs[{index}] must contain id and status")
                continue
            if ref["status"] not in SOURCE_STATUSES:
                errors.append(f"source_refs[{index}].status must be one of {SOURCE_STATUSES}")
            if "content" in ref or "text" in ref:
                errors.append(f"source_refs[{index}] must store references, not inlined content")
    policy = contract.get("tool_policy", {})
    if not isinstance(policy, dict):
        errors.append("tool_policy must be an object")
    else:
        for key in ("requested_ops", "scope_ops", "allowed_ops", "denied_ops"):
            ops = policy.get(key, [])
            if not isinstance(ops, list):
                errors.append(f"tool_policy.{key} must be an array")
                continue
            unknown = set(ops) - set(OPS)
            if unknown:
                errors.append(f"tool_policy.{key} contains unknown ops: {sorted(unknown)}")
        if policy.get("basis") != "intersection_with_scope":
            errors.append("tool_policy.basis must be intersection_with_scope")
    for index, rewrite in enumerate(contract.get("material_rewrites", [])):
        if rewrite.get("surfaced") is not True:
            errors.append(f"material_rewrites[{index}] must be surfaced")
    return errors


def validate_monotonic_narrowing(contract: dict[str, Any], scope: Scope) -> list[str]:
    """Prove intake narrowed relative to the role scope instead of widening it."""
    errors = validate_intake_contract(contract)
    if errors:
        return errors
    policy = contract["tool_policy"]
    requested_ops = set(policy.get("requested_ops", []))
    scope_ops = set(policy.get("scope_ops", []))
    allowed_ops = set(policy.get("allowed_ops", []))
    if scope_ops != set(scope.ops):
        errors.append("tool_policy.scope_ops must mirror the role scope")
    if allowed_ops != requested_ops.intersection(scope.ops):
        errors.append("tool_policy.allowed_ops must equal requested_ops ∩ role scope ops")
    if not allowed_ops.issubset(scope.ops):
        errors.append("tool_policy.allowed_ops widens beyond the role scope")
    if blp.rank(contract["blp_level"]) > blp.rank(scope.max_blp):
        errors.append(f"blp_level {contract['blp_level']} exceeds role max_blp {scope.max_blp}")
    return errors


def build_intake_contract(fields: dict[str, Any], *, signing_key: str | None = None) -> IntakeContract:
    """Assemble and validate an intake contract with deterministic id/signature."""
    original_prompt = str(fields["original_prompt"])
    normalized_prompt = fields.get("normalized_prompt")
    normalization_changes = fields.get("normalization_changes")
    if normalized_prompt is None or normalization_changes is None:
        normalized_prompt, normalization_changes = mechanical_normalize_prompt(original_prompt)
    contract = {
        "schema_version": INTAKE_CONTRACT_VERSION,
        "original_prompt": original_prompt,
        "original_prompt_sha256": prompt_sha256(original_prompt),
        "normalized_prompt": normalized_prompt,
        "normalization_changes": list(normalization_changes),
        "intent_label": fields.get("intent_label", "unknown"),
        "ambiguities": list(fields.get("ambiguities", [])),
        "constraints": list(fields.get("constraints", [])),
        "success_criteria": list(fields.get("success_criteria", [])),
        "failure_criteria": list(fields.get("failure_criteria", [])),
        "risk_level": fields.get("risk_level", "medium"),
        "blp_level": fields.get("blp_level", "UNCLASSIFIED"),
        "provenance_requirements": dict(fields.get("provenance_requirements", {})),
        "tool_policy": dict(fields.get("tool_policy", {})),
        "recommended_action": fields.get("recommended_action", "abstain"),
        "material_rewrites": list(fields.get("material_rewrites", [])),
        "confidence": round(float(fields.get("confidence", 0.0)), 4),
    }
    contract["contract_id"] = contract_id_for(contract)
    if signing_key:
        signed = build_claim(
            {
                "idempotency_key": contract["contract_id"],
                "content": _canonical(_stable_payload(contract)),
                "sources": [],
                "parents": [],
                "blp_level": contract["blp_level"],
            },
            created_at=SIGNING_CREATED_AT,
            signing_key=signing_key,
        )
        contract["signature"] = signed["signature"]
    errors = validate_intake_contract(contract)
    if errors:
        raise ValueError("; ".join(errors))
    return IntakeContract(contract)


__all__ = [
    "INTAKE_CONTRACT_VERSION",
    "INTAKE_SCHEMA_ID",
    "INTENT_LABELS",
    "RECOMMENDED_ACTIONS",
    "RISK_LEVELS",
    "IntakeContract",
    "build_intake_contract",
    "contract_id_for",
    "mechanical_normalize_prompt",
    "prompt_sha256",
    "validate_intake_contract",
    "validate_monotonic_narrowing",
]
