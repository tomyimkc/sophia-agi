#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Validate and summarize context-packing scaffold artifacts.

This is an offline verifier for metadata shape and no-overclaim invariants. It
does not train a model, score a benchmark, or promote capability evidence.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "training" / "context_packing" / "manifest.json"
EVIDENCE_CARD_SCHEMA = ROOT / "training" / "context_packing" / "evidence_card.schema.json"

REQUIRED_STATES = {"accepted", "rejected", "held", "abstain"}
REQUIRED_CHANNELS = {
    "task",
    "system_contract",
    "source_evidence",
    "working_memory",
    "episodic_memory",
    "semantic_memory",
    "tool_trace",
    "council_notes",
    "failure_context",
}
SCAFFOLD_SAMPLE_CLAIM_STATUSES = {"illustrative", "not_evidence"}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.exists():
        return None, [f"{rel(path)}: missing"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, [f"{rel(path)}: invalid JSON: {exc}"]
    if not isinstance(data, dict):
        return None, [f"{rel(path)}: expected a JSON object"]
    return data, []


def require(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def require_candidate_boundary(errors: list[str], data: dict[str, Any], label: str) -> None:
    require(errors, data.get("candidateOnly") is True, f"{label}: candidateOnly must be true")
    require(errors, data.get("canClaimAGI") is False, f"{label}: canClaimAGI must be false")
    boundary = str(data.get("claimBoundary", ""))
    require(errors, "canClaimAGI=false" in boundary, f"{label}: claimBoundary must include canClaimAGI=false")


def _validate_card(card: dict[str, Any], label: str) -> list[str]:
    errors: list[str] = []
    require(errors, card.get("schema") == "sophia.context_packing.evidence_card.v1", f"{label}: schema mismatch")
    require_candidate_boundary(errors, card, label)

    status = card.get("status")
    require(errors, status in REQUIRED_STATES, f"{label}: status must be one of accepted/rejected/held/abstain")
    require(
        errors,
        card.get("claimStatus") in SCAFFOLD_SAMPLE_CLAIM_STATUSES,
        f"{label}: sample cards must be illustrative or not_evidence",
    )
    require(errors, card.get("contextChannel") in REQUIRED_CHANNELS, f"{label}: unknown contextChannel")

    source = card.get("source")
    if isinstance(source, dict):
        for key in ("kind", "pathOrUri", "sourceId"):
            require(errors, isinstance(source.get(key), str) and source[key].strip(), f"{label}: source.{key} required")
    else:
        errors.append(f"{label}: source must be an object")

    summary = card.get("summary")
    if isinstance(summary, dict):
        require(errors, isinstance(summary.get("text"), str) and summary["text"].strip(), f"{label}: summary.text required")
        require(errors, isinstance(summary.get("preservedClaims"), list), f"{label}: preservedClaims must be a list")
        require(errors, isinstance(summary.get("knownLimits"), list), f"{label}: knownLimits must be a list")
    else:
        errors.append(f"{label}: summary must be an object")

    verifier = card.get("verifier")
    if isinstance(verifier, dict):
        require(errors, verifier.get("state") == status, f"{label}: verifier.state must match status")
        require(errors, isinstance(verifier.get("method"), str) and verifier["method"].strip(), f"{label}: verifier.method required")
        require(errors, isinstance(verifier.get("reasons"), list) and verifier["reasons"], f"{label}: verifier.reasons required")
    else:
        errors.append(f"{label}: verifier must be an object")

    token_estimate = card.get("tokenEstimate")
    if isinstance(token_estimate, dict):
        raw_tokens = token_estimate.get("rawTokens")
        packed_tokens = token_estimate.get("packedTokens")
        budget_tokens = token_estimate.get("budgetTokens")
        require(errors, isinstance(raw_tokens, int) and raw_tokens >= 0, f"{label}: rawTokens must be a non-negative integer")
        require(errors, isinstance(packed_tokens, int) and packed_tokens >= 0, f"{label}: packedTokens must be a non-negative integer")
        require(errors, isinstance(budget_tokens, int) and budget_tokens > 0, f"{label}: budgetTokens must be a positive integer")
        if isinstance(packed_tokens, int) and isinstance(budget_tokens, int):
            require(errors, packed_tokens <= budget_tokens, f"{label}: packedTokens exceeds budgetTokens")
    else:
        errors.append(f"{label}: tokenEstimate must be an object")

    training_use = card.get("trainingUse")
    if isinstance(training_use, dict):
        eligible = training_use.get("eligible")
        if status != "accepted":
            require(errors, eligible is False, f"{label}: non-accepted sample cards cannot be training-eligible")
        require(
            errors,
            training_use.get("requiresHumanReview") is True,
            f"{label}: sample cards must require human review",
        )
        require(errors, training_use.get("disjointFromEval") is True, f"{label}: sample cards must be eval-disjoint")
    else:
        errors.append(f"{label}: trainingUse must be an object")

    return errors


def validate_manifest(path: Path = MANIFEST) -> tuple[list[str], dict[str, Any]]:
    data, errors = load_json(path)
    if data is None:
        return errors, {"cards": 0}

    require(errors, data.get("schema") == "sophia.context_packing.manifest.v1", f"{rel(path)}: schema mismatch")
    require(errors, data.get("status") == "scaffolding_only_not_run", f"{rel(path)}: status must be scaffolding_only_not_run")
    require_candidate_boundary(errors, data, rel(path))

    schema_path = ROOT / str(data.get("evidenceCardSchema", ""))
    require(errors, schema_path == EVIDENCE_CARD_SCHEMA, f"{rel(path)}: evidenceCardSchema path mismatch")
    require(errors, schema_path.exists(), f"{rel(path)}: evidenceCardSchema does not exist")

    states = set(data.get("verdictValues", []))
    channels = set(data.get("contextChannels", []))
    require(errors, REQUIRED_STATES <= states, f"{rel(path)}: missing verdict states")
    require(errors, REQUIRED_CHANNELS <= channels, f"{rel(path)}: missing context channels")

    policy = data.get("packingPolicy")
    if isinstance(policy, dict):
        require(errors, policy.get("requireSourceIds") is True, f"{rel(path)}: requireSourceIds must be true")
        require(errors, policy.get("requireVerifierState") is True, f"{rel(path)}: requireVerifierState must be true")
        require(errors, policy.get("failClosedOnMissingEvidence") is True, f"{rel(path)}: failClosedOnMissingEvidence must be true")
    else:
        errors.append(f"{rel(path)}: packingPolicy must be an object")

    cards = data.get("sampleEvidenceCards", [])
    require(errors, isinstance(cards, list), f"{rel(path)}: sampleEvidenceCards must be a list")

    by_state: Counter[str] = Counter()
    by_channel: Counter[str] = Counter()
    raw_tokens = 0
    packed_tokens = 0

    for idx, card in enumerate(cards if isinstance(cards, list) else [], start=1):
        if not isinstance(card, dict):
            errors.append(f"{rel(path)}:sampleEvidenceCards[{idx}]: must be an object")
            continue
        errors.extend(_validate_card(card, f"{rel(path)}:sampleEvidenceCards[{idx}]"))
        by_state[str(card.get("status"))] += 1
        by_channel[str(card.get("contextChannel"))] += 1
        token_estimate = card.get("tokenEstimate", {})
        if isinstance(token_estimate, dict):
            raw_tokens += int(token_estimate.get("rawTokens", 0))
            packed_tokens += int(token_estimate.get("packedTokens", 0))

    summary = {
        "cards": len(cards) if isinstance(cards, list) else 0,
        "states": dict(sorted(by_state.items())),
        "channels": dict(sorted(by_channel.items())),
        "rawTokens": raw_tokens,
        "packedTokens": packed_tokens,
        "savedTokens": max(raw_tokens - packed_tokens, 0),
        "claimBoundary": data.get("claimBoundary"),
    }
    return errors, summary


def validate(path: Path = MANIFEST) -> tuple[int, dict[str, Any]]:
    schema, schema_errors = load_json(EVIDENCE_CARD_SCHEMA)
    errors = list(schema_errors)
    if schema is not None:
        require(errors, schema.get("title") == "Sophia Context Packing Evidence Card", f"{rel(EVIDENCE_CARD_SCHEMA)}: title mismatch")
        required = set(schema.get("required", []))
        require(errors, {"schema", "cardId", "status", "claimStatus", "verifier"} <= required, f"{rel(EVIDENCE_CARD_SCHEMA)}: missing required card fields")

    manifest_errors, summary = validate_manifest(path)
    errors.extend(manifest_errors)
    result = {
        "ok": not errors,
        "errors": errors,
        "summary": summary,
    }
    return (0 if not errors else 1), result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Sophia context-packing scaffold artifacts")
    parser.add_argument("--manifest", default=str(MANIFEST), help="Path to context-packing manifest")
    parser.add_argument("--json", action="store_true", help="Print machine-readable validation output")
    args = parser.parse_args()

    code, result = validate(Path(args.manifest))
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif result["errors"]:
        print("Context-packing validation failed:")
        for error in result["errors"]:
            print(f"- {error}")
    else:
        summary = result["summary"]
        print(
            "Context-packing validation OK "
            f"({summary['cards']} sample cards, {summary['packedTokens']}/{summary['rawTokens']} packed tokens)"
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
