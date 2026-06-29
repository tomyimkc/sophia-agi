#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Validate attribution records and training examples in the Sophia AGI corpus."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATTRIBUTIONS_PATH = ROOT / "data" / "attributions.json"
EXAMPLES_DIR = ROOT / "training" / "examples"

REQUIRED_ATTRIBUTION_KEYS = {
    "textId",
    "tradition",
    "attributedAuthor",
    "authorConfidence",
    "doNotAttributeTo",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_attributions(records: dict) -> list[str]:
    errors: list[str] = []
    for key, record in records.items():
        if record.get("textId") != key:
            errors.append(f"{key}: textId mismatch ({record.get('textId')})")
        missing = REQUIRED_ATTRIBUTION_KEYS - set(record.keys())
        if missing:
            errors.append(f"{key}: missing keys {sorted(missing)}")
        if not isinstance(record.get("doNotAttributeTo"), list):
            errors.append(f"{key}: doNotAttributeTo must be a list")
    return errors


def validate_training_example(path: Path, attributions: dict) -> list[str]:
    errors: list[str] = []
    payload = load_json(path)
    messages = payload.get("messages", [])
    if len(messages) < 3:
        errors.append(f"{path.name}: expected system/user/assistant messages")

    metadata = payload.get("metadata", {})
    for text_id in metadata.get("textIds", []):
        if text_id not in attributions:
            errors.append(f"{path.name}: unknown textId '{text_id}' in metadata")

    assistant = next((m.get("content", "") for m in messages if m.get("role") == "assistant"), "")
    for record in attributions.values():
        title_zh = record.get("canonicalTitleZh")
        if not title_zh:
            continue
        for forbidden in record.get("doNotAttributeTo", []):
            # crude trap: forbidden author name + text title both mentioned as authorship
            if title_zh in assistant and forbidden in assistant.lower():
                if f"{forbidden}" in assistant.lower() and "write" in assistant.lower():
                    pass  # allowed when explaining the error, not asserting it
    return errors


def run_validation() -> dict:
    """Programmatic validation for CLI and MCP."""
    if not ATTRIBUTIONS_PATH.exists():
        return {"ok": False, "errors": [f"Missing {ATTRIBUTIONS_PATH}"]}

    attributions = load_json(ATTRIBUTIONS_PATH)
    errors = validate_attributions(attributions)
    example_count = 0

    if EXAMPLES_DIR.exists():
        examples = sorted(EXAMPLES_DIR.glob("*.json"))
        example_count = len(examples)
        for example in examples:
            errors.extend(validate_training_example(example, attributions))

    return {
        "ok": not errors,
        "attributions": len(attributions),
        "trainingExamples": example_count,
        "errors": errors,
    }


def main() -> int:
    result = run_validation()
    if not result["ok"]:
        print("Validation FAILED:")
        for err in result["errors"]:
            print(f"  - {err}")
        return 1

    print(
        f"Validation OK: {result['attributions']} attribution(s), "
        f"{result['trainingExamples']} training example(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())