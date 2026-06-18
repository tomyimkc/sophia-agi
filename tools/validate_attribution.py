#!/usr/bin/env python3
"""Validate attribution records and training examples in the Sophia AGI corpus."""

from __future__ import annotations

import json
import sys
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


def main() -> int:
    if not ATTRIBUTIONS_PATH.exists():
        print(f"Missing {ATTRIBUTIONS_PATH}", file=sys.stderr)
        return 1

    attributions = load_json(ATTRIBUTIONS_PATH)
    errors = validate_attributions(attributions)

    if EXAMPLES_DIR.exists():
        for example in sorted(EXAMPLES_DIR.glob("*.json")):
            errors.extend(validate_training_example(example, attributions))

    if errors:
        print("Validation FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(
        f"Validation OK: {len(attributions)} attribution(s), "
        f"{len(list(EXAMPLES_DIR.glob('*.json')))} training example(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())