#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent/structured_output.py — Stage D structured-output discipline.

Falsifiable invariants:
  1. A well-formed tool call / belief write validates clean.
  2. Missing required, bad enum, too-short string/array, and disallowed extra
     property are all caught (fail-closed before dispatch).
  3. Malformed JSON is an error, never silently treated as empty.
  4. The bundled schemas load and the GBNF emitter produces a grammar whose
     prelude + a root rule are present (constrained-decoding bridge).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.structured_output import (  # noqa: E402
    is_valid,
    load_schema,
    parse_and_validate,
    schema_to_gbnf,
    validate_belief_write,
    validate_tool_call,
)


def test_valid_tool_call() -> None:
    assert validate_tool_call({"tool_id": "kb.lookup", "args": {}}) == []
    assert validate_tool_call({"tool_id": "kb.lookup", "args": {"q": "x"},
                               "role": "role_09_agents", "clearance": "SECRET"}) == []


def test_tool_call_missing_required() -> None:
    errs = validate_tool_call({"args": {}})
    assert any("tool_id" in e for e in errs)


def test_tool_call_bad_enum() -> None:
    errs = validate_tool_call({"tool_id": "x", "args": {}, "clearance": "NOPE"})
    assert any("enum" in e for e in errs)


def test_tool_call_additional_property_rejected() -> None:
    errs = validate_tool_call({"tool_id": "x", "args": {}, "surprise": 1})
    assert any("additional property" in e for e in errs)


def test_belief_write_requires_sources() -> None:
    assert validate_belief_write({"content": "c", "sources": ["s"], "blp_level": "UNCLASSIFIED"}) == []
    errs = validate_belief_write({"content": "c", "sources": [], "blp_level": "UNCLASSIFIED"})
    assert any("minItems" in e for e in errs)


def test_belief_write_empty_content_rejected() -> None:
    errs = validate_belief_write({"content": "", "sources": ["s"], "blp_level": "UNCLASSIFIED"})
    assert any("minLength" in e for e in errs)


def test_malformed_json_is_error() -> None:
    obj, errs = parse_and_validate("{not json", load_schema("gateway_call"))
    assert obj is None
    assert errs and "invalid JSON" in errs[0]


def test_gbnf_has_prelude_and_root() -> None:
    g = schema_to_gbnf(load_schema("gateway_call"))
    assert "ws ::=" in g
    assert "string ::=" in g
    assert g.count("root ::=") == 1
    # enum values from belief schema must appear as quoted terminals
    gb = schema_to_gbnf(load_schema("belief_write"))
    assert '"\\"UNCLASSIFIED\\""' in gb


def test_is_valid_shortcut() -> None:
    assert is_valid({"tool_id": "x", "args": {}}, load_schema("gateway_call")) is True
    assert is_valid({"args": {}}, load_schema("gateway_call")) is False


def main() -> int:
    import inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_structured_output: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
