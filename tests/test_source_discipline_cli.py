#!/usr/bin/env python3
"""Tests for tools/source_discipline_cli.py — the source-discipline checker CLI.

Offline: reuses agent.verifiers.provenance_faithful (a local-regex check, no model,
no network). Proves the Sophia source-discipline rule survives the CLI boundary that
the OpenClaw before_agent_finalize plugin calls across.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.source_discipline_cli import check  # noqa: E402


def test_forbidden_attribution_fails() -> None:
    out = check("Confucius wrote the Dao De Jing.")
    assert out["passed"] is False
    assert out["violations"], "expected at least one violation"


def test_negation_passes() -> None:
    # the carve-out: a page that CORRECTLY debunks the merge must pass
    out = check("Confucius did not write the Dao De Jing.")
    assert out["passed"] is True
    assert out["violations"] == []


def test_benign_text_passes() -> None:
    out = check("The lecture covered hexagonal grid pathfinding in Godot.")
    assert out["passed"] is True


def test_result_shape() -> None:
    out = check("hello there")
    assert set(out) >= {"passed", "reasons", "violations"}
    assert isinstance(out["reasons"], list) and isinstance(out["violations"], list)


def main() -> int:
    test_forbidden_attribution_fails()
    test_negation_passes()
    test_benign_text_passes()
    test_result_shape()
    print("test_source_discipline_cli: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
