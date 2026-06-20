#!/usr/bin/env python3
"""Tests for the mode-free check_claim MCP tool (sophia_mcp). All offline.

Unlike sophia_gate_check (moded, needs a question), check_claim is a pure
provenance verifier surface: text in -> {passed, reasons, violations}.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sophia_mcp import tools_impl  # noqa: E402


def test_forbidden_attribution_flagged() -> None:
    out = tools_impl.check_claim("Confucius wrote the Dao De Jing.")
    assert out["passed"] is False
    assert out["violations"]


def test_correction_passes() -> None:
    out = tools_impl.check_claim("Confucius did not write the Dao De Jing.")
    assert out["passed"] is True


def test_benign_passes() -> None:
    out = tools_impl.check_claim("The library opens at nine in the morning.")
    assert out["passed"] is True
    assert out["violations"] == []


def test_shape_is_stable() -> None:
    out = tools_impl.check_claim("anything")
    assert set(out) == {"passed", "reasons", "violations"}
    assert isinstance(out["reasons"], list) and isinstance(out["violations"], list)


def main() -> int:
    test_forbidden_attribution_flagged()
    test_correction_passes()
    test_benign_passes()
    test_shape_is_stable()
    print("test_mcp_check_claim: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
