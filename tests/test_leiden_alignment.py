#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the Leiden Declaration alignment layer.

The Leiden Declaration on AI and Mathematics asks for tool disclosure, human-retained
responsibility, FAIR adherence, and honesty about what is not yet done. This repo's posture
is to make those checkable rather than asserted. These tests pin the load-bearing invariants:

  (a) the generated Tool & Computational Resource Disclosure is deterministic and its
      --check gate is in sync with the committed file;
  (b) the Leiden compliance receipt is deterministic, names all five values, declares
      open gaps, and stays at canClaimAGI:false;
  (c) the FAIR self-assessment is fail-closed (unknown license lowers 'reusable');
  (d) the no-AI-authorship lint actually rejects an AI-authorship sentence;
  (e) every Leiden artifact carries canClaimAGI:false (no overclaim sneaks in here).

Pure stdlib, deterministic, offline — safe for the fast-ci core lane.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_tool_disclosure, leiden_receipt, lint_claims  # noqa: E402
from pretraining.data_passport.passport import stamp_pack, fair_assessment  # noqa: E402


def test_tool_disclosure_deterministic_and_in_sync() -> None:
    """render() is pure; the committed doc matches --check."""
    a = build_tool_disclosure.render()
    b = build_tool_disclosure.render()
    assert a == b, "tool-disclosure render is not deterministic"
    assert "Tool &amp; Computational Resource Disclosure" in a
    assert "canClaimAGI: false" in a
    assert build_tool_disclosure.main(["--check"]) == 0, (
        "docs/TOOL-DISCLOSURE.md is stale — run python tools/build_tool_disclosure.py"
    )


def test_leiden_receipt_shape_and_in_sync() -> None:
    """The receipt names all five values, declares gaps, and is committed in sync."""
    rec = leiden_receipt.build()
    ids = {v["id"] for v in rec["values"]}
    assert ids == {
        "proof_and_certainty", "attribution_and_responsibility",
        "transparency_and_verifiability", "shared_standards", "autonomous_direction",
    }, f"unexpected value set: {ids}"
    for v in rec["values"]:
        assert v["status"] in ("operationalized", "partial"), v
    assert rec["canClaimAGI"] is False
    assert len(rec["open_gaps"]) >= 1
    # determinism
    assert json.dumps(leiden_receipt.build()) == json.dumps(rec)
    assert leiden_receipt.main(["--check"]) == 0, (
        "agi-proof/leiden-compliance.json is stale — run python tools/leiden_receipt.py"
    )


def test_fair_assessment_is_fail_closed() -> None:
    """An unknown license must lower 'reusable' below 1.0 (not silently pass)."""
    rows = [
        {"prompt": "a sufficiently long and unique training prompt example here",
         "completion": "answer", "source": "x", "license": "MIT"},
        {"prompt": "another distinct and adequately long training prompt text here",
         "completion": "answer two", "source": "y"},  # no license
    ]
    sheet = stamp_pack(rows)["datasheet"]
    fair = sheet["fair"]
    assert fair["findable"] == 1.0
    assert fair["reusable"] < 1.0, "unknown license did not lower reusability (not fail-closed)"
    # empty pack degrades to zeros, never raises
    assert fair_assessment([])["reusable"] == 0.0


def test_no_ai_authorship_lint_rejects() -> None:
    """The Leiden authorship rule must catch an explicit AI-authorship sentence."""
    import re
    bad = "this theorem was authored by claude and the model is the author".lower()
    hit = any(re.search(pat, bad) for pat, _ in lint_claims.FORBIDDEN)
    assert hit, "no-AI-authorship rule failed to flag an AI-authorship sentence"
    # a normal human-authorship sentence must NOT trip any rule
    ok = "this result was authored by the maintainer with ai assistance".lower()
    assert not any(re.search(pat, ok) for pat, _ in lint_claims.FORBIDDEN), (
        "authorship rule is over-broad (flagged a clean human-authorship sentence)"
    )


def test_no_overclaim_in_leiden_artifacts() -> None:
    """canClaimAGI:false must hold in both generated Leiden artifacts."""
    rec = json.loads((ROOT / "agi-proof" / "leiden-compliance.json").read_text(encoding="utf-8"))
    assert rec["canClaimAGI"] is False
    decl = json.loads((ROOT / "agi-proof" / "tool-disclosure.json").read_text(encoding="utf-8"))
    assert decl["canClaimAGI"] is False


def main() -> int:
    test_tool_disclosure_deterministic_and_in_sync()
    test_leiden_receipt_shape_and_in_sync()
    test_fair_assessment_is_fail_closed()
    test_no_ai_authorship_lint_rejects()
    test_no_overclaim_in_leiden_artifacts()
    print("test_leiden_alignment: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
