#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Prompt-quality verifier: checkable prompt scoring, no-overclaim reuse, fail-closed bar.

Deterministic and offline — no model, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.prompt_quality_verifier import (  # noqa: E402
    offline_invariants,
    prompt_quality,
    prompt_quality_ok,
    score_prompt,
)


def test_offline_invariants_pass() -> None:
    ok, detail = offline_invariants()
    assert ok, detail["checks"]


def test_well_formed_prompt_passes_all_required() -> None:
    p = ("On branch glm/x ONLY, do not touch agent/gate.py. Success = `make claim-check` GO and "
         "tests/test_x.py passes. If blocked, report NO-GO; canClaimAGI stays false.")
    s = score_prompt(p)
    assert s["passed"]
    for d in ("success_criterion", "bounded_scope", "abstention_path", "no_overclaim"):
        assert s["dimensions"][d], d


def test_vague_prompt_fails() -> None:
    s = score_prompt("Make it better.")
    assert not s["passed"]
    assert any("success_criterion" in r for r in s["reasons"])


def test_overclaim_reuses_lint_claims_patterns() -> None:
    s = score_prompt("Ship the world's first proven AGI; it's a breakthrough that makes AI safe.")
    assert s["dimensions"]["no_overclaim"] is False
    assert s["overclaims"]


def test_claim_ok_marker_exempts() -> None:
    s = score_prompt("world's first AGI — illustrative framing only. claim-ok. "
                     "Success = test passes; only on branch x; if blocked, abstain.")
    assert s["dimensions"]["no_overclaim"] is True


def test_predicate_and_verifier_callable_agree() -> None:
    good = ("Success = CI gate GO; scope: only docs/; if unsure, ask first. See docs/x.md.")
    assert prompt_quality_ok(good) is True
    v = prompt_quality()
    out = v(good)
    assert out["passed"] is True and "detail" in out


def test_grounding_is_recommended_not_fatal() -> None:
    # No file/path reference, but the four load-bearing dims present -> still passes.
    p = ("Success = the test passes and the gate is GO. Do not change scope beyond the one fix. "
         "If blocked, report NO-GO and abstain.")
    s = score_prompt(p)
    assert s["passed"] is True
    assert s["dimensions"]["grounding"] is False
    assert any("grounding" in r for r in s["reasons"])


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} prompt_quality_verifier tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
