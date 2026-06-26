# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Hurdle 2 — the identical scorer path works across structurally different families.

Proves the transfer claim's mechanical half: one ``score_for_domain`` dispatch scores
provenance, math, and coding through their own SOUND verifiers, accepting correct
answers and rejecting wrong ones in each. This is harness evidence, not a model run.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.benchmark_checks import DOMAIN_BENCH, load_json
from agent.domain_scorers import KIND_SCORERS, kind_for, score_for_domain

ROOT = Path(__file__).resolve().parents[1]


def test_kinds_registered():
    for kind in ("provenance", "math", "coding"):
        assert kind in KIND_SCORERS


def test_domain_routing_defaults():
    assert kind_for("philosophy") == "provenance"
    assert kind_for("math") == "math"
    assert kind_for("coding") == "coding"
    # explicit case kind overrides the domain default
    assert kind_for("philosophy", {"kind": "math"}) == "math"
    # unknown domain falls back to provenance (back-compat)
    assert kind_for("unknown_domain") == "provenance"


def test_math_scorer_accepts_correct_rejects_wrong():
    case = {"id": "m1", "kind": "math", "question": "What is 17 plus 25? End with 'answer = <number>'.",
            "expectedAnswer": 42}
    ok, _ = score_for_domain("math", case, "The sum is 42. answer = 42", ctx={})
    assert ok
    bad, reasons = score_for_domain("math", case, "answer = 41", ctx={})
    assert not bad and reasons


def test_math_scorer_soundness_veto():
    # Correct final answer but a FALSE equality in the working must still fail.
    case = {"id": "m2", "kind": "math", "expectedAnswer": 42}
    ok, reasons = score_for_domain("math", case, "Since 2 + 2 = 5, ... answer = 42", ctx={})
    assert not ok
    assert any("false arithmetic" in r for r in reasons)


def test_coding_scorer_accepts_passing_rejects_broken():
    # A syntax error fails in BOTH execute and syntax-only modes, so this assertion
    # holds regardless of SOPHIA_ALLOW_CODE_EXEC.
    good = "```python\nassert 2 + 2 == 4\n```"
    broken = "```python\ndef f(:\n    pass\n```"
    case = {"id": "c1", "kind": "coding", "timeoutSec": 5}
    ok, _ = score_for_domain("coding", case, good, ctx={"allow_execution": True})
    assert ok
    failed, reasons = score_for_domain("coding", case, broken, ctx={"allow_execution": True})
    assert not failed and reasons


def test_coding_scorer_rejects_runtime_failure_when_executing():
    # Only meaningful when execution is enabled; a passing assert vs a failing one.
    import os

    if os.environ.get("SOPHIA_ALLOW_CODE_EXEC", "1").strip() in ("0", "false", "no"):
        return  # syntax-only mode cannot distinguish runtime failures
    bad = "```python\nassert 2 + 2 == 5\n```"
    case = {"id": "c2", "kind": "coding", "timeoutSec": 5}
    failed, reasons = score_for_domain("coding", case, bad, ctx={"allow_execution": True})
    assert not failed and reasons


def test_provenance_scorer_still_works():
    # A deny-attribution trap: must mention the author AND deny the attribution.
    case = {"id": "p1", "mustDenyAttribution": {"author": "confucius"}}
    ok, _ = score_for_domain(
        "philosophy", case,
        "Confucius did not write the Analects; it was compiled by his disciples.",
        ctx={"traditions": {}},
    )
    assert ok
    miss, reasons = score_for_domain("philosophy", case, "The Analects is a great book.", ctx={"traditions": {}})
    assert not miss and reasons


def test_new_benchmarks_present_and_routable():
    for domain in ("math", "coding"):
        path = DOMAIN_BENCH[domain]
        assert path.exists(), f"missing benchmark for {domain}"
        bench = load_json(path)
        cases = bench.get("cases", [])
        assert cases, f"{domain} benchmark has no cases"
        for case in cases:
            assert kind_for(domain, case) == domain  # math->math, coding->coding


def test_full_packs_score_with_committed_responses():
    """Every committed coding response passes; every math case scores cleanly on its own answer."""
    coding = load_json(DOMAIN_BENCH["coding"])
    for case in coding["cases"]:
        ok, reasons = score_for_domain("coding", case, case["response"], ctx={"allow_execution": True})
        assert ok, f"coding case {case['id']} failed: {reasons}"

    math = load_json(DOMAIN_BENCH["math"])
    for case in math["cases"]:
        resp = f"answer = {case['expectedAnswer']}"
        ok, reasons = score_for_domain("math", case, resp, ctx={})
        assert ok, f"math case {case['id']} failed: {reasons}"
