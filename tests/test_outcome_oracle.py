#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""T2: the single outcome oracle must agree with agent.verifiers (one definition of
'correct' for both the RL reward and the SFT filter), offline and deterministic."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import outcome_oracle as oracle  # noqa: E402
from agent import verifiers as V  # noqa: E402


def test_keyword_axis_agrees_with_verifier() -> None:
    spec = {"mustInclude": ["laozi"], "mustAvoid": ["confucius wrote"]}
    good = "The Dao De Jing is attributed to Laozi."
    bad = "Confucius wrote the Dao De Jing."
    for text in (good, bad):
        direct = V.keyword(must_include=["laozi"], must_avoid=["confucius wrote"])(text, None, {})
        via = oracle.evaluate(spec, text)
        assert direct["passed"] == via["passed"], (text, direct, via)
    assert oracle.evaluate(spec, good)["passed"] is True
    assert oracle.evaluate(spec, bad)["passed"] is False


def test_math_axis_agrees_with_verifier() -> None:
    # Parity is the T2 invariant: the oracle's verdict must equal the verifier's, whatever
    # it is (when sympy is absent both fail closed — still in agreement).
    spec = {"mathEquivalent": "x**2 - 1"}
    for text in ("the answer is (x-1)*(x+1)", "the answer is x**2 + 1"):
        direct = V.math_equivalent("x**2 - 1")(text, None, {})
        via = oracle.evaluate(spec, text)
        assert direct["passed"] == via["passed"], (text, direct, via)


def test_checks_map_names_the_failing_axis() -> None:
    spec = {"mustInclude": ["laozi"], "regex": r"\bdao de jing\b"}
    verdict = oracle.evaluate(spec, "Some unrelated text.")
    # both axes present; both should be False here
    assert verdict["passed"] is False
    # axes are named by the verifier factory (keyword, regex_match, ...)
    assert set(verdict["checks"]) == {"keyword", "regex_match"}
    assert verdict["checks"]["keyword"] is False


def test_reward_is_bounded_pass_fail() -> None:
    spec = {"mustInclude": ["laozi"]}
    assert oracle.reward(spec, "Laozi wrote it") == 1.0
    assert oracle.reward(spec, "nope") == -1.0


def main() -> int:
    test_keyword_axis_agrees_with_verifier()
    test_math_axis_agrees_with_verifier()
    test_checks_map_names_the_failing_axis()
    test_reward_is_bounded_pass_fail()
    print("test_outcome_oracle: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
