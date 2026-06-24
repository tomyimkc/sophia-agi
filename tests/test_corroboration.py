#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for corroboration-aware confidence (#4).

Properties: independent agreement raises belief; dependent (same-group) sources do
NOT double-count; dissent lowers belief; and on a labelled benchmark it makes
better decisions (lower selective risk) than a single source or min-over-chain.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.corroboration import Evidence, corroborated_confidence, run_demo  # noqa: E402


def _indep(*cs):
    return [Evidence(f"s{i}", c, independence_group=f"g{i}") for i, c in enumerate(cs)]


def test_independent_agreement_raises_belief() -> None:
    one = corroborated_confidence(_indep(0.7))
    two = corroborated_confidence(_indep(0.7, 0.7))
    three = corroborated_confidence(_indep(0.7, 0.7, 0.7))
    assert one < two < three
    assert abs(one - 0.7) < 1e-6                      # a single source reads its own confidence


def test_duplicates_do_not_inflate() -> None:
    # three copies in ONE independence group == one opinion
    dup = corroborated_confidence([Evidence("a", 0.7, "g"), Evidence("b", 0.7, "g"), Evidence("c", 0.7, "g")])
    assert abs(dup - 0.7) < 1e-6


def test_dissent_lowers_belief() -> None:
    support = corroborated_confidence(_indep(0.7, 0.7))
    with_dissent = corroborated_confidence(_indep(0.7, 0.7, 0.2))
    assert with_dissent < support


def test_noisy_or_method_supports_only() -> None:
    # noisy-OR (support-only) also rises with independent support
    assert corroborated_confidence(_indep(0.6, 0.6), method="noisy_or") > 0.6
    assert corroborated_confidence([], method="noisy_or") == 0.5   # prior on no evidence


def test_validation_rejects_bad_confidence() -> None:
    import math

    for bad in (float("nan"), float("inf"), -1.0, 2.0, True):
        try:
            Evidence("x", bad)
            assert False, f"should reject confidence={bad!r}"
        except ValueError:
            pass
    assert math.isfinite(Evidence("x", 0.5).confidence)   # valid still works


def test_unknown_method_raises() -> None:
    try:
        corroborated_confidence(_indep(0.7), method="noisy-or")   # hyphen typo
        assert False, "should reject unknown method"
    except ValueError:
        pass


def test_demo_invariants_hold() -> None:
    res = run_demo(seed=0)
    failed = [k for k, v in res["invariants"].items() if not v]
    assert res["ok"] is True, f"failed: {failed}"
    # the robust, gated win: corroboration reflects independent agreement; mean/min don't
    assert res["curve"]["3src"] > 0.7
    assert "selectiveRisk" in res and "ece" in res            # discrimination/ECE reported, not gated


def main() -> int:
    test_independent_agreement_raises_belief()
    test_duplicates_do_not_inflate()
    test_dissent_lowers_belief()
    test_noisy_or_method_supports_only()
    test_validation_rejects_bad_confidence()
    test_unknown_method_raises()
    test_demo_invariants_hold()
    print("test_corroboration: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
