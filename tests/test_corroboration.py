#!/usr/bin/env python3
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


def test_demo_invariants_hold() -> None:
    res = run_demo(seed=0)
    failed = [k for k, v in res["invariants"].items() if not v]
    assert res["ok"] is True, f"failed: {failed}"
    # the durable win is discrimination
    assert res["selectiveRisk"]["corroborated"] < res["selectiveRisk"]["single"]
    assert res["selectiveRisk"]["corroborated"] < res["selectiveRisk"]["min"]


def main() -> int:
    test_independent_agreement_raises_belief()
    test_duplicates_do_not_inflate()
    test_dissent_lowers_belief()
    test_noisy_or_method_supports_only()
    test_demo_invariants_hold()
    print("test_corroboration: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
