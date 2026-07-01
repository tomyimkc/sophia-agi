#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the multi-judge labeling panel. Offline, stdlib only, no torch."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_judge_panel import majority_consensus, panel_verdict, pairwise_agreement, run_panel  # noqa: E402


def _items(n: int) -> list[dict]:
    return [{"id": f"i{i}"} for i in range(n)]


def test_majority_consensus_and_ties() -> None:
    # 3 judges, item 0 unanimous, item 1 split 2-1, item 2 three-way-ish tie (binary can't 3-way,
    # so force a tie with 2 judges elsewhere). Here all 3 binary -> always a majority.
    labels = [["safe", "unsafe"], ["safe", "unsafe"], ["safe", "safe"]]
    consensus, unresolved = majority_consensus(labels)
    assert consensus == ["safe", "unsafe"], consensus
    assert unresolved == 0, unresolved


def test_two_judge_tie_is_unresolved() -> None:
    labels = [["safe"], ["unsafe"]]  # 2 judges disagree -> tie
    consensus, unresolved = majority_consensus(labels)
    assert consensus == [None] and unresolved == 1


def test_high_agreement_panel_go() -> None:
    # 3 near-identical judges over 12 items -> high kappa, no ties -> GO
    base = ["unsafe", "safe"] * 6
    judges = {"a": list(base), "b": list(base), "c": list(base[:-1] + ["unsafe"])}
    out = run_panel(_items(12), judges, seed=0)
    assert out["verdict"] == "GO", out
    assert out["unresolved"] == 0, out
    assert all(row["consensusLabel"] is not None for row in out["labeledSet"]), out


def test_coin_flip_panel_no_go() -> None:
    base = ["unsafe", "safe"] * 6
    coin = ["safe", "unsafe", "safe", "safe", "unsafe", "unsafe", "safe", "unsafe", "unsafe", "safe", "safe", "unsafe"]
    judges = {"a": list(base), "b": list(base), "coin": coin}
    out = run_panel(_items(12), judges, seed=0)
    assert out["verdict"] == "NO-GO", out
    assert any("low_agreement" in f for f in out["criticalFailures"]), out


def test_single_judge_is_not_a_panel() -> None:
    v = panel_verdict({"a": ["safe", "unsafe"]}, [], 0.0)
    assert v["verdict"] == "NO-GO", v
    assert any("not_2_families" in f for f in v["criticalFailures"]), v


def test_misaligned_lengths_raise() -> None:
    try:
        run_panel(_items(3), {"a": ["safe", "safe"], "b": ["safe", "safe", "safe"]})
    except ValueError:
        return
    raise AssertionError("expected ValueError on misaligned judge vectors")


def test_pairwise_reports_kappa_and_ac1() -> None:
    base = ["unsafe", "safe"] * 6
    pw = pairwise_agreement({"a": list(base), "b": list(base)}, seed=0)
    assert len(pw) == 1, pw
    assert pw[0]["kappa"] == 1.0 and pw[0]["ac1"] == 1.0, pw


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
