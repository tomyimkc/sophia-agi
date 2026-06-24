#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the cross-entity generalization benchmark.

The falsifiable properties: an entity-disjoint split (no shared author/work);
memorized rules do not transfer to unseen entities (≈0 recall) though precise
(≈0 FP); a structural detector transfers (high recall) but is imprecise (high FP).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import cross_entity as ce  # noqa: E402

PAIRS = [
    {"claimed": "Alice", "work": "Alpha"},
    {"claimed": "Bob", "work": "Beta"},
    {"claimed": "Carol", "work": "Gamma"},
    {"claimed": "Dave", "work": "Delta"},
    {"claimed": "Erin", "work": "Epsilon"},
    {"claimed": "Frank", "work": "Zeta"},
]
CONTROLS = [{"gold": "Zoe", "work": "Omega"}, {"gold": "Yan", "work": "Psi"}]


def test_split_is_entity_disjoint() -> None:
    train, test = ce.entity_disjoint_split(PAIRS, seed=0)
    assert train and test
    assert not ({p["claimed"] for p in train} & {p["claimed"] for p in test})
    assert not ({p["work"] for p in train} & {p["work"] for p in test})
    assert len(train) + len(test) == len(PAIRS)


def test_asserts_attribution_is_entity_agnostic() -> None:
    assert ce._asserts_attribution("Alice is the author of Alpha.") is True
    assert ce._asserts_attribution("Omega is attributed to Zoe.") is True
    assert ce._asserts_attribution("The weather is nice today.") is False


def test_memorized_does_not_transfer_structural_does_but_imprecise() -> None:
    r = ce.run_cross_entity(PAIRS, CONTROLS, seed=0)
    assert r["entityDisjoint"] is True
    assert r["withinEntityRecall"] >= 0.8           # works on seen entities
    assert r["crossEntityRecall_memorized"] <= 0.1  # does NOT transfer
    assert r["memorizedFalsePositive"] == 0.0       # but is precise
    assert r["crossEntityRecall_structural"] >= 0.8  # structure transfers
    assert r["structuralFalsePositive"] >= 0.5      # ... but is imprecise


def main() -> int:
    test_split_is_entity_disjoint()
    test_asserts_attribution_is_entity_agnostic()
    test_memorized_does_not_transfer_structural_does_but_imprecise()
    print("test_cross_entity: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
