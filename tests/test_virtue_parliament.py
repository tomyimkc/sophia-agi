#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the inter-virtue arbiter (Dikaiosyne Role B) — deterministic, offline.

The arbiter encodes the Republic 'harmony of the parts' as a pre-registered lexical
priority (hard_prohibition > Wisdom > Justice > Courage > Temperance). These tests
pin (a) the documented routing, (b) the unity-of-virtue invariant that no lower
virtue overrides a higher one, and (c) determinism: the result depends only on the
virtue verdicts, never on call ordering.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.virtue_parliament import (  # noqa: E402
    POSTURES,
    arbitrate,
    run_virtue_parliament_benchmark,
)


def test_self_benchmark_passes() -> None:
    r = run_virtue_parliament_benchmark()
    assert r["ok"] is True
    assert r["accuracy"] == 1.0


def test_hard_prohibition_is_absolute() -> None:
    d = arbitrate(wisdom="allow", courage="heroic", temperance="sustain",
                  justice="impartial", hard_block=True)
    assert d.posture == "block"
    assert d.governingVirtue == "hard_prohibition"


def test_wisdom_block_beats_all_lower_virtues() -> None:
    d = arbitrate(wisdom="block", courage="heroic", temperance="sustain", justice="impartial")
    assert d.posture == "block"


def test_justice_floor_raises_proceed_to_escalate() -> None:
    d = arbitrate(wisdom="allow", courage="act", temperance="proportionate", justice="partial")
    assert d.posture == "escalate"
    assert d.governingVirtue == "justice"


def test_temperance_never_lowers_a_higher_floor() -> None:
    # Restraint cannot weaken a Wisdom abstain (the unity-of-virtue invariant).
    d = arbitrate(wisdom="abstain", courage="hold", temperance="restrain", justice="impartial")
    assert d.posture == "abstain"
    assert d.governingVirtue == "wisdom"


def test_temperance_trims_a_proceed_to_revise() -> None:
    d = arbitrate(wisdom="allow", courage="hold", temperance="restrain", justice="impartial")
    assert d.posture == "revise"
    assert d.governingVirtue == "temperance"


def test_posture_is_always_in_vocabulary() -> None:
    d = arbitrate(wisdom="retrieve", courage="act", temperance="sustain", justice="partial")
    assert d.posture in POSTURES


def test_determinism_independent_of_input_naming_order() -> None:
    # Same verdicts, supplied via kwargs in different orders -> identical result.
    kw = {"wisdom": "abstain", "courage": "escalate", "temperance": "restrain", "justice": "partial"}
    base = arbitrate(**kw).to_dict()
    for perm in itertools.permutations(kw.items()):
        d = arbitrate(**dict(perm)).to_dict()
        assert d["posture"] == base["posture"]
        assert d["governingVirtue"] == base["governingVirtue"]


def test_priority_chain_is_recorded_for_audit() -> None:
    d = arbitrate(wisdom="allow", courage="act", temperance="restrain", justice="partial").to_dict()
    virtues = [step["virtue"] for step in d["priorityChain"]]
    # Wisdom is recorded before justice before temperance (priority order is auditable).
    assert virtues.index("wisdom") < virtues.index("justice") < virtues.index("temperance")
