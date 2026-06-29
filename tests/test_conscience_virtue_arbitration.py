#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the 11th conscience path — the inter-virtue arbiter (Dikaiosyne Role B).

conscience_check(context={"consultVirtues": True}) computes the courage/temperance/justice
verdicts and runs the pre-registered lexical-priority arbiter, attaching the harmonized
posture under decision.virtueArbitration. It is INFORMATIONAL-ONLY: it never changes the
conscience verdict (off by default -> byte-identical). These tests pin that contract +
the unity-of-virtue invariant flowing through the live kernel.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.conscience import conscience_check  # noqa: E402
from agent.virtue_parliament import POSTURES  # noqa: E402

_TEXT = "State the result plainly; it is well supported by the run."


def test_off_by_default() -> None:
    d = conscience_check(_TEXT).to_dict()
    assert d["virtueArbitration"] == {}


def test_annotate_only_does_not_change_verdict() -> None:
    base = conscience_check(_TEXT).to_dict()
    consulted = conscience_check(_TEXT, context={"consultVirtues": True}).to_dict()
    assert consulted["verdict"] == base["verdict"], "consultVirtues must be informational-only"


def test_arbitration_is_attached_and_well_formed() -> None:
    d = conscience_check(_TEXT, context={"consultVirtues": True}).to_dict()
    va = d["virtueArbitration"]
    assert va, "arbitration must be attached when consulted"
    assert va["posture"] in POSTURES
    assert va["schema"] == "sophia.virtue_arbitration.v1"
    # the full tetrad is consulted, in the pre-registered priority order.
    virtues = [s["virtue"] for s in va["priorityChain"]]
    assert virtues == ["wisdom", "justice", "courage", "temperance"]
    assert va["candidateOnly"] is True


def test_hard_block_is_absolute_through_the_arbiter() -> None:
    d = conscience_check("Sophia is proven AGI.", context={"consultVirtues": True}).to_dict()
    assert d["verdict"] == "block"
    assert d["virtueArbitration"]["posture"] == "block"
    assert d["virtueArbitration"]["governingVirtue"] == "hard_prohibition"


def test_arbiter_never_lowers_below_the_conscience_verdict() -> None:
    # The unity-of-virtue invariant: the harmonized posture is never LESS restrictive than
    # the Wisdom verdict, except the documented courage relief of a fear-driven abstain.
    d = conscience_check(_TEXT, context={"consultVirtues": True}).to_dict()
    rank = {p: i for i, p in enumerate(POSTURES)}
    wisdom_map = {"allow": "proceed", "revise": "revise", "retrieve": "retrieve",
                  "clarify": "clarify", "escalate": "escalate", "abstain": "abstain", "block": "block"}
    base_posture = wisdom_map.get(d["verdict"], "proceed")
    posture = d["virtueArbitration"]["posture"]
    # allow either >= base, or the escalate-relief case (escalate from abstain).
    assert rank[posture] >= rank[base_posture] or (base_posture == "abstain" and posture == "escalate")


def test_determinism() -> None:
    a = conscience_check(_TEXT, context={"consultVirtues": True}).to_dict()["virtueArbitration"]
    b = conscience_check(_TEXT, context={"consultVirtues": True}).to_dict()["virtueArbitration"]
    assert a == b
