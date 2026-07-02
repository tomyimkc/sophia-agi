# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic, download-free tests for the NLI entailment backend (injected scorer).

Uses local duck-typed fakes for claim/source so the test does NOT import agent.fact_check_gate
(which requires py3.11+ for a possessive-quantifier regex) and stays runnable on any Python.
agent.nli_grounding reads claim.text / source.title / source.snippet via getattr, so these
fakes exercise the real contract.
"""
from __future__ import annotations

from dataclasses import dataclass

from agent.nli_grounding import build_nli_entailment


@dataclass
class _Claim:
    text: str = "X wrote Y"
    type: str = "authorship_external"
    risk: str = "normal"


@dataclass
class _Src:
    title: str = ""
    snippet: str = ""
    id: str = "e1"


def test_maps_entailment_to_entails():
    fn = build_nli_entailment(scorer=lambda p, h: (0.01, 0.98, 0.01))
    assert fn(_Claim(), _Src(title="record", snippet="Y was written by X")) == "entails"


def test_maps_contradiction_to_contradicts():
    fn = build_nli_entailment(scorer=lambda p, h: (0.97, 0.01, 0.02))
    assert fn(_Claim(), _Src(title="record", snippet="Y was written by Z")) == "contradicts"


def test_neutral_is_irrelevant():
    fn = build_nli_entailment(scorer=lambda p, h: (0.05, 0.10, 0.85))
    assert fn(_Claim(), _Src(snippet="X was born in 1952")) == "irrelevant"


def test_below_threshold_is_irrelevant_not_overconfident():
    fn = build_nli_entailment(scorer=lambda p, h: (0.30, 0.40, 0.30), entail_threshold=0.5)
    assert fn(_Claim(), _Src(snippet="loosely related")) == "irrelevant"


def test_empty_evidence_or_claim_is_irrelevant():
    fn = build_nli_entailment(scorer=lambda p, h: (0.0, 1.0, 0.0))
    assert fn(_Claim(), _Src()) == "irrelevant"                       # no premise
    assert fn(_Claim(text=""), _Src(snippet="anything")) == "irrelevant"  # no hypothesis


def test_returns_only_contract_labels():
    fn = build_nli_entailment(scorer=lambda p, h: (0.4, 0.5, 0.1))
    assert fn(_Claim(), _Src(snippet="evidence")) in {"entails", "contradicts", "irrelevant"}
