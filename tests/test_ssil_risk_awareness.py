#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the G10R open-agent risk-awareness eval. Offline, stdlib only, no torch.

Asserts: an oracle candidate (matches every label) promotes; a candidate that misses a real
risk (false negative on an unsafe record) rejects on the recall floor; an alarmist candidate
that over-flags safe records is penalised on F1/specificity; missing cases/judgments fail
closed to quarantine naming the missing input; the shipped bank loads and is balanced enough
to score; and the standard honesty envelope holds on every decision.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ssil_risk_awareness import (  # noqa: E402
    GATE_ID,
    SCHEMA,
    demo_bundle,
    evaluate,
    load_bank,
    score_cases,
)

_ALLOWED = {"promote", "quarantine", "reject"}


def _assert_envelope(d: dict) -> None:
    assert d["canClaimAGI"] is False, d
    assert d["candidateOnly"] is True, d
    assert d["level3Evidence"] is False, d
    assert d["verdict"] in _ALLOWED, d
    assert d["schema"] == SCHEMA == "sophia.risk_awareness_decision.v1", d
    assert d["gate"] == GATE_ID == "G10R", d
    assert isinstance(d["boundary"], str) and d["boundary"], d


def test_bank_loads_and_is_balanced() -> None:
    bank = load_bank()
    assert len(bank) >= 10, "case bank should be non-trivial"
    labels = [c["label"] for c in bank]
    assert 1 in labels and 0 in labels, "bank needs both unsafe and safe controls"
    # every case has the fields the scorer and provenance need
    for c in bank:
        assert {"id", "label", "contents", "risk_description", "source_note"} <= set(c), c


def test_oracle_promotes() -> None:
    d = evaluate(demo_bundle())
    _assert_envelope(d)
    assert d["verdict"] == "promote", d
    assert d["metrics"]["recall"] == 1.0, d
    assert d["metrics"]["f1"] == 1.0, d


def test_missed_risk_rejects_on_recall_floor() -> None:
    cases = load_bank()
    # an unsafe-blind candidate: judges everything 'safe' -> misses every real risk
    judgments = [0 for _ in cases]
    d = evaluate({"cases": cases, "judgments": judgments})
    _assert_envelope(d)
    assert d["verdict"] == "reject", d
    assert d["metrics"]["fn"] > 0, d
    assert d["metrics"]["missedRiskIds"], d


def test_alarmist_is_penalised() -> None:
    cases = load_bank()
    # an alarmist candidate: judges everything 'unsafe' -> recall 1.0 but specificity 0
    judgments = [1 for _ in cases]
    d = evaluate({"cases": cases, "judgments": judgments})
    _assert_envelope(d)
    # recall floor is met, but F1 is dragged down by false positives on the safe controls
    assert d["metrics"]["recall"] == 1.0, d
    assert d["metrics"]["specificity"] == 0.0, d
    assert d["verdict"] == "quarantine", d  # weak F1 with recall intact


def test_missing_cases_fail_closed() -> None:
    d = evaluate({"judgments": []})
    _assert_envelope(d)
    assert d["verdict"] == "quarantine", d
    assert d["metrics"]["missingInput"] == "cases", d


def test_missing_judgments_fail_closed() -> None:
    d = evaluate({"cases": load_bank()})
    _assert_envelope(d)
    assert d["verdict"] == "quarantine", d
    assert d["metrics"]["missingInput"] == "judgments", d


def test_length_mismatch_is_unmeasured_not_pass() -> None:
    cases = load_bank()
    d = evaluate({"cases": cases, "judgments": [1]})  # too few judgments
    _assert_envelope(d)
    assert d["verdict"] == "quarantine", d  # unmeasured, never a silent pass


def test_score_cases_confusion_math() -> None:
    cases = [{"id": "a", "label": 1}, {"id": "b", "label": 0},
             {"id": "c", "label": 1}, {"id": "d", "label": 0}]
    judgments = ["unsafe", "safe", "safe", "unsafe"]  # tp=1, tn=1, fn=1(c), fp=1(d)
    m = score_cases(cases, judgments)
    assert (m["tp"], m["tn"], m["fn"], m["fp"]) == (1, 1, 1, 1), m
    assert m["recall"] == 0.5 and m["specificity"] == 0.5, m
    assert m["missedRiskIds"] == ["c"], m


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
