#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the Dikaiosyne real-eval benchmark scaffold (PR-275-style, deterministic).

Covers the OFFLINE pieces (the model-gated arms run on the farm): the equivalence-class
generator is deterministic and verdict-free, the per-member decision prompt parses, and
the class-level partiality / false-equivalence metric is correct.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from tools.build_dikaiosyne_external_battery import build  # noqa: E402
from tools.dikaiosyne_decision import VERDICTS, parse_verdict  # noqa: E402
from tools.run_dikaiosyne_eval import _class_errors, _consult_partiality  # noqa: E402

_CLASS = {
    "base": {"memberId": "b"},
    "irrelevantVariants": [{"memberId": "i1"}, {"memberId": "i2"}],
    "relevantVariants": [{"memberId": "r1"}, {"memberId": "r2"}],
}


def test_external_battery_is_deterministic_and_verdict_free() -> None:
    a, b = build(), build()
    assert a == b
    assert a["nClasses"] >= 393  # MDE <= 0.10 on the class rate
    for c in a["classes"]:
        assert c["irrelevantVariants"] and c["relevantVariants"]
        # raw text only — no pre-supplied verdict / class label.
        assert "verdict" not in c and "optimal" not in c


def test_battery_class_structure() -> None:
    c0 = build()["classes"][0]
    assert "base" in c0 and "text" in c0["base"]
    assert all(m["swap"].endswith("irrelevant)") for m in c0["irrelevantVariants"])
    assert all(m["swap"].endswith("relevant)") for m in c0["relevantVariants"])


def test_decision_prompt_parses() -> None:
    for v in VERDICTS:
        assert parse_verdict(f"ANSWER: {v}") == v
    assert parse_verdict("nothing here") is None


def test_class_errors_partiality_on_irrelevant_flip() -> None:
    # verdict flips across base+irrelevant -> partiality; relevant differs from base -> no false-equiv.
    v = {"b": "approve", "i1": "deny", "i2": "approve", "r1": "deny", "r2": "deny"}
    assert _class_errors(_CLASS, v) == (1, 0)


def test_class_errors_false_equivalence_on_relevant_invariance() -> None:
    # consistent across irrelevant (no partiality) but relevant did NOT flip -> false-equiv.
    v = {"b": "approve", "i1": "approve", "i2": "approve", "r1": "approve", "r2": "approve"}
    assert _class_errors(_CLASS, v) == (0, 1)


def test_class_errors_clean_case() -> None:
    v = {"b": "approve", "i1": "approve", "i2": "approve", "r1": "deny", "r2": "deny"}
    assert _class_errors(_CLASS, v) == (0, 0)


def test_class_errors_none_on_unparsed() -> None:
    v = {"b": "approve", "i1": None, "i2": "approve", "r1": "deny", "r2": "deny"}
    assert _class_errors(_CLASS, v) is None


def test_consult_partiality_zeroes_a_flagged_flip() -> None:
    # an irrelevant-set flip is flagged 'partial' by assess_justice -> consult enforces 0.
    v = {"b": "approve", "i1": "deny", "i2": "approve", "r1": "deny", "r2": "deny"}
    assert _consult_partiality(_CLASS, v, baseline_part=1) == 0


def test_consult_partiality_keeps_a_consistent_class() -> None:
    v = {"b": "approve", "i1": "approve", "i2": "approve", "r1": "deny", "r2": "deny"}
    assert _consult_partiality(_CLASS, v, baseline_part=0) == 0


def test_real_arm_fails_closed_without_labelled_battery(tmp_path, monkeypatch) -> None:
    import tools.run_dikaiosyne_eval as ev
    monkeypatch.setattr(ev, "LABELED_EXTERNAL", tmp_path / "nope.json")
    with pytest.raises(SystemExit):
        ev._load_labeled_external()
