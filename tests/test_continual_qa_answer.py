#!/usr/bin/env python3
"""Tests for agent.continual_qa_answer — grounded answering + multi-judge scoring.

Verifies abstention without a source, robust judge-JSON parsing, the verdict rubric for
assert vs abstain expectations, and Cohen's κ. All model calls are mocked, so this is
offline, deterministic, dependency-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.continual_qa_answer import (  # noqa: E402
    ABSTAIN_TEXT, cohen_kappa, generate_grounded, judge_answer, verdict,
)


def test_grounded_abstains_without_source() -> None:
    called = []
    assert generate_grounded("q", None, lambda s, u: called.append(1) or "x") == ABSTAIN_TEXT
    assert called == []        # no model call when there is no source


def test_grounded_uses_source() -> None:
    out = generate_grounded("Who?", "id: x | body: a fact", lambda s, u: "grounded answer")
    assert out == "grounded answer"


def test_judge_parses_messy_json() -> None:
    raw = 'Sure: {"abstains": false, "answersQuestion": true, "faithful": true, "fabricatesAttribution": false} ok'
    r = judge_answer("q", "a", lambda s, u: raw)
    assert r["answersQuestion"] and r["faithful"] and not r["abstains"]


def test_verdict_rubric() -> None:
    good = {"abstains": False, "answersQuestion": True, "faithful": True, "fabricatesAttribution": False}
    fabricated = {"abstains": False, "answersQuestion": True, "faithful": True, "fabricatesAttribution": True}
    declined = {"abstains": True, "answersQuestion": False, "faithful": True, "fabricatesAttribution": False}
    assert verdict(good, "assert") is True
    assert verdict(fabricated, "assert") is False          # fabrication fails an assert
    assert verdict(declined, "abstain") is True            # correct refusal passes
    assert verdict(good, "abstain") is False               # answering when it should abstain fails


def test_cohen_kappa() -> None:
    assert cohen_kappa([True, True, False, False], [True, True, False, False]) == 1.0
    assert cohen_kappa([], []) == 0.0
    # perfect disagreement -> negative κ
    assert cohen_kappa([True, False], [False, True]) < 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
