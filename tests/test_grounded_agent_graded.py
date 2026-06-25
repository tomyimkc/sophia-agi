#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the graded answer/hedge/abstain overlay on agent.grounded_agent.

Covers the previously-unwired calibration router now live in the grounded path:
a low-confidence gate pass is downgraded to abstain; a mid-confidence pass is hedged
(answer surfaced, flagged, original kept); a high-confidence pass is untouched; an
already-abstaining policy is never upgraded (fail-closed); and graded=True with no
confidence signal is a guaranteed no-op (zero drift). LLM + gate mocked; offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.continual_qa_answer import ABSTAIN_TEXT  # noqa: E402
from agent.continual_qa_hybrid import STRICT  # noqa: E402
from agent.grounded_agent import apply_graded_decision, grounded_answer  # noqa: E402
from okf.page import Page  # noqa: E402

_ALLOW = lambda q, a: True  # noqa: E731

_RICH = ("# analects (論語)\n\nThe Analects is a compiled record of conversations attributed to "
         "Confucius and his disciples.\n")


def _pages():
    return [Page(path=Path("analects.md"),
                 meta={"id": "analects", "pageType": "text",
                       "canonicalTitleEn": "analects", "domain": "philosophy"},
                 body=_RICH)]


def _strict_answer(complete):
    return grounded_answer("What are the analects?", complete, pages=_pages(),
                           attribution_check=_ALLOW)


# --------------------------------------------------------------------------- #
# apply_graded_decision unit behavior (downgrade-only, fail-closed)
# --------------------------------------------------------------------------- #

def test_low_confidence_pass_is_downgraded_to_abstain() -> None:
    out = {"answer": "grounded answer", "policy": STRICT, "gated": False}
    apply_graded_decision(out, confidence=0.30)  # below lo=0.4
    assert out["policy"] == "graded_abstain_low_confidence"
    assert out["answer"] == ABSTAIN_TEXT
    assert out["rawAnswer"] == "grounded answer"  # original preserved for audit
    assert out["graded"]["applied"] and out["graded"]["action"] == "abstain"


def test_mid_confidence_pass_is_hedged() -> None:
    out = {"answer": "grounded answer", "policy": STRICT, "gated": False}
    apply_graded_decision(out, confidence=0.55)  # in [lo, hi)
    assert out["policy"] == "grounded_strict_hedged"
    assert out["answer"].startswith("(low confidence) ")
    assert out["rawAnswer"] == "grounded answer"
    assert out["graded"]["action"] == "hedge"


def test_high_confidence_pass_untouched() -> None:
    out = {"answer": "grounded answer", "policy": STRICT, "gated": False}
    apply_graded_decision(out, confidence=0.95)  # >= hi
    assert out["policy"] == STRICT
    assert out["answer"] == "grounded answer"
    assert "rawAnswer" not in out
    assert out["graded"]["action"] == "answer"


def test_abstain_is_never_upgraded_even_at_high_confidence() -> None:
    # A high-confidence "near miss" would let the router *hedge* — but we must never
    # resurrect an answer a fail-closed policy already suppressed.
    out = {"answer": ABSTAIN_TEXT, "policy": "fallback_gated_abstain", "gated": True}
    apply_graded_decision(out, confidence=0.99)
    assert out["policy"] == "fallback_gated_abstain"
    assert out["answer"] == ABSTAIN_TEXT
    assert "rawAnswer" not in out


def test_no_confidence_signal_is_a_noop() -> None:
    out = {"answer": "grounded answer", "policy": STRICT, "gated": False}
    apply_graded_decision(out)  # no confidence / evidence / samples
    assert out["policy"] == STRICT
    assert out["answer"] == "grounded answer"
    assert out["graded"]["applied"] is False


def test_thresholds_are_honored() -> None:
    out = {"answer": "x", "policy": STRICT, "gated": False}
    # Raise the floor above 0.6 so 0.55 now falls below lo -> abstain.
    apply_graded_decision(out, confidence=0.55, thresholds={"hi": 0.8, "lo": 0.6})
    assert out["policy"] == "graded_abstain_low_confidence"


def test_self_consistency_samples_drive_confidence() -> None:
    out = {"answer": "Paris", "policy": STRICT, "gated": False}
    # 3/4 agree -> confidence 0.75 >= hi -> answer untouched.
    apply_graded_decision(out, self_consistency_samples=["Paris", "Paris", "Paris", "Lyon"])
    assert out["graded"]["action"] == "answer" and out["policy"] == STRICT
    out2 = {"answer": "Paris", "policy": STRICT, "gated": False}
    # 2/5 agree -> 0.4 in [lo, hi) -> hedge.
    apply_graded_decision(out2, self_consistency_samples=["Paris", "Paris", "A", "B", "C"])
    assert out2["graded"]["action"] == "hedge"


# --------------------------------------------------------------------------- #
# end-to-end through grounded_answer
# --------------------------------------------------------------------------- #

def test_grounded_answer_graded_off_is_unchanged() -> None:
    out = _strict_answer(lambda s, u: "grounded answer")
    assert out["policy"] == STRICT and "graded" not in out


def test_grounded_answer_graded_low_confidence_abstains() -> None:
    out = grounded_answer("What are the analects?", lambda s, u: "grounded answer",
                          pages=_pages(), attribution_check=_ALLOW,
                          graded=True, confidence=0.1)
    assert out["policy"] == "graded_abstain_low_confidence"
    assert out["answer"] == ABSTAIN_TEXT and out["rawAnswer"] == "grounded answer"


def test_grounded_answer_graded_high_confidence_passes() -> None:
    out = grounded_answer("What are the analects?", lambda s, u: "grounded answer",
                          pages=_pages(), attribution_check=_ALLOW,
                          graded=True, confidence=0.9)
    assert out["policy"] == STRICT and out["answer"] == "grounded answer"
    assert out["graded"]["applied"] and out["graded"]["action"] == "answer"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
