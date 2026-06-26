# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the conformal on_fail route in agent/guarded.py (C1 wiring).

Same flow as the graded route but routes hedge-vs-abstain on a certified split-conformal
threshold. With no fitted policy artifact present the threshold falls back to
``1 - DEFAULT hi = 0.3`` (a safe no-op), which is what these offline tests exercise.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.guarded import ON_FAIL_MODES, guarded_complete  # noqa: E402

_FORBIDDEN = [
    "Confucius wrote the Dao De Jing.",
    "Plato wrote the Dao De Jing.",
    "Socrates wrote the Dao De Jing.",
]


class _Gen:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def __call__(self, system, user):
        text = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return SimpleNamespace(ok=True, text=text)


def _run(gen, **kw):
    return guarded_complete(
        "Who wrote the Dao De Jing?",
        on_fail="conformal",
        generate=gen,
        retrieve_fn=lambda q, top_k=8: [],
        format_context_fn=lambda chunks: "",
        **kw,
    )


def test_conformal_mode_registered():
    assert "conformal" in ON_FAIL_MODES


def test_high_confidence_near_miss_hedges():
    # Identical gate-failing answers -> self-consistency 1.0 -> nonconformity 0 <= tau.
    gen = _Gen([_FORBIDDEN[0]])
    res = _run(gen, samples=3)
    assert res.action == "hedged"
    assert res.ok is True and res.passed is False
    assert gen.calls == 3
    assert any("conformal confidence=1.0" in r for r in res.reasons)


def test_low_confidence_abstains():
    # Divergent answers -> low self-consistency -> nonconformity > tau -> abstain.
    gen = _Gen(_FORBIDDEN)
    res = _run(gen, samples=3)
    assert res.action == "abstained"
    assert res.ok is True
    assert any("-> abstain" in r for r in res.reasons)


def test_single_sample_is_fail_closed():
    # samples=1 -> neutral confidence 0.5 -> nonconformity 0.5 > 0.3 -> abstain.
    gen = _Gen([_FORBIDDEN[0]])
    res = _run(gen, samples=1)
    assert res.action == "abstained"
    assert gen.calls == 1


def test_clean_pass_unaffected():
    gen = _Gen(["Laozi is traditionally credited with the Dao De Jing; authorship is disputed."])
    res = _run(gen, samples=3)
    assert res.action == "clean" and res.passed is True
    assert gen.calls == 1


def _main():
    test_conformal_mode_registered()
    test_high_confidence_near_miss_hedges()
    test_low_confidence_abstains()
    test_single_sample_is_fail_closed()
    test_clean_pass_unaffected()
    print("test_guarded_conformal: OK")


if __name__ == "__main__":
    _main()
