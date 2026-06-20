#!/usr/bin/env python3
"""Tests for the best-of-N provenance reranker (agent/best_of.py). Offline.

Sample up to N candidates, rank by the source-discipline gate (a passing answer
always beats a violating one; ties break on fewer violations then a score), and
early-exit on the first gate-passing sample to save compute. The model is injected.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import best_of as bo  # noqa: E402
from agent.model import ModelResult  # noqa: E402

# One record, two forbidden authors -> lets us produce candidates with 1 vs 2 violations.
RECORDS = {"phoenix": {"canonicalTitleEn": "Project Phoenix Charter", "doNotAttributeTo": ["Alice", "Bob"]}}
CLEAN = "The Project Phoenix Charter was ratified by the founding committee."
CLEAN_LONG = "The Project Phoenix Charter was ratified by the founding committee after a long spring review."
ONE_VIOLATION = "Alice wrote the Project Phoenix Charter."
TWO_VIOLATIONS = "Alice wrote the Project Phoenix Charter. Bob wrote the Project Phoenix Charter."


def _gen(*responses, oks=None):
    box = {"i": 0}

    def generate(system: str, user: str) -> ModelResult:
        idx = min(box["i"], len(responses) - 1)
        ok = True if oks is None else oks[idx]
        box["i"] += 1
        return ModelResult(text=responses[idx], provider="mock", model="t", ok=ok)

    return generate


def _kw(**over):
    base = dict(
        records=RECORDS,
        retrieve_fn=lambda query, top_k=8: [],
        format_context_fn=lambda chunks: "(context)",
    )
    base.update(over)
    return base


def test_early_exit_on_first_passing() -> None:
    res = bo.best_of("q", n=4, generate=_gen(ONE_VIOLATION, CLEAN, CLEAN), early_exit=True, **_kw())
    assert res.passed is True
    assert res.text == CLEAN
    assert res.samples == 2          # stopped after the first passing sample
    assert res.chosen_index == 1


def test_none_pass_picks_fewest_violations() -> None:
    res = bo.best_of("q", n=2, generate=_gen(TWO_VIOLATIONS, ONE_VIOLATION), early_exit=True, **_kw())
    assert res.passed is False
    assert res.text == ONE_VIOLATION  # fewer violations wins
    assert res.samples == 2
    assert res.chosen_index == 1


def test_score_breaks_ties_among_passing() -> None:
    res = bo.best_of(
        "q", n=2, generate=_gen(CLEAN, CLEAN_LONG), early_exit=False,
        score_fn=lambda text: float(len(text)), **_kw(),
    )
    assert res.passed is True
    assert res.text == CLEAN_LONG     # higher score among passing
    assert res.samples == 2


def test_model_error_candidate_is_skipped() -> None:
    res = bo.best_of("q", n=4, generate=_gen("", CLEAN, oks=[False, True]), early_exit=True, **_kw())
    assert res.passed is True
    assert res.text == CLEAN
    assert res.samples == 2
    assert len(res.candidates) == 1   # the failed generation produced no candidate


def test_all_errors_returns_empty_unpassed() -> None:
    res = bo.best_of("q", n=2, generate=_gen("", "", oks=[False, False]), early_exit=True, **_kw())
    assert res.passed is False
    assert res.text == ""
    assert res.chosen_index == -1


def main() -> int:
    test_early_exit_on_first_passing()
    test_none_pass_picks_fewest_violations()
    test_score_breaks_ties_among_passing()
    test_model_error_candidate_is_skipped()
    test_all_errors_returns_empty_unpassed()
    print("test_best_of: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
