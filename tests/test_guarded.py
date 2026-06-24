#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the guarded completion loop (agent/guarded.py). Offline.

guarded_complete wraps a small model: retrieve+format_context, generate, and on a
provenance violation either repair once → else cited-abstain (or hedge/passthrough).
The model is injected as a `generate` callable so we can drive the repair path
deterministically without a live model.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import guarded as g  # noqa: E402
from agent.model import ModelResult  # noqa: E402

# A user-supplied record so the test does not depend on the seeded corpus.
RECORDS = {"phoenix": {"canonicalTitleEn": "Project Phoenix Charter", "doNotAttributeTo": ["Alice"]}}
VIOLATING = "Alice wrote the Project Phoenix Charter in 2019."
CLEAN = "The Project Phoenix Charter was ratified in 2019 by the founding committee."


def _gen(*responses, ok=True, error=None):
    """A fake `generate` returning each response in turn (sticky on the last)."""
    box = {"i": 0, "calls": []}

    def generate(system: str, user: str) -> ModelResult:
        box["calls"].append((system, user))
        idx = min(box["i"], len(responses) - 1)
        box["i"] += 1
        return ModelResult(text=responses[idx], provider="mock", model="t", ok=ok, error=error)

    generate.box = box  # type: ignore[attr-defined]
    return generate


def _kw():
    """Common injection: trivial offline retrieval + the test records."""
    return dict(
        records=RECORDS,
        retrieve_fn=lambda query, top_k=8: [],
        format_context_fn=lambda chunks: "(context)",
    )


def test_clean_answer_passes_through() -> None:
    gen = _gen(CLEAN)
    res = g.guarded_complete("Who ratified the charter?", generate=gen, on_fail="repair", **_kw())
    assert res.passed is True and res.ok is True
    assert res.action == "clean"
    assert res.text == CLEAN
    assert res.attempts == 1


def test_repair_fixes_violation() -> None:
    gen = _gen(VIOLATING, CLEAN)  # first violates, repair returns clean
    res = g.guarded_complete("Who wrote it?", generate=gen, on_fail="repair", **_kw())
    assert res.action == "repaired"
    assert res.passed is True and res.ok is True
    assert res.text == CLEAN
    assert res.attempts == 2


def test_repair_exhausted_falls_back_to_cited_abstention() -> None:
    gen = _gen(VIOLATING, VIOLATING)  # repair still violates -> abstain
    res = g.guarded_complete("Who wrote it?", generate=gen, on_fail="repair", **_kw())
    assert res.action == "abstained"
    assert res.ok is True  # a safe abstention is an accepted outcome
    assert res.attempts == 2
    # the abstention itself MUST pass the gate (no forbidden attribution leaks)
    assert g.check_claim(res.text, records=RECORDS)["passed"] is True


def test_abstain_mode_skips_repair() -> None:
    gen = _gen(VIOLATING)
    res = g.guarded_complete("Who wrote it?", generate=gen, on_fail="abstain", **_kw())
    assert res.action == "abstained"
    assert res.attempts == 1  # no repair generation attempted
    assert g.check_claim(res.text, records=RECORDS)["passed"] is True


def test_hedge_mode_flags_but_keeps_text() -> None:
    gen = _gen(VIOLATING)
    res = g.guarded_complete("Who wrote it?", generate=gen, on_fail="hedge", **_kw())
    assert res.action == "hedged"
    assert res.passed is False
    assert VIOLATING in res.text  # original kept
    assert "Alice" in " ".join(res.violations)
    lowered = res.text.lower()
    assert "unverified" in lowered or "could not" in lowered  # a visible disclaimer


def test_passthrough_returns_unguarded_text() -> None:
    gen = _gen(VIOLATING)
    res = g.guarded_complete("Who wrote it?", generate=gen, on_fail="passthrough", **_kw())
    assert res.action == "passthrough"
    assert res.passed is False and res.ok is False
    assert res.text == VIOLATING
    assert res.attempts == 1


def test_model_error_surfaces() -> None:
    gen = _gen("", ok=False, error="no API key")
    res = g.guarded_complete("anything", generate=gen, on_fail="repair", **_kw())
    assert res.action == "model_error"
    assert res.ok is False
    assert "no API key" in (res.reasons[0] if res.reasons else "")


def test_on_fail_env_default_is_respected() -> None:
    os.environ["SOPHIA_ON_FAIL"] = "abstain"
    try:
        gen = _gen(VIOLATING)
        res = g.guarded_complete("Who wrote it?", generate=gen, **_kw())  # no on_fail arg
        assert res.action == "abstained"
        assert res.attempts == 1
    finally:
        os.environ.pop("SOPHIA_ON_FAIL", None)


def test_invalid_mode_raises() -> None:
    try:
        g.guarded_complete("q", generate=_gen(CLEAN), on_fail="bogus", **_kw())
        raise AssertionError("expected ValueError for invalid on_fail mode")
    except ValueError:
        pass


def test_check_claim_mode_free() -> None:
    bad = g.check_claim(VIOLATING, records=RECORDS)
    assert bad["passed"] is False
    assert bad["violations"]
    good = g.check_claim(CLEAN, records=RECORDS)
    assert good["passed"] is True
    # negation/correction passes
    corrected = g.check_claim("Alice did not write the Project Phoenix Charter.", records=RECORDS)
    assert corrected["passed"] is True


def main() -> int:
    test_clean_answer_passes_through()
    test_repair_fixes_violation()
    test_repair_exhausted_falls_back_to_cited_abstention()
    test_abstain_mode_skips_repair()
    test_hedge_mode_flags_but_keeps_text()
    test_passthrough_returns_unguarded_text()
    test_model_error_surfaces()
    test_on_fail_env_default_is_respected()
    test_invalid_mode_raises()
    test_check_claim_mode_free()
    print("test_guarded: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
