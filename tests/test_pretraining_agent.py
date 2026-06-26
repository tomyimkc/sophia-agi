#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the pretraining-researcher reviewer agent.

Asserts the agent is honest and fail-closed: it never claims AGI, it audits gates rather
than rubber-stamping, a missing report is cannot_assess (never pass), and it surfaces the
real critiques (high dup rate, low eval coverage). Offline, deterministic, dependency-free.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pretraining.agent import researcher  # noqa: E402
from pretraining.agent.researcher import review_all, role_card  # noqa: E402


def test_never_claims_agi() -> None:
    review = review_all()
    assert review["canClaimAGI"] is False
    assert role_card()["canClaimAGI"] is False
    # the persona is explicitly a critic, not an AGI agent
    assert "reviewer" in review["agent"]


def test_audits_all_studies_with_valid_verdicts() -> None:
    review = review_all()
    assert set(review["studies"]) == set(researcher._AUDITS)
    for name, s in review["studies"].items():
        assert s["verdict"] in {"pass", "concern", "cannot_assess"}
        assert "next_experiments" in s and isinstance(s["next_experiments"], list)
        assert s["track"] in {"algorithm", "data"}


def test_fail_closed_on_missing_report(monkeypatch=None) -> None:
    # Force every report to be "missing" -> all cannot_assess, never pass.
    import pretraining.agent.researcher as r
    original = r._load
    r._load = lambda rel: None
    try:
        review = review_all()
        assert review["tally"]["pass"] == 0
        assert review["tally"]["cannot_assess"] == len(r._AUDITS)
        assert review["overall"] == "open-items"
    finally:
        r._load = original


def test_surfaces_real_critiques_when_reports_present() -> None:
    # When the committed artifacts exist, the agent must flag the known issues, not hide them.
    review = review_all()
    if review["studies"]["data_passport"]["verdict"] == "cannot_assess":
        return  # artifacts not generated in this checkout; nothing to assert
    passport_crit = " ".join(review["studies"]["data_passport"]["critiques"]).lower()
    assert "duplicate" in passport_crit or "unlicensed" in passport_crit
    # eval matrix must report the multimodal gap among uncovered cells
    evm = review["studies"]["eval_matrix"]
    assert "multimodal_uncovered=True" in evm["evidence"] or evm["verdict"] == "cannot_assess"


def test_rubric_covers_both_tracks() -> None:
    review = review_all()
    cov = review["rubric_coverage"]
    assert "algorithm" in cov and "data" in cov
    # at least one dimension in each track should be backed by a study
    for track in ("algorithm", "data"):
        statuses = [d["status"] for d in cov[track].values()]
        assert any(st in {"strong", "partial"} for st in statuses)


def test_llm_critique_is_optional_and_degrades() -> None:
    # Default review has no llm_critique; requesting it must not raise even offline.
    assert "llm_critique" not in review_all()
    review = review_all(llm=True)
    assert "llm_critique" in review
    assert "available" in review["llm_critique"]


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
