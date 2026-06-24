#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for fact-check learning candidate quarantine/recheck loop."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.fact_check_eval import load_jsonl, run_fact_check_eval  # noqa: E402
from agent.fact_check_flywheel import extract_learning_candidates, run_flywheel_from_report  # noqa: E402
from agent.live_sources import FixtureFactBackend  # noqa: E402

PACK = ROOT / "eval" / "fact_check" / "heldout_v1.jsonl"
FIXTURES = ROOT / "eval" / "fact_check" / "fixtures_v1.json"


def test_extract_and_recheck_learning_candidates() -> None:
    rows = load_jsonl(PACK)
    b = FixtureFactBackend.from_file(FIXTURES)
    report = run_fact_check_eval(rows, retriever=b.retriever, entailment=b.entailment,
                                 doi_resolver=b.doi_resolver, url_resolver=b.url_resolver)
    candidates = extract_learning_candidates(report)
    assert candidates
    assert all(c["promotionState"] == "pending_quarantine" for c in candidates)
    flywheel = run_flywheel_from_report(report, retriever=b.retriever, entailment=b.entailment,
                                        doi_resolver=b.doi_resolver, url_resolver=b.url_resolver)
    assert flywheel["canonicalWikiWrite"] is False
    assert flywheel["nPromotedProvisional"] > 0
    assert flywheel["nCandidates"] == len(candidates)


def main() -> int:
    test_extract_and_recheck_learning_candidates()
    print("test_fact_check_flywheel: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
