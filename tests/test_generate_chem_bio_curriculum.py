# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for tools/generate_chem_bio_curriculum.py output format and oracle gating."""
from __future__ import annotations

import json
from pathlib import Path

from tools.generate_chem_bio_curriculum import (
    OUT_DIR,
    generate_heldout,
    generate_problems,
    run,
    verify_and_build_rows,
    verify_problem,
)

ROOT = Path(__file__).resolve().parents[1]


def test_generate_problems_has_three_tiers() -> None:
    probs = generate_problems()
    assert set(probs) == {"tier0", "tier1", "tier2"}
    assert "chemistry" in probs["tier0"] and "biology" in probs["tier0"]
    assert "abstention" in probs["tier1"]


def test_every_problem_passes_its_oracle() -> None:
    # The defining contract: each generated row's gold is accepted by the oracle.
    for tier, buckets in generate_problems().items():
        for domain, probs in buckets.items():
            for prob in probs:
                v = verify_problem(prob)
                assert v["verdict"] == "accepted", (tier, domain, prob["id"], v)


def test_row_schema_and_metadata() -> None:
    rows, stats = verify_and_build_rows(generate_problems())
    assert stats["totals"]["kept"] > 50
    row = rows[0]
    assert len(row["messages"]) == 2
    assert row["messages"][0]["role"] == "user"
    meta = row["metadata"]
    assert meta["source"] == "sophia-chem-bio-curriculum"
    assert meta["verifierVerdict"] == "accepted"
    assert meta["trainingOracleOnly"] is True
    assert meta["domain"] in ("chemistry", "biology")


def test_check_run_is_clean_and_nonempty() -> None:
    code, result = run(check_only=True)
    assert code == 0, result
    assert result["contamination"]["clean"] is True
    assert result["selfOverlap"] == []
    assert result["rowCount"] > 50


def test_committed_pack_matches_generator() -> None:
    # Drift guard: the committed sft_all.jsonl must equal a fresh generation.
    rows, _ = verify_and_build_rows(generate_problems())
    committed = [json.loads(line) for line in
                 (OUT_DIR / "sft_all.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert committed == rows, "committed chem-bio pack drifted from the generator — regenerate"


def test_heldout_is_oracle_consistent() -> None:
    ho = generate_heldout()
    assert len(ho) >= 12
    for item in ho:
        assert item["externalSource"] is False
        assert item["goldAnswer"]
