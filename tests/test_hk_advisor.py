# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for HK bilingual advisor benchmark seal (Phase 0) and verifier (Phase 1+)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_seal_verifies():
    from provenance_bench.hk_advisor_benchmark import verify_manifest
    result = verify_manifest(root=ROOT)
    assert result["ok"], result
    assert result["nCases"] == 90
    assert result["balance"]["answerable"] == 30
    assert result["balance"]["abstain"] == 30
    assert result["balance"]["traps"] == 30
    assert result["balance"]["trap_fabrication_bait"] == 10
    assert result["balance"]["trap_fake_citation"] == 10
    assert result["balance"]["trap_unanswerable"] == 10
    assert result["bilingualSplit"]["yue"] == 45
    assert result["bilingualSplit"]["en"] == 45


def test_benchmark_manifest_flags():
    manifest = json.loads(
        (ROOT / "data" / "hk_advisor_benchmark" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["candidateOnly"] is True
    assert manifest["canClaimAGI"] is False
    assert manifest["sealed"] is True
    assert manifest["trainingDisjoint"] is True


def test_hk_advisor_prompt_set_loads():
    from provenance_bench.dataset_guard import hk_advisor_benchmark_prompt_set
    forbidden = hk_advisor_benchmark_prompt_set(root=ROOT)
    assert len(forbidden) == 90
