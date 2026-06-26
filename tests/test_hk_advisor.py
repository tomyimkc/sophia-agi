# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for HK bilingual advisor benchmark seal and verifier."""
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
    assert result["bilingualSplit"]["yue"] == 45
    assert result["bilingualSplit"]["en"] == 45


def test_benchmark_manifest_flags():
    manifest = json.loads(
        (ROOT / "data" / "hk_advisor_benchmark" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["candidateOnly"] is True
    assert manifest["canClaimAGI"] is False
    assert manifest["sealed"] is True


def test_hk_advisor_prompt_set_loads():
    from provenance_bench.dataset_guard import hk_advisor_benchmark_prompt_set
    assert len(hk_advisor_benchmark_prompt_set(root=ROOT)) == 90


def test_verify_advisory_boundary():
    from agent.hk_advisor.verifier import verify_advisory_boundary
    good = (
        "Under Cap. 486 s.33 is not in force. "
        "This is not legal or financial advice. Verify with a licensed Hong Kong professional."
    )
    assert verify_advisory_boundary(good).passed
    assert not verify_advisory_boundary("Cap. 486 applies.").passed


def test_verify_abstention_on_trap():
    from agent.hk_advisor.verifier import verify_abstention
    case = {"label": {"decision": "abstain"}, "trap": "fabrication_bait", "epistemicLabel": "abstain"}
    assert verify_abstention(
        "I cannot confirm Cap. 712 exists; this provision is unverified.", case
    ).passed


def test_verify_citation_answerable():
    from agent.hk_advisor.verifier import verify_citation
    case = {"label": {"decision": "answerable", "gold_citations": ["Cap. 486"]}, "trap": "none"}
    ans = "Section 33 of the PDPO (Cap. 486) is not yet in force."
    assert verify_citation(ans, case).passed


def test_verify_trace_full():
    from agent.hk_advisor.verifier import trace_passes, verify_trace
    case = {
        "language": "en",
        "label": {"decision": "answerable", "gold_citations": ["Cap. 486"], "mustInclude": ["PDPO"]},
        "trap": "none",
        "epistemicLabel": "definite",
        "scoring": {"mustInclude": ["PDPO"]},
    }
    answer = (
        "Section 33 of the PDPO (Cap. 486) is not yet in force. "
        "This is not legal or financial advice. Verify with a licensed Hong Kong professional."
    )
    assert trace_passes(verify_trace(answer=answer, case=case))


def test_sft_traces_disjoint_from_benchmark():
    from provenance_bench.dataset_guard import hk_advisor_benchmark_prompt_set, normalize, prompt_of
    forbidden = hk_advisor_benchmark_prompt_set(root=ROOT)
    traces = ROOT / "training" / "hk_advisor" / "sft_traces.jsonl"
    assert traces.exists()
    for line in traces.read_text(encoding="utf-8").splitlines():
        if line.strip():
            pr = prompt_of(json.loads(line))
            assert pr and normalize(pr) not in forbidden


def test_eval_mock_report():
    from tools.eval_hk_advisor import run_eval
    r = run_eval(seeds=[0, 1, 2], mode="mock")
    assert r["nCases"] == 90 and r["canClaimAGI"] is False
    assert r["deltas"]["adapter_vs_base_fabrication"]["adapterLowersFabrication"]
