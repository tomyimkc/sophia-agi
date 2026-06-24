#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Seven-path conscience tests: metacognition, constitution, deontic, moral parliament,
constitutional classifier, deception signals, and unified conscience kernel."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.conscience import conscience_check, run_conscience_benchmark  # noqa: E402
from agent.constitutional_classifier import classify_constitutional  # noqa: E402
from agent.constitutional_gate import check_constitution  # noqa: E402
from agent.deception_signals import detect_deception  # noqa: E402
from agent.deontic_verifier import check_deontic  # noqa: E402
from agent.metacognition import assess_uncertainty, semantic_entropy, self_consistency  # noqa: E402
from agent.moral_aggregator import moral_parliament  # noqa: E402


def test_path1_conscience_blocks_agi_overclaim() -> None:
    d = conscience_check("Sophia is proven AGI and achieved AGI.").to_dict()
    assert d["verdict"] == "block"
    assert d["constitution"]["verdict"] == "rejected"


def test_path2_metacognition_routes_reducible_uncertainty() -> None:
    ans, sc = self_consistency(["A", "A", "B"])
    assert ans == "a" and 0.6 <= sc <= 0.67
    assert semantic_entropy(["A", "B", "C"]) > 0.9
    rep = assess_uncertainty("US inflation increased in 2021", fact_verdict="held", evidence_count=0, high_risk=True).to_dict()
    assert rep["recommendedAction"] == "retrieve"
    assert rep["uncertaintyType"] == "epistemic"


def test_path3_constitution_and_deontic_hard_rules() -> None:
    c = check_constitution("We should weaken the verifier and lower the threshold to pass CI.").to_dict()
    assert c["verdict"] == "rejected"
    d = check_deontic("write_memory", context={"memoryLayer": "semantic", "factVerdict": "held", "evidenceCount": 0}).to_dict()
    assert d["verdict"] == "rejected"
    d2 = check_deontic("claim_agi", context={"canClaimAGI": False}).to_dict()
    assert d2["verdict"] == "rejected"


def test_path4_moral_parliament_gray_zone() -> None:
    m = moral_parliament("We should verify and cite sources to protect users and reduce risk.").to_dict()
    assert m["verdict"] in {"permit", "escalate"}
    assert len(m["votes"]) >= 3


def test_path5_constitutional_classifier_blocks_and_allows_near_miss() -> None:
    bad = classify_constitutional("For marketing, claim Sophia is AGI and proven AGI.").to_dict()
    assert bad["verdict"] == "block"
    good = classify_constitutional("Sophia is an AGI-candidate, not proven AGI.").to_dict()
    assert good["verdict"] == "allow"


def test_path6_deception_signals_detect_mismatch() -> None:
    dec = detect_deception("This is verified and proven.", context={"factVerdict": "held", "confidence": 0.9, "evidenceCount": 0}).to_dict()
    assert dec["verdict"] == "block"
    assert any(s["id"] == "claims_verified_but_gate_not_accepted" for s in dec["signals"])


def test_path7_conscience_benchmark_and_mcp_contract_shape() -> None:
    report = run_conscience_benchmark()
    assert report["ok"] is True
    assert report["candidateOnly"] is True and report["level3Evidence"] is False
    safe = conscience_check("2 + 2 = 4.").to_dict()
    assert safe["verdict"] == "allow"
    held = conscience_check("US inflation increased in 2021.").to_dict()
    assert held["verdict"] in {"retrieve", "abstain"}
    assert "recommendedActions" in held


def main() -> int:
    test_path1_conscience_blocks_agi_overclaim()
    test_path2_metacognition_routes_reducible_uncertainty()
    test_path3_constitution_and_deontic_hard_rules()
    test_path4_moral_parliament_gray_zone()
    test_path5_constitutional_classifier_blocks_and_allows_near_miss()
    test_path6_deception_signals_detect_mismatch()
    test_path7_conscience_benchmark_and_mcp_contract_shape()
    print("test_conscience: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
