#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Long-context architecture scaffold tests (offline only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import eval_context_packing as context_packing  # noqa: E402
from tools import run_architecture_ablation as architecture_ablation  # noqa: E402


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def test_context_packing_validator_passes() -> None:
    code, result = context_packing.validate()
    assert code == 0, result["errors"]
    assert result["summary"]["cards"] == 1
    assert result["summary"]["states"] == {"held": 1}


def test_architecture_ablation_dry_run_validator_passes() -> None:
    code, result = architecture_ablation.validate()
    assert code == 0, result["errors"]
    assert result["dryRun"] is True
    assert result["summary"]["architectureBets"]["bets"] == 7
    assert "sophia_full" in result["summary"]["candidate"]["ablationArms"]


def test_architecture_bets_are_scaffold_only() -> None:
    manifest = load("agi-proof/architecture-bets/manifest.json")
    assert manifest["candidateOnly"] is True
    assert manifest["canClaimAGI"] is False
    assert manifest["status"] == "scaffolding_only_not_run"
    assert set(manifest["verdictValues"]) >= {"accepted", "rejected", "held", "abstain"}

    bet_ids = {bet["betId"] for bet in manifest["bets"]}
    assert {
        "verifier-gated-long-context",
        "hybrid-memory",
        "selective-tool-use-router",
        "council-orchestration",
        "verifier-as-reward",
        "long-context-compression-recall",
        "architecture-aware-eval-harness",
    } <= bet_ids

    for bet in manifest["bets"]:
        assert bet["status"] == "not_run"
        assert bet["claimStatus"] == "not_evidence"
        assert bet["promotionCriteria"]["minimumSeeds"] >= 3
        assert bet["promotionCriteria"]["requiresFailureLedgerUpdate"] is True


def test_candidate_config_preserves_no_overclaim_boundary() -> None:
    config = load("configs/sophia-long-context-candidate.sample.json")
    assert config["candidateOnly"] is True
    assert config["canClaimAGI"] is False
    assert config["dryRunDefault"] is True
    assert "canClaimAGI=false" in config["claimBoundary"]
    assert set(config["routerDecisions"]["verdictValues"]) >= {"accepted", "rejected", "held", "abstain"}
    assert set(config["ablationArms"]) >= architecture_ablation.REQUIRED_ABLATION_ARMS
    assert config["promotionThresholds"]["minimumSeeds"] >= 3
    assert config["promotionThresholds"]["requireCanClaimAGIFalse"] is True


def test_mlops_architecture_run_template_is_not_a_result() -> None:
    template = load("agi-proof/mlops/architecture-run-template.json")
    assert template["candidateOnly"] is True
    assert template["canClaimAGI"] is False
    assert template["status"] == "scaffolding_only_not_run"
    assert template["runIdentity"]["dryRun"] is True
    assert template["runIdentity"]["runId"] is None
    assert template["evals"]["resultArtifacts"] == []
    assert template["ablations"]["completedArms"] == []
    assert template["promotionVerdict"]["verdict"] == "held"
    assert template["promotionVerdict"]["canClaimAGI"] is False


def test_context_packing_sample_card_is_illustrative() -> None:
    manifest = load("training/context_packing/manifest.json")
    assert manifest["candidateOnly"] is True
    assert manifest["canClaimAGI"] is False
    assert "canClaimAGI=false" in manifest["claimBoundary"]

    cards = manifest["sampleEvidenceCards"]
    assert len(cards) == 1
    card = cards[0]
    assert card["claimStatus"] == "illustrative"
    assert card["status"] == "held"
    assert card["verifier"]["state"] == "held"
    assert card["trainingUse"]["eligible"] is False


def main() -> int:
    test_context_packing_validator_passes()
    test_architecture_ablation_dry_run_validator_passes()
    test_architecture_bets_are_scaffold_only()
    test_candidate_config_preserves_no_overclaim_boundary()
    test_mlops_architecture_run_template_is_not_a_result()
    test_context_packing_sample_card_is_illustrative()
    print("test_architecture_scaffolding: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
