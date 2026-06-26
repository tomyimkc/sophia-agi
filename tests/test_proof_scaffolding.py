#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for benchmark and MLOps proof scaffolding artifacts."""

from __future__ import annotations

import json
import sys
from tempfile import TemporaryDirectory
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import validate_proof_scaffolding as v  # noqa: E402


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def test_validate_proof_scaffolding_passes() -> None:
    assert v.validate_all() == []


def test_external_benchmark_manifest_declares_no_results() -> None:
    manifest = load("agi-proof/external-benchmarks/manifest.json")
    assert manifest["candidateOnly"] is True
    assert manifest["canClaimAGI"] is False
    assert manifest["status"] == "scaffolding_only_not_run"

    families = {entry["benchmarkFamily"]: entry for entry in manifest["benchmarkFamilies"]}
    assert {
        "arc_agi",
        "gaia",
        "swe_bench",
        "metr_autonomy",
        "third_party_hidden_pack",
    } <= set(families)

    for entry in families.values():
        result = entry["resultDeclaration"]
        assert entry["status"] == "not_run"
        assert entry["claimStatus"] == "not_evidence"
        assert result["score"] is None
        assert result["total"] is None
        assert result["metric"] is None
        assert result["ci95"] is None
        assert result["seeds"] == []
        assert result["canClaimAGI"] is False


def test_hidden_third_party_workflow_is_scaffold_only() -> None:
    manifest = load("agi-proof/hidden-reviewer-packs/third-party-workflow.manifest.json")
    assert manifest["candidateOnly"] is True
    assert manifest["canClaimAGI"] is False
    assert manifest["status"] == "scaffolding_only_not_run"
    assert "hidden-review-third-party-not-run" in manifest["failureLedgerRefs"]

    stages = {stage["id"]: stage for stage in manifest["stages"]}
    assert "publish-commitments" in stages
    assert "publish-aggregate" in stages
    assert manifest["artifactManifestTemplate"]["resultArtifact"] is None
    assert manifest["artifactManifestTemplate"]["canClaimAGI"] is False


def test_mlops_registry_and_tracking_preserve_claim_boundary() -> None:
    registry = load("agi-proof/mlops/checkpoint-registry.json")
    assert registry["candidateOnly"] is True
    assert registry["canClaimAGI"] is False
    assert registry["checkpoints"] == []
    assert registry["retentionPolicy"]["keepWeightsOutOfGit"] is True
    assert registry["retentionPolicy"]["requireFailureLedgerLink"] is True

    spec = load("agi-proof/mlops/experiment-tracking-spec.json")
    assert spec["candidateOnly"] is True
    assert spec["canClaimAGI"] is False
    assert spec["aggregation"]["minimumSeeds"] >= 3
    assert "failureLedgerRef" in spec["runRecord"]["requiredFields"]
    assert "canClaimAGI" in spec["runRecord"]["requiredFields"]


def test_validator_catches_external_overclaim(tmp_path: Path) -> None:
    manifest = load("agi-proof/external-benchmarks/manifest.json")
    manifest["benchmarkFamilies"][0]["status"] = "completed_candidate"
    manifest["benchmarkFamilies"][0]["resultDeclaration"]["score"] = 0.8
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    errors = v.validate_external_manifest(path)
    assert any("scaffolding must not declare a run" in error for error in errors)
    assert any("score must be null until run" in error for error in errors)


def main() -> int:
    test_validate_proof_scaffolding_passes()
    test_external_benchmark_manifest_declares_no_results()
    test_hidden_third_party_workflow_is_scaffold_only()
    test_mlops_registry_and_tracking_preserve_claim_boundary()

    with TemporaryDirectory() as td:
        test_validator_catches_external_overclaim(Path(td))
    print("test_proof_scaffolding: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
