#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Validate proof scaffolding manifests without running benchmarks.

This is intentionally dependency-light. JSON Schema files document the artifact
contracts, while this tool enforces the no-overclaim invariants that matter for
candidate-only benchmark and MLOps declarations.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_MANIFEST = ROOT / "agi-proof" / "external-benchmarks" / "manifest.json"
HIDDEN_WORKFLOW = ROOT / "agi-proof" / "hidden-reviewer-packs" / "third-party-workflow.manifest.json"
CHECKPOINT_REGISTRY = ROOT / "agi-proof" / "mlops" / "checkpoint-registry.json"
EXPERIMENT_SPEC = ROOT / "agi-proof" / "mlops" / "experiment-tracking-spec.json"
NEW_DOCS = [
    ROOT / "agi-proof" / "external-benchmarks" / "README.md",
    ROOT / "agi-proof" / "hidden-reviewer-packs" / "README.md",
    ROOT / "agi-proof" / "mlops" / "README.md",
    ROOT / "agi-proof" / "mlops" / "adapter-card-template.md",
    ROOT / "agi-proof" / "mlops" / "replication-runbook.md",
]

REQUIRED_EXTERNAL_FAMILIES = {
    "arc_agi",
    "gaia",
    "swe_bench",
    "metr_autonomy",
    "third_party_hidden_pack",
}
FORBIDDEN_UNQUALIFIED_PATTERNS = [
    re.compile(r"\bproven\s+agi\b", re.IGNORECASE),
    re.compile(r"\bis\s+agi\b", re.IGNORECASE),
    re.compile(r"\bguarantees?\s+(safety|truth|no hallucination)\b", re.IGNORECASE),
]


def load_json(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.exists():
        return None, [f"{path.relative_to(ROOT)}: missing"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, [f"{path.relative_to(ROOT)}: invalid JSON: {exc}"]
    if not isinstance(data, dict):
        return None, [f"{path.relative_to(ROOT)}: expected a JSON object"]
    return data, []


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def require(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def require_candidate_boundary(errors: list[str], data: dict[str, Any], path: Path) -> None:
    require(errors, data.get("candidateOnly") is True, f"{rel(path)}: candidateOnly must be true")
    require(errors, data.get("canClaimAGI") is False, f"{rel(path)}: canClaimAGI must be false")
    boundary = str(data.get("claimBoundary", "")).lower()
    require(errors, "not" in boundary or "only" in boundary, f"{rel(path)}: claimBoundary must be explicitly bounded")


def validate_external_manifest(path: Path = EXTERNAL_MANIFEST) -> list[str]:
    data, errors = load_json(path)
    if data is None:
        return errors

    require_candidate_boundary(errors, data, path)
    require(errors, data.get("kind") == "external-benchmark-workflow-manifest", f"{rel(path)}: wrong kind")
    require(errors, data.get("status") == "scaffolding_only_not_run", f"{rel(path)}: status must remain scaffolding_only_not_run")
    ledger_refs = data.get("failureLedgerRefs", [])
    require(errors, "external-benchmarks-not-run" in ledger_refs, f"{rel(path)}: missing external benchmark ledger ref")

    families = data.get("benchmarkFamilies", [])
    require(errors, isinstance(families, list) and families, f"{rel(path)}: benchmarkFamilies must be non-empty")
    seen = {item.get("benchmarkFamily") for item in families if isinstance(item, dict)}
    require(errors, REQUIRED_EXTERNAL_FAMILIES <= seen, f"{rel(path)}: missing benchmark family declarations")

    for item in families:
        if not isinstance(item, dict):
            errors.append(f"{rel(path)}: benchmark family entry must be an object")
            continue
        prefix = f"{rel(path)}:{item.get('id', '<missing-id>')}"
        require(errors, item.get("status") == "not_run", f"{prefix}: scaffolding must not declare a run")
        require(errors, item.get("claimStatus") == "not_evidence", f"{prefix}: claimStatus must be not_evidence")
        result = item.get("resultDeclaration", {})
        require(errors, isinstance(result, dict), f"{prefix}: resultDeclaration must be an object")
        require(errors, result.get("canClaimAGI") is False, f"{prefix}: result canClaimAGI must be false")
        require(errors, result.get("score") is None, f"{prefix}: score must be null until run")
        require(errors, result.get("total") is None, f"{prefix}: total must be null until run")
        require(errors, result.get("metric") is None, f"{prefix}: metric must be null until run")
        require(errors, result.get("ci95") is None, f"{prefix}: ci95 must be null until run")
        require(errors, result.get("seeds") == [], f"{prefix}: seeds must be empty until run")

    contract = data.get("artifactContract", {})
    rules = contract.get("resultPromotionRules", {}) if isinstance(contract, dict) else {}
    require(errors, rules.get("minimumSeedsForHeadline", 0) >= 3, f"{rel(path)}: headline rule must require >=3 seeds")
    require(errors, rules.get("requireFailureLedgerUpdate") is True, f"{rel(path)}: promotion must require ledger update")
    require(errors, rules.get("requireCanClaimAGIFalse") is True, f"{rel(path)}: promotion must preserve canClaimAGI false")
    return errors


def validate_hidden_workflow(path: Path = HIDDEN_WORKFLOW) -> list[str]:
    data, errors = load_json(path)
    if data is None:
        return errors

    require_candidate_boundary(errors, data, path)
    require(errors, data.get("kind") == "third-party-hidden-pack-workflow", f"{rel(path)}: wrong kind")
    require(errors, data.get("status") == "scaffolding_only_not_run", f"{rel(path)}: status must remain scaffolding_only_not_run")
    ledger_refs = data.get("failureLedgerRefs", [])
    require(errors, "hidden-review-third-party-not-run" in ledger_refs, f"{rel(path)}: missing hidden third-party ledger ref")

    stages = data.get("stages", [])
    require(errors, isinstance(stages, list) and len(stages) >= 5, f"{rel(path)}: expected staged workflow")
    stage_ids = {stage.get("id") for stage in stages if isinstance(stage, dict)}
    require(errors, "publish-commitments" in stage_ids, f"{rel(path)}: missing commitments stage")
    require(errors, "publish-aggregate" in stage_ids, f"{rel(path)}: missing aggregate publication stage")

    template = data.get("artifactManifestTemplate", {})
    require(errors, isinstance(template, dict), f"{rel(path)}: artifactManifestTemplate must be an object")
    require(errors, template.get("canClaimAGI") is False, f"{rel(path)}: template canClaimAGI must be false")
    require(errors, template.get("resultArtifact") is None, f"{rel(path)}: resultArtifact must be null in scaffold")
    return errors


def validate_checkpoint_registry(path: Path = CHECKPOINT_REGISTRY) -> list[str]:
    data, errors = load_json(path)
    if data is None:
        return errors

    require_candidate_boundary(errors, data, path)
    require(errors, data.get("kind") == "checkpoint-registry", f"{rel(path)}: wrong kind")
    policy = data.get("retentionPolicy", {})
    if isinstance(policy, dict):
        require(errors, policy.get("keepWeightsOutOfGit") is True, f"{rel(path)}: weights must stay out of git")
        require(errors, policy.get("requireSha256") is True, f"{rel(path)}: checksums must be required")
        require(errors, policy.get("requireFailureLedgerLink") is True, f"{rel(path)}: ledger link must be required")
    else:
        errors.append(f"{rel(path)}: retentionPolicy must be an object")

    checkpoints = data.get("checkpoints", [])
    require(errors, isinstance(checkpoints, list), f"{rel(path)}: checkpoints must be a list")
    for checkpoint in checkpoints if isinstance(checkpoints, list) else []:
        if not isinstance(checkpoint, dict):
            errors.append(f"{rel(path)}: checkpoint entry must be an object")
            continue
        prefix = f"{rel(path)}:{checkpoint.get('checkpointId', '<missing-id>')}"
        require(errors, checkpoint.get("candidateOnly") is True, f"{prefix}: candidateOnly must be true")
        require(errors, checkpoint.get("canClaimAGI") is False, f"{prefix}: canClaimAGI must be false")
        promotion = checkpoint.get("promotion", {})
        verdict = promotion.get("verdict") if isinstance(promotion, dict) else None
        require(errors, verdict in {"not_evaluated", "reject", "promote_internal"}, f"{prefix}: invalid promotion verdict")
    return errors


def validate_experiment_spec(path: Path = EXPERIMENT_SPEC) -> list[str]:
    data, errors = load_json(path)
    if data is None:
        return errors

    require_candidate_boundary(errors, data, path)
    require(errors, data.get("kind") == "experiment-tracking-aggregation-spec", f"{rel(path)}: wrong kind")
    aggregation = data.get("aggregation", {})
    if isinstance(aggregation, dict):
        require(errors, aggregation.get("minimumSeeds", 0) >= 3, f"{rel(path)}: aggregation must require >=3 seeds")
        require(errors, "failure" in str(aggregation.get("failureRule", "")).lower(), f"{rel(path)}: failureRule must preserve failures")
        require(errors, "two independent judge" in str(aggregation.get("judgeRule", "")).lower(), f"{rel(path)}: judgeRule must require independent judges")
    else:
        errors.append(f"{rel(path)}: aggregation must be an object")

    run_record = data.get("runRecord", {})
    required = run_record.get("requiredFields", []) if isinstance(run_record, dict) else []
    require(errors, "failureLedgerRef" in required, f"{rel(path)}: run records must include failureLedgerRef")
    require(errors, "canClaimAGI" in required, f"{rel(path)}: run records must include canClaimAGI")
    return errors


def validate_new_docs(paths: list[Path] = NEW_DOCS) -> list[str]:
    errors: list[str] = []
    for path in paths:
        if not path.exists():
            errors.append(f"{rel(path)}: missing")
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_UNQUALIFIED_PATTERNS:
            for match in pattern.finditer(text):
                line_no = text[: match.start()].count("\n") + 1
                errors.append(f"{rel(path)}:{line_no}: forbidden unqualified claim: {match.group(0)!r}")
    return errors


def validate_all() -> list[str]:
    errors: list[str] = []
    errors.extend(validate_external_manifest())
    errors.extend(validate_hidden_workflow())
    errors.extend(validate_checkpoint_registry())
    errors.extend(validate_experiment_spec())
    errors.extend(validate_new_docs())
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate candidate-only proof scaffolding artifacts")
    parser.add_argument("--json", action="store_true", help="Print machine-readable validation output")
    args = parser.parse_args()

    errors = validate_all()
    if args.json:
        print(json.dumps({"ok": not errors, "errors": errors}, indent=2))
    elif errors:
        print("Proof scaffolding validation failed:")
        for error in errors:
            print(f"- {error}")
    else:
        print("Proof scaffolding validation OK")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
