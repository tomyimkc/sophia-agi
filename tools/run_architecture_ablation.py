#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Dry-run validator for Sophia long-context architecture ablation scaffolds.

This command summarizes planned ablations and proof artifacts only. It never
launches training, model inference, GPU work, or RunPod jobs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sophia-long-context-candidate.sample.json"
DEFAULT_BETS = ROOT / "agi-proof" / "architecture-bets" / "manifest.json"
DEFAULT_RUN_TEMPLATE = ROOT / "agi-proof" / "mlops" / "architecture-run-template.json"

REQUIRED_STATES = {"accepted", "rejected", "held", "abstain"}
REQUIRED_CHANNELS = {
    "task",
    "system_contract",
    "source_evidence",
    "working_memory",
    "episodic_memory",
    "semantic_memory",
    "tool_trace",
    "council_notes",
    "failure_context",
}
REQUIRED_ABLATION_ARMS = {
    "raw",
    "long_context_only",
    "long_context_plus_memory",
    "long_context_plus_tools",
    "verifier_gated",
    "sophia_full",
}
EXPECTED_BETS = {
    "verifier-gated-long-context",
    "hybrid-memory",
    "selective-tool-use-router",
    "council-orchestration",
    "verifier-as-reward",
    "long-context-compression-recall",
    "architecture-aware-eval-harness",
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.exists():
        return None, [f"{rel(path)}: missing"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, [f"{rel(path)}: invalid JSON: {exc}"]
    if not isinstance(data, dict):
        return None, [f"{rel(path)}: expected a JSON object"]
    return data, []


def require(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def require_candidate_boundary(errors: list[str], data: dict[str, Any], label: str) -> None:
    require(errors, data.get("candidateOnly") is True, f"{label}: candidateOnly must be true")
    require(errors, data.get("canClaimAGI") is False, f"{label}: canClaimAGI must be false")
    boundary = str(data.get("claimBoundary", ""))
    require(errors, "canClaimAGI=false" in boundary, f"{label}: claimBoundary must include canClaimAGI=false")


def _path_from_root(value: Any) -> Path:
    return ROOT / str(value)


def validate_candidate_config(path: Path = DEFAULT_CONFIG) -> tuple[list[str], dict[str, Any]]:
    data, errors = load_json(path)
    if data is None:
        return errors, {}

    label = rel(path)
    require(errors, data.get("schema") == "sophia.long_context_candidate.v1", f"{label}: schema mismatch")
    require(errors, data.get("status") == "scaffolding_only_not_run", f"{label}: status must be scaffolding_only_not_run")
    require(errors, data.get("dryRunDefault") is True, f"{label}: dryRunDefault must be true")
    require_candidate_boundary(errors, data, label)

    channels = set(data.get("contextChannels", []))
    require(errors, REQUIRED_CHANNELS <= channels, f"{label}: missing context channels")

    router = data.get("routerDecisions")
    if isinstance(router, dict):
        require(errors, REQUIRED_STATES <= set(router.get("verdictValues", [])), f"{label}: router missing verdict states")
        require(errors, router.get("failClosedOnMissingEvidence") is True, f"{label}: router must fail closed")
    else:
        errors.append(f"{label}: routerDecisions must be an object")

    memory_layers = data.get("memoryLayers", [])
    if isinstance(memory_layers, list):
        layer_names = {layer.get("layer") for layer in memory_layers if isinstance(layer, dict)}
        require(errors, {"scratch", "episodic", "semantic", "procedural", "negative"} <= layer_names, f"{label}: missing memory layers")
    else:
        errors.append(f"{label}: memoryLayers must be a list")

    eval_suites = data.get("evalSuites", [])
    if isinstance(eval_suites, list):
        require(errors, bool(eval_suites), f"{label}: evalSuites must be non-empty")
        for suite in eval_suites:
            if not isinstance(suite, dict):
                errors.append(f"{label}: eval suite entry must be an object")
                continue
            require(errors, suite.get("status") == "planned", f"{label}:{suite.get('id', '<missing-id>')}: status must be planned")
    else:
        errors.append(f"{label}: evalSuites must be a list")

    ablation_arms = set(data.get("ablationArms", []))
    require(errors, REQUIRED_ABLATION_ARMS <= ablation_arms, f"{label}: missing required ablation arms")

    thresholds = data.get("promotionThresholds")
    if isinstance(thresholds, dict):
        require(errors, thresholds.get("minimumSeeds", 0) >= 3, f"{label}: minimumSeeds must be >=3")
        require(errors, thresholds.get("requireFailureLedgerUpdate") is True, f"{label}: promotion must require ledger update")
        require(errors, thresholds.get("requireCanClaimAGIFalse") is True, f"{label}: promotion must preserve canClaimAGI=false")
    else:
        errors.append(f"{label}: promotionThresholds must be an object")

    artifacts = data.get("artifactPaths")
    if isinstance(artifacts, dict):
        for key in ("architectureDoc", "architectureBets", "contextPackingManifest", "mlopsRunTemplate", "failureLedger"):
            artifact_path = _path_from_root(artifacts.get(key, ""))
            require(errors, artifact_path.exists(), f"{label}: artifactPaths.{key} does not exist")
    else:
        errors.append(f"{label}: artifactPaths must be an object")

    summary = {
        "name": data.get("name"),
        "ablationArms": sorted(ablation_arms),
        "evalSuites": [suite.get("id") for suite in eval_suites if isinstance(suite, dict)],
        "contextWindowTarget": data.get("contextWindows", {}).get("targetTokens")
        if isinstance(data.get("contextWindows"), dict)
        else None,
    }
    return errors, summary


def validate_bets_manifest(path: Path = DEFAULT_BETS) -> tuple[list[str], dict[str, Any]]:
    data, errors = load_json(path)
    if data is None:
        return errors, {}

    label = rel(path)
    require(errors, data.get("kind") == "architecture-bet-manifest", f"{label}: kind mismatch")
    require(errors, data.get("status") == "scaffolding_only_not_run", f"{label}: status must be scaffolding_only_not_run")
    require_candidate_boundary(errors, data, label)
    require(errors, REQUIRED_STATES <= set(data.get("verdictValues", [])), f"{label}: missing verdict states")

    rules = data.get("promotionRules")
    if isinstance(rules, dict):
        require(errors, rules.get("minimumSeeds", 0) >= 3, f"{label}: promotion minimumSeeds must be >=3")
        require(errors, rules.get("requireAblations") is True, f"{label}: promotion must require ablations")
        require(errors, rules.get("requireFailureLedgerUpdate") is True, f"{label}: promotion must require ledger update")
        require(errors, rules.get("requireCanClaimAGIFalse") is True, f"{label}: promotion must preserve canClaimAGI=false")
    else:
        errors.append(f"{label}: promotionRules must be an object")

    bets = data.get("bets", [])
    require(errors, isinstance(bets, list) and bets, f"{label}: bets must be a non-empty list")
    bet_ids = {bet.get("betId") for bet in bets if isinstance(bet, dict)}
    require(errors, EXPECTED_BETS <= bet_ids, f"{label}: missing expected architecture bets")

    by_status: Counter[str] = Counter()
    for bet in bets if isinstance(bets, list) else []:
        if not isinstance(bet, dict):
            errors.append(f"{label}: bet entry must be an object")
            continue
        bet_label = f"{label}:{bet.get('betId', '<missing-id>')}"
        by_status[str(bet.get("status"))] += 1
        require(errors, bet.get("status") == "not_run", f"{bet_label}: scaffold bet status must be not_run")
        require(errors, bet.get("claimStatus") == "not_evidence", f"{bet_label}: claimStatus must be not_evidence")
        require(errors, len(bet.get("ablationArms", [])) >= 2, f"{bet_label}: expected at least two ablation arms")
        require(errors, bool(bet.get("metrics")), f"{bet_label}: metrics required")
        require(errors, bool(bet.get("verifierGates")), f"{bet_label}: verifierGates required")
        require(errors, bool(bet.get("failureLedgerRefs")), f"{bet_label}: failureLedgerRefs required")

        criteria = bet.get("promotionCriteria")
        if isinstance(criteria, dict):
            require(errors, criteria.get("minimumSeeds", 0) >= 3, f"{bet_label}: promotion minimumSeeds must be >=3")
            require(errors, criteria.get("mustBeatBaseline") is True, f"{bet_label}: mustBeatBaseline must be true")
            require(errors, criteria.get("requiresFailureLedgerUpdate") is True, f"{bet_label}: ledger update required")
        else:
            errors.append(f"{bet_label}: promotionCriteria must be an object")

    summary = {
        "bets": len(bets) if isinstance(bets, list) else 0,
        "betIds": sorted(bet_id for bet_id in bet_ids if isinstance(bet_id, str)),
        "statuses": dict(sorted(by_status.items())),
    }
    return errors, summary


def validate_run_template(path: Path = DEFAULT_RUN_TEMPLATE) -> tuple[list[str], dict[str, Any]]:
    data, errors = load_json(path)
    if data is None:
        return errors, {}

    label = rel(path)
    require(errors, data.get("schema") == "sophia.architecture_run_template.v1", f"{label}: schema mismatch")
    require(errors, data.get("kind") == "architecture-run-template", f"{label}: kind mismatch")
    require(errors, data.get("status") == "scaffolding_only_not_run", f"{label}: status must be scaffolding_only_not_run")
    require_candidate_boundary(errors, data, label)

    run_identity = data.get("runIdentity")
    if isinstance(run_identity, dict):
        require(errors, run_identity.get("dryRun") is True, f"{label}: runIdentity.dryRun must be true")
        require(errors, run_identity.get("runId") is None, f"{label}: runId must be null before execution")
    else:
        errors.append(f"{label}: runIdentity must be an object")

    for key in ("candidateConfig", "architectureBets"):
        artifact_path = _path_from_root(data.get(key, ""))
        require(errors, artifact_path.exists(), f"{label}: {key} does not exist")

    context_packing = data.get("contextPacking")
    if isinstance(context_packing, dict):
        for key in ("manifest", "evidenceCardSchema"):
            artifact_path = _path_from_root(context_packing.get(key, ""))
            require(errors, artifact_path.exists(), f"{label}: contextPacking.{key} does not exist")
    else:
        errors.append(f"{label}: contextPacking must be an object")

    evals = data.get("evals")
    if isinstance(evals, dict):
        require(errors, evals.get("minimumSeeds", 0) >= 3, f"{label}: evals.minimumSeeds must be >=3")
        require(errors, evals.get("resultArtifacts") == [], f"{label}: resultArtifacts must be empty before runs")
    else:
        errors.append(f"{label}: evals must be an object")

    ablations = data.get("ablations")
    if isinstance(ablations, dict):
        require(errors, REQUIRED_ABLATION_ARMS <= set(ablations.get("requiredArms", [])), f"{label}: missing required ablation arms")
        require(errors, ablations.get("completedArms") == [], f"{label}: completedArms must be empty before runs")
    else:
        errors.append(f"{label}: ablations must be an object")

    failure_ledger = data.get("failureLedger")
    if isinstance(failure_ledger, dict):
        require(errors, failure_ledger.get("updated") is False, f"{label}: failureLedger.updated must be false before runs")
        require(errors, _path_from_root(failure_ledger.get("path", "")).exists(), f"{label}: failureLedger.path does not exist")
    else:
        errors.append(f"{label}: failureLedger must be an object")

    verdict = data.get("promotionVerdict")
    if isinstance(verdict, dict):
        require(errors, verdict.get("verdict") == "held", f"{label}: initial promotion verdict must be held")
        require(errors, verdict.get("candidateOnly") is True, f"{label}: promotion candidateOnly must be true")
        require(errors, verdict.get("canClaimAGI") is False, f"{label}: promotion canClaimAGI must be false")
        require(errors, REQUIRED_STATES <= set(verdict.get("allowedValues", [])), f"{label}: promotion allowedValues missing states")
    else:
        errors.append(f"{label}: promotionVerdict must be an object")

    summary = {
        "templateVersion": data.get("templateVersion"),
        "plannedSuites": evals.get("plannedSuites", []) if isinstance(evals, dict) else [],
        "requiredArms": sorted(ablations.get("requiredArms", [])) if isinstance(ablations, dict) else [],
        "promotionVerdict": verdict.get("verdict") if isinstance(verdict, dict) else None,
    }
    return errors, summary


def validate(
    config_path: Path = DEFAULT_CONFIG,
    bets_path: Path = DEFAULT_BETS,
    run_template_path: Path = DEFAULT_RUN_TEMPLATE,
) -> tuple[int, dict[str, Any]]:
    errors: list[str] = []
    config_errors, config_summary = validate_candidate_config(config_path)
    bets_errors, bets_summary = validate_bets_manifest(bets_path)
    template_errors, template_summary = validate_run_template(run_template_path)
    errors.extend(config_errors)
    errors.extend(bets_errors)
    errors.extend(template_errors)

    result = {
        "ok": not errors,
        "errors": errors,
        "dryRun": True,
        "summary": {
            "candidate": config_summary,
            "architectureBets": bets_summary,
            "runTemplate": template_summary,
            "claimBoundary": "Dry-run validation only; no model capability is claimed and canClaimAGI=false.",
        },
    }
    return (0 if not errors else 1), result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Sophia architecture ablation scaffold without running training")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to long-context candidate config")
    parser.add_argument("--bets", default=str(DEFAULT_BETS), help="Path to architecture bet manifest")
    parser.add_argument("--run-template", default=str(DEFAULT_RUN_TEMPLATE), help="Path to MLOps architecture run template")
    parser.add_argument("--json", action="store_true", help="Print machine-readable validation output")
    args = parser.parse_args()

    code, result = validate(Path(args.config), Path(args.bets), Path(args.run_template))
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif result["errors"]:
        print("Architecture ablation dry-run validation failed:")
        for error in result["errors"]:
            print(f"- {error}")
    else:
        summary = result["summary"]
        print(
            "Architecture ablation dry-run validation OK "
            f"({summary['architectureBets']['bets']} bets, "
            f"{len(summary['candidate']['ablationArms'])} ablation arms)"
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
