#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Build Sophia's AGI-candidate proof evidence manifest.

The manifest is intentionally conservative: it records current reproducible
evidence and open proof gaps without claiming Sophia is proven AGI.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AGI_PROOF_DIR = ROOT / "agi-proof"
OUTPUT = AGI_PROOF_DIR / "evidence-manifest.json"
DOMAINS = ("philosophy", "psychology", "history", "religion", "personality")


def load_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def benchmark_case_count(domain: str) -> int:
    data = load_json(ROOT / "tests" / f"benchmark-{domain}.json", {})
    cases = data.get("cases", [])
    return len(cases) if isinstance(cases, list) else 0


def leaderboard_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for domain in DOMAINS:
        board = load_json(ROOT / "benchmark" / "results" / f"leaderboard-{domain}.json", {})
        entries = board.get("entries", [])
        best = None
        if entries:
            best = max(entries, key=lambda item: float(item.get("score_pct", 0)))
        summary[domain] = {
            "cases": int(board.get("cases") or benchmark_case_count(domain)),
            "entries": entries,
            "best": best,
        }
    return summary


def latest_hidden_report() -> dict[str, Any] | None:
    reports = hidden_public_reports()
    if not reports:
        return None
    return max(reports, key=hidden_report_sort_key)


def hidden_report_sort_key(report: dict[str, Any]) -> tuple[datetime, str]:
    run_at = str(report.get("runAt", ""))
    try:
        parsed = datetime.fromisoformat(run_at)
    except ValueError:
        parsed = datetime.min
    return parsed, str(report.get("artifact", ""))


def public_artifact(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_hidden_report(path: Path) -> dict[str, Any] | None:
    report = load_json(path, None)
    if not isinstance(report, dict):
        return None
    enriched = dict(report)
    enriched["artifact"] = public_artifact(path)
    return enriched


def hidden_public_reports() -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in sorted((AGI_PROOF_DIR / "benchmark-results").glob("hidden-*.public-report.json")):
        report = load_hidden_report(path)
        if report:
            reports.append(report)
    return reports


def hidden_commitments() -> dict[str, Any]:
    prepared = AGI_PROOF_DIR / "hidden-reviewer-packs" / "prepared-pack-2026-06-19.commitments.json"
    fresh = AGI_PROOF_DIR / "hidden-reviewer-packs" / "fresh-reviewer-pack-2026-06-19.commitments.json"
    fresh_report_paths = {
        "grok": (
            AGI_PROOF_DIR
            / "benchmark-results"
            / "hidden-fresh-reviewer-pack-2026-06-19-sophia-grok.public-report.json"
        ),
        "deepseek": (
            AGI_PROOF_DIR
            / "benchmark-results"
            / "hidden-fresh-reviewer-pack-2026-06-19-sophia-deepseek.public-report.json"
        ),
    }
    fresh_results = {
        backend: report
        for backend, path in fresh_report_paths.items()
        if (report := load_hidden_report(path)) is not None
    }
    fresh_result = max(fresh_results.values(), key=hidden_report_sort_key) if fresh_results else None
    return {
        "spentPreparedPack": load_json(prepared, None),
        "freshCandidatePack": load_json(fresh, None),
        "freshCandidateRun": fresh_result,
        "freshCandidateRuns": fresh_results,
        "freshCandidateStatus": (
            "spent-on-2026-06-19; candidate pack was not independent third-party evidence; "
            "rerun requires a new unspent reviewer-controlled pack"
            if fresh_result
            else "sealed-unrun-candidate-third-party-pack; independent evidence requires external reviewer signature"
        ),
    }


def build_manifest(*, generated: str | None = None) -> dict[str, Any]:
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    models = load_json(ROOT / "models" / "manifest.json", {})
    rag_summary = load_json(ROOT / "benchmark" / "model_runs" / "rag-claude-summary.json", {})
    leaderboards = leaderboard_summary()
    benchmark_total = sum(item["cases"] for item in leaderboards.values())
    hidden_report = latest_hidden_report()
    commitments = hidden_commitments()

    return {
        "version": version,
        "generated": generated or datetime.now().isoformat(timespec="seconds"),
        "claimBoundary": (
            "Sophia is an AGI-candidate proof package and provenance-aware "
            "reasoning system. This repository does not prove true AGI."
        ),
        "operationalDefinition": {
            "summary": (
                "For this repo, AGI evidence means broad task competence, transfer, "
                "tool use, long-horizon autonomy, learning under distribution shift, "
                "baseline superiority, and independent reproduction."
            ),
            "sourceFamilies": [
                {
                    "name": "OpenAI Charter",
                    "emphasis": "highly autonomous systems that outperform humans at economically valuable work",
                    "url": "https://openai.com/charter/",
                },
                {
                    "name": "Google DeepMind Levels of AGI",
                    "emphasis": "generality, performance, and autonomy levels",
                    "url": "https://arxiv.org/abs/2311.02462",
                },
                {
                    "name": "Legg and Hutter universal intelligence",
                    "emphasis": "achieving goals across a wide range of environments",
                    "url": "https://arxiv.org/abs/0712.3329",
                },
                {
                    "name": "Chollet ARC / measure of intelligence",
                    "emphasis": "skill-acquisition efficiency on novel tasks",
                    "url": "https://arxiv.org/abs/1911.01547",
                },
            ],
        },
        "currentEvidence": {
            "trainingExamples": len(list((ROOT / "training" / "examples").glob("*.json"))),
            "domains": list(DOMAINS),
            "benchmarkCases": benchmark_total,
            "leaderboards": leaderboards,
            "localModel": {
                "name": models.get("benchmarkScore", {}).get("model", "sophia-v1"),
                "base": models.get("baseModel", ""),
                "passed": models.get("benchmarkScore", {}).get("passed"),
                "total": models.get("benchmarkScore", {}).get("total"),
                "scorePct": models.get("benchmarkScore", {}).get("scorePct"),
            },
            "ragClaude": {
                "passed": rag_summary.get("passed"),
                "total": rag_summary.get("total"),
                "scorePct": (
                    round((float(rag_summary.get("passed", 0)) / float(rag_summary.get("total", 1))) * 100, 2)
                    if rag_summary.get("total")
                    else None
                ),
            },
            "hiddenLatestPublicReport": hidden_report,
            "hiddenPreparedPack": hidden_report,
            "hiddenPackCommitments": commitments,
        },
        "proofLadder": [
            {"level": 0, "name": "Corpus and schema", "status": "implemented"},
            {"level": 1, "name": "Reproducible local benchmarks", "status": "implemented"},
            {"level": 2, "name": "RAG and local model baselines", "status": "implemented"},
            {"level": 3, "name": "Baseline and ablation comparisons", "status": "runner-implemented-awaiting-live-run"},
            {"level": 4, "name": "Hidden reviewer packs", "status": "protocol-ready"},
            {"level": 5, "name": "External public benchmarks", "status": "not_run"},
            {"level": 6, "name": "Third-party replication", "status": "not_run"},
        ],
        "externalBenchmarks": [
            {"name": "ARC-AGI / ARC-AGI-3", "status": "not_run", "purpose": "novel reasoning and skill acquisition"},
            {"name": "GAIA-style tasks", "status": "not_run", "purpose": "tool-using assistant reasoning"},
            {"name": "SWE-bench-style repo tasks", "status": "not_run", "purpose": "software maintenance agency"},
            {"name": "METR-style autonomy", "status": "not_run", "purpose": "long-horizon autonomous work"},
        ],
        "requiredProofData": [
            "pre-registered AGI definition and thresholds",
            "hidden reviewer packs that Sophia cannot see before evaluation",
            "baseline and ablation deltas against raw models and missing Sophia components",
            "long-horizon task logs with intervention counts",
            "learning-under-shift pre/post results with append-only memory records",
            "failure ledger with claim impact",
            "RLVR held-out pass@1 under the no-overclaim gate (pre-registered; Open until a gated live run)",
            "third-party reproduction on a clean clone",
        ],
        "artifactIndex": {
            "todo": "agi-proof/TODO.md",
            "definition": "agi-proof/definition.md",
            "thresholds": "agi-proof/preregistered-thresholds.md",
            "benchmarkResults": "agi-proof/benchmark-results/README.md",
            "externalBenchmarks": "agi-proof/external-benchmarks/README.md",
            "baselineAblation": "agi-proof/baseline-ablation/README.md",
            "baselineAblationRunner": "tools/run_ablation_sophia.py",
            "learningShiftRunner": "tools/run_learning_shift.py",
            "longHorizonHarness": "tools/run_long_horizon.py",
            "replicationHarness": "tools/run_replication_check.py",
            "architectureDiagram": "docs/09-Agent/Sophia-Architecture.md",
            "hiddenReview": "agi-proof/hidden-reviewer-packs/README.md",
            "manualSemanticReview": "agi-proof/hidden-reviewer-packs/MANUAL-SEMANTIC-REVIEW.md",
            "webEvidence": "docs/09-Agent/Online-RAG.md",
            "rubricReviewTools": "docs/09-Agent/Sophia-Agent.md",
            "hiddenPackCommitments": "agi-proof/hidden-reviewer-packs/prepared-pack-2026-06-19.commitments.json",
            "freshHiddenPackCommitments": (
                "agi-proof/hidden-reviewer-packs/fresh-reviewer-pack-2026-06-19.commitments.json"
            ),
            "longHorizon": "agi-proof/long-horizon-runs/README.md",
            "learningShift": "agi-proof/learning-under-shift/README.md",
            "failureLedger": "agi-proof/failure-ledger.md",
            "replication": "agi-proof/third-party-replication/README.md",
            "rlvrRunner": "tools/run_rlvr.py",
            "rlvrReward": "provenance_bench/rl_reward.py",
            "rlvrDataset": "provenance_bench/rl_dataset.py",
            "rlvrReport": "agi-proof/benchmark-results/rlvr.public-report.json",
            "rlvrExperiment": "docs/09-Agent/RLVR-Experiment.md",
            "localAgentDeltaRunner": "tools/run_local_agent_delta.py",
            "localAgentCore": "provenance_bench/local_agent.py",
            "localAgentDeltaReport": "agi-proof/benchmark-results/local-agent-delta.public-report.json",
            "localAgentDeltaScreenNullReport": "agi-proof/benchmark-results/provenance-delta-dolphin-lexical-screen-null.public-report.json",
            "unifiedUpliftRunner": "tools/run_unified_uplift.py",
            "unifiedUpliftReport": "agi-proof/benchmark-results/unified-uplift.public-report.json",
            "gateEntityAliases": "agent/entity_aliases.py",
            "gateRetrievalGrounding": "agent/grounded_gate.py",
            "gateClaimRouter": "agent/claim_router.py",
            "gateGradedDecision": "agent/graded_decision.py",
            "gateActiveLearning": "agent/gate_feedback.py",
            "gateTemporalVerifier": "agent/temporal_verifier.py",
            "gateActiveLearningPromotion": "tools/promote_pending.py",
            "codeExecVerifier": "provenance_bench/code_exec.py",
            "codeReward": "provenance_bench/code_reward.py",
            "codeBenchmark": "benchmark/code_tasks.json",
            "codeUpliftRunner": "tools/run_code_uplift.py",
            "councilPanelRunner": "tools/run_council_panel.py",
            "religionFigureCouncil": "docs/08-Domains/Religion-Figure-Council.md",
            "codingCouncil": "docs/08-Domains/Coding-Council.md",
            "factCheckRealityGapDoc": "docs/11-Platform/Fact-Check-Reality-Gap.md",
            "factCheckLiveEvalRunner": "tools/run_fact_check_live_eval.py",
            "factCheckLiveEvalReport": "agi-proof/fact-check-live/fact-check-live-eval.public-report.json",
            "factCheckFlywheelRunner": "tools/run_fact_check_flywheel.py",
            "reflexiveSelfGateRunner": "tools/run_reflexive_self_gate.py",
            "reflexiveSelfGateReport": "agi-proof/self-gate/reflexive-self-gate.public-report.json",
        },
    }


def manifest_text(manifest: dict[str, Any]) -> str:
    """Canonical serialisation of a manifest — the exact bytes written to disk."""
    return json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"


def write_manifest(output: Path = OUTPUT) -> Path:
    """Build the manifest and write it to ``output`` (default behaviour). Returns
    the path written."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(manifest_text(build_manifest()), encoding="utf-8")
    return output


def check_manifest(output: Path = OUTPUT) -> int:
    """Verify ``output`` is up to date WITHOUT writing. The volatile ``generated``
    timestamp is normalised out (a fresh manifest is built with the committed file's
    own ``generated``) so only real content drift counts. Returns 0 if current, 1
    on drift/missing (with a ``DRIFT`` message on stderr)."""
    if not output.exists():
        print(f"DRIFT: {output} is missing — run tools/build_agi_proof_package.py", file=sys.stderr)
        return 1
    existing = output.read_text(encoding="utf-8")
    try:
        parsed = json.loads(existing)
    except json.JSONDecodeError:
        parsed = None
    # Fail-closed on any malformed/non-object manifest: a list/scalar/None counts as
    # drift, not a crash (committed_generated stays None so the rebuild diverges).
    committed_generated = parsed.get("generated") if isinstance(parsed, dict) else None
    if manifest_text(build_manifest(generated=committed_generated)) != existing:
        print(f"DRIFT: {output} is stale — run tools/build_agi_proof_package.py", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build (or --check) the AGI-proof evidence manifest")
    parser.add_argument("--check", action="store_true", help="verify the committed manifest is current without writing")
    args = parser.parse_args()
    if args.check:
        return check_manifest()
    output = write_manifest()
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
