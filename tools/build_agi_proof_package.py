#!/usr/bin/env python3
"""Build Sophia's AGI-candidate proof evidence manifest.

The manifest is intentionally conservative: it records current reproducible
evidence and open proof gaps without claiming Sophia is proven AGI.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AGI_PROOF_DIR = ROOT / "agi-proof"
OUTPUT = AGI_PROOF_DIR / "evidence-manifest.json"
DOMAINS = ("philosophy", "psychology", "history", "religion")


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


def build_manifest() -> dict[str, Any]:
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    models = load_json(ROOT / "models" / "manifest.json", {})
    rag_summary = load_json(ROOT / "benchmark" / "model_runs" / "rag-claude-summary.json", {})
    leaderboards = leaderboard_summary()
    benchmark_total = sum(item["cases"] for item in leaderboards.values())

    return {
        "version": version,
        "generated": datetime.now().isoformat(timespec="seconds"),
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
        },
        "proofLadder": [
            {"level": 0, "name": "Corpus and schema", "status": "implemented"},
            {"level": 1, "name": "Reproducible local benchmarks", "status": "implemented"},
            {"level": 2, "name": "RAG and local model baselines", "status": "implemented"},
            {"level": 3, "name": "Baseline and ablation comparisons", "status": "protocol-ready"},
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
            "third-party reproduction on a clean clone",
        ],
        "artifactIndex": {
            "definition": "agi-proof/definition.md",
            "thresholds": "agi-proof/preregistered-thresholds.md",
            "externalBenchmarks": "agi-proof/external-benchmarks/README.md",
            "baselineAblation": "agi-proof/baseline-ablation/README.md",
            "hiddenReview": "agi-proof/hidden-reviewer-packs/README.md",
            "longHorizon": "agi-proof/long-horizon-runs/README.md",
            "learningShift": "agi-proof/learning-under-shift/README.md",
            "failureLedger": "agi-proof/failure-ledger.md",
            "replication": "agi-proof/third-party-replication/README.md",
        },
    }


def main() -> int:
    AGI_PROOF_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    OUTPUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
