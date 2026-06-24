#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Bundle repo stats and leaderboards into web/data/manifest.json."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB_DATA = ROOT / "web" / "data" / "manifest.json"
DOMAINS = ("philosophy", "psychology", "history", "religion", "personality")


def _rag_chunk_count() -> int:
    chunks_path = ROOT / "rag" / "index" / "chunks.jsonl"
    if not chunks_path.exists():
        return 0
    return sum(1 for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip())


def _models_meta() -> dict:
    manifest_path = ROOT / "models" / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}


def _agi_proof_meta() -> dict:
    manifest_path = ROOT / "agi-proof" / "evidence-manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}


def main() -> int:
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    examples = len(list((ROOT / "training" / "examples").glob("*.json")))
    leaderboards = {}
    for domain in DOMAINS:
        path = ROOT / "benchmark" / "results" / f"leaderboard-{domain}.json"
        if path.exists():
            leaderboards[domain] = json.loads(path.read_text(encoding="utf-8"))

    models = _models_meta()
    agi_proof = _agi_proof_meta()
    payload = {
        "version": version,
        "trainingExamples": examples,
        "domains": list(DOMAINS),
        "leaderboards": leaderboards,
        "rag": {
            "indexChunks": _rag_chunk_count(),
            "backends": ["gemini", "vertex", "claude"],
            "webEvidenceProviders": ["brave", "tavily", "serpapi"],
            "webEvidenceDefault": "off-for-hidden-evals",
            "cli": "python tools/sophia_rag.py",
            "docs": "docs/09-Agent/Online-RAG.md",
        },
        "localModel": {
            "name": "sophia-v1",
            "base": models.get("baseModel", "Qwen/Qwen2.5-3B-Instruct"),
            "benchmark": models.get("benchmarkScore", {}),
            "hf": models.get("hfModelRepo", ""),
        },
        "agiProof": {
            "claimBoundary": agi_proof.get(
                "claimBoundary",
                "Sophia is an AGI-candidate proof package; true AGI is not claimed.",
            ),
            "proofLadder": agi_proof.get("proofLadder", []),
            "externalBenchmarks": agi_proof.get("externalBenchmarks", []),
            "requiredProofData": agi_proof.get("requiredProofData", []),
            "artifactIndex": agi_proof.get("artifactIndex", {}),
            "docs": "agi-proof/README.md",
            "manifest": "agi-proof/evidence-manifest.json",
        },
        "conscienceKernel": {
            "claimBoundary": "Candidate moral + epistemic control infrastructure; not proof of AGI.",
            "verdicts": ["allow", "revise", "retrieve", "clarify", "escalate", "abstain", "block"],
            "paths": [
                "agent/conscience.py",
                "agent/metacognition.py",
                "constitution/constitution.v1.json",
                "agent/deontic_verifier.py",
                "agent/moral_aggregator.py",
                "agent/constitutional_classifier.py",
                "agent/deception_signals.py",
                "sophia_mcp/server.py",
            ],
            "docs": "docs/11-Platform/Conscience-Kernel.md",
            "artifact": "agi-proof/conscience/conscience-eval.public-report.json",
            "benchmark": "python tools/build_conscience_proof_package.py",
        },
        "moralGateV2": {
            "claimBoundary": "Functional moral-control system grounded in an overlapping-consensus public standard; not subjective moral consciousness and not AGI proof.",
            "method": "Rawlsian overlapping consensus: cross-tradition hard floor + gray-zone moral parliament (8 theories, Confucian and Daoist kept distinct) + legitimacy provenance (is/ought).",
            "paths": [
                "moral_corpus/public_standard.v1.json",
                "agent/moral_ontology.py",
                "agent/public_standard_gate.py",
                "constitution/constitution.v2.json",
                "agent/moral_aggregator.py",
            ],
            "docs": "docs/11-Platform/Public-Moral-Standard.md",
            "artifact": "agi-proof/conscience/moral-public-standard-eval.public-report.json",
            "benchmark": "python tools/run_moral_public_standard_eval.py",
        },
        "allPhaseBenchmarks": {
            "claimBoundary": "Candidate all-phase benchmark infrastructure; not AGI proof and not public GPQA/SWE/LiveCodeBench/LMSYS results until real-model validation clears the no-overclaim gate.",
            "phases": [
                "SEIB-100",
                "Belief Revision 50",
                "AgentBench-Sophia 30",
                "GPQA-Provenance smoke",
                "Code Provenance 30",
                "SEIB-Arena-20 smoke",
            ],
            "paths": [
                "eval/seib/seib_100_v1.jsonl",
                "eval/belief_revision/belief_revision_50_v1.jsonl",
                "eval/agentbench_sophia/agentbench_sophia_30_v1.jsonl",
                "eval/gpqa_provenance/gpqa_provenance_smoke_v1.jsonl",
                "eval/code_provenance/code_provenance_30_v1.jsonl",
                "eval/arena/arena_20_v1.jsonl",
            ],
            "docs": "docs/11-Platform/Benchmark-Phases.md",
            "artifact": "agi-proof/benchmark-results/all-phase-benchmarks.public-report.json",
            "benchmark": "python tools/run_all_phase_benchmarks.py",
        },
        "agiKernel": {
            "claimBoundary": "Missing-pillars mechanisms are candidate infrastructure, not Level-3 AGI evidence.",
            "docs": "docs/11-Platform/AGI-Missing-Pillars.md",
            "artifact": "agi-proof/agi-kernel/missing-pillars.public-report.json",
            "benchmark": "python tools/run_agi_missing_pillars.py",
        },
        "links": {
            "github": "https://github.com/tomyimkc/sophia-agi",
            "huggingface": "https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus",
            "hfModel": "https://huggingface.co/tomyimkc/sophia-agi-lora-v1",
        },
    }
    WEB_DATA.parent.mkdir(parents=True, exist_ok=True)
    WEB_DATA.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {WEB_DATA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
