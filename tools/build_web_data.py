#!/usr/bin/env python3
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
