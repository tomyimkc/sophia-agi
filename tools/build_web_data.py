#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Bundle repo stats and leaderboards into web/data/manifest.json."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB_DATA = ROOT / "web" / "data" / "manifest.json"
DOMAINS = ("philosophy", "psychology", "history", "religion", "personality")

# Items mentioning the training method/recipe are kept OFF the public manifest
# (see tools/lint_web_privacy.py for the published-site privacy policy).
_TRAINING_DETAIL_RE = re.compile(r"RLVR|pass@1|reward model|fine[\s\-]?tun", re.I)


def _sanitize_required(items: list) -> list:
    """Drop required-proof items that would reveal the training method."""
    return [item for item in items if not _TRAINING_DETAIL_RE.search(str(item))]


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


def _patreon_supporters_summary() -> dict:
    """Lightweight public summary for the web manifest.
    Only count + tier names (no individual names) to keep it tasteful.
    """
    supporters_path = ROOT / "data" / "patreon" / "supporters.json"
    tiers_config_path = ROOT / "data" / "patreon" / "tiers.json"

    if not supporters_path.exists():
        return {}

    try:
        data = json.loads(supporters_path.read_text(encoding="utf-8"))
        tiers = data.get("tiers", {}) or {}
        total = data.get("count", 0) or sum(len(v) for v in tiers.values())

        # Prefer canonical order + nice names from config
        tier_order = data.get("tier_order") or []
        if not tier_order and tiers_config_path.exists():
            cfg = json.loads(tiers_config_path.read_text(encoding="utf-8"))
            tier_order = [t["patreon_title"] for t in cfg.get("tiers", [])]

        # Only include tiers that actually have members, in preferred order
        active_tiers = [t for t in tier_order if t in tiers and tiers[t]]
        if not active_tiers:
            active_tiers = [t for t, names in tiers.items() if names]

        return {
            "count": int(total),
            "tiers": active_tiers,
            "lastSync": data.get("synced_at"),
            "patreonUrl": "https://www.patreon.com/c/aideveloper_tomyim",
        }
    except (json.JSONDecodeError, OSError, KeyError) as exc:
        # Malformed or missing supporters data should not crash the web build, but
        # a bare pass would hide a real bug — log to stderr so it is debuggable.
        import sys
        print(f"[build_web_data] warning: patreon supporters summary skipped: {exc}", file=sys.stderr)
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
    # Public-safe manifest only. Intentionally EXCLUDES training/architecture
    # details (base model, adapter/LoRA path, internal module map / artifactIndex,
    # training-pipeline runners). The site renders results + the claim boundary,
    # not how the system is built. The web privacy guard enforces this.
    score_pct = models.get("benchmarkScore", {}).get("scorePct")
    payload = {
        "version": version,
        "trainingExamples": examples,
        "domains": list(DOMAINS),
        "leaderboards": leaderboards,
        "rag": {
            "indexChunks": _rag_chunk_count(),
        },
        "localModel": {
            # score only — no base model, adapter path, or model repo
            "benchmark": {"scorePct": score_pct} if score_pct is not None else {},
        },
        "agiProof": {
            "claimBoundary": agi_proof.get(
                "claimBoundary",
                "Sophia is an AGI-candidate proof package; true AGI is not claimed.",
            ),
            "proofLadder": agi_proof.get("proofLadder", []),
            "externalBenchmarks": agi_proof.get("externalBenchmarks", []),
            "requiredProofData": _sanitize_required(agi_proof.get("requiredProofData", [])),
            "docs": "agi-proof/README.md",
        },
        "links": {
            "github": "https://github.com/tomyimkc/sophia-agi",
            "huggingface": "https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus",
            "patreon": "https://www.patreon.com/c/aideveloper_tomyim",
        },
        "supporters": _patreon_supporters_summary(),
    }
    WEB_DATA.parent.mkdir(parents=True, exist_ok=True)
    WEB_DATA.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {WEB_DATA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
