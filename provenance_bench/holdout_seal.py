# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Seal the LoRA holdout split for pre-registered train-verify runs.

The holdout file may contain gold assistant answers used only for validation /
early-stop — synthetic data builders must never read those answers. This module
commits a prompt-only manifest (id + user prompt digests) so tampering is
detectable, and exposes helpers for CI guards.

Pure stdlib, deterministic, offline.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from provenance_bench.dataset_guard import prompt_of

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOLDOUT = ROOT / "training" / "lora" / "holdout.jsonl"
DEFAULT_MANIFEST = ROOT / "agi-proof" / "sophia-7b-train-verify" / "heldout-seal.manifest.json"

# Modules allowed to read full holdout rows (answers included). Everyone else must
# use benchmark question sets / deleak guards — never ingest assistant gold.
ALLOWED_HOLDOUT_READERS = frozenset(
    {
        "tools/build_local_sophia_dataset.py",
        "tools/train_lora.py",
        "tools/prepare_lora_dataset.py",
        "tools/run_training_safety.py",
        "tools/seal_sophia_7b_holdout.py",
        "provenance_bench/holdout_seal.py",
    }
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def row_digest(row: dict[str, Any]) -> str:
    """Prompt-only digest — assistant gold is NOT hashed (stays off manifest)."""
    prompt = prompt_of(row) or ""
    payload = {
        "id": row.get("id", ""),
        "prompt": prompt,
        "holdoutReason": row.get("holdoutReason", ""),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def content_hash(rows: list[dict[str, Any]]) -> str:
    """Order-independent seal over prompt-only digests."""
    digests = sorted(row_digest(r) for r in rows)
    return hashlib.sha256("|".join(digests).encode("utf-8")).hexdigest()


def build_manifest(
    holdout_path: Path = DEFAULT_HOLDOUT,
    *,
    experiment: str = "sophia-7b-train-verify",
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
) -> dict[str, Any]:
    rows = _load_jsonl(holdout_path)
    return {
        "schema": "sophia.holdout_seal.v1",
        "experiment": experiment,
        "baseModel": base_model,
        "holdoutSource": str(holdout_path.relative_to(ROOT)),
        "rowCount": len(rows),
        "contentHash": content_hash(rows),
        "digestMethod": "sha256(json({id,prompt,holdoutReason}, sort_keys=True)) per row; "
        "contentHash = sha256(sorted row digests joined by |)",
        "assistantGoldPolicy": "withheld from manifest; synthetic builders must not read holdout answers",
        "allowedReaders": sorted(ALLOWED_HOLDOUT_READERS),
        "cases": [
            {
                "id": row.get("id", f"row-{i}"),
                "holdoutReason": row.get("holdoutReason", ""),
                "sha256": row_digest(row),
            }
            for i, row in enumerate(rows)
        ],
    }


def verify_manifest(
    manifest_path: Path = DEFAULT_MANIFEST,
    holdout_path: Path = DEFAULT_HOLDOUT,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = _load_jsonl(holdout_path)
    live_hash = content_hash(rows)
    ok = live_hash == manifest.get("contentHash") and len(rows) == manifest.get("rowCount")
    return {
        "ok": ok,
        "manifestHash": manifest.get("contentHash"),
        "liveHash": live_hash,
        "rowCount": len(rows),
        "manifestRowCount": manifest.get("rowCount"),
    }
