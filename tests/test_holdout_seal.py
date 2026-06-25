# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
from __future__ import annotations

import json
from pathlib import Path

from provenance_bench.holdout_seal import (
    ALLOWED_HOLDOUT_READERS,
    build_manifest,
    content_hash,
    row_digest,
    verify_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
HOLDOUT = ROOT / "training" / "lora" / "holdout.jsonl"
MANIFEST = ROOT / "agi-proof" / "sophia-7b-train-verify" / "heldout-seal.manifest.json"


def test_holdout_manifest_matches_live() -> None:
    result = verify_manifest(MANIFEST, HOLDOUT)
    assert result["ok"], result


def test_row_digest_ignores_assistant_content() -> None:
    row = json.loads(HOLDOUT.read_text(encoding="utf-8").splitlines()[0])
    d1 = row_digest(row)
    mutated = dict(row)
    for msg in mutated.get("messages", []):
        if msg.get("role") == "assistant":
            msg["content"] = "TAMPERED"
    assert row_digest(mutated) == d1


def test_content_hash_stable() -> None:
    rows = [json.loads(line) for line in HOLDOUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert content_hash(rows) == build_manifest(HOLDOUT)["contentHash"]


def test_synthetic_builders_not_in_allowed_readers() -> None:
    synthetic = {
        "tools/wiki_to_training.py",
        "tools/mine_hard_negatives.py",
        "tools/build_moral_gate_sft.py",
    }
    assert not synthetic & ALLOWED_HOLDOUT_READERS


def test_tools_do_not_read_holdout_outside_allowlist() -> None:
    needle = "training/lora/holdout.jsonl"
    offenders: list[str] = []
    for path in sorted((ROOT / "tools").glob("*.py")):
        rel = str(path.relative_to(ROOT))
        if rel in ALLOWED_HOLDOUT_READERS:
            continue
        text = path.read_text(encoding="utf-8")
        if needle in text:
            offenders.append(rel)
    assert offenders == [], f"unexpected holdout readers: {offenders}"
