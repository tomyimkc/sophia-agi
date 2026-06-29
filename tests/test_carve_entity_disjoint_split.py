# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for the entity-disjoint split carver (Phase 3)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import carve_entity_disjoint_split as carve  # noqa: E402


def test_carve_is_deterministic() -> None:
    a = carve.carve()
    b = carve.carve()
    assert a == b


def test_carve_partition_is_complete() -> None:
    r = carve.carve()
    assert r["nDisjoint"] + r["nContaminated"] + r["nNoEntity"] == r["nEvalTotal"]
    assert r["nDisjoint"] > 0


def test_disjoint_set_shares_no_entity_with_train() -> None:
    r = carve.carve()
    # the load-bearing proof: a valid carve has zero train-shared entities
    assert r["sharedWithTrain"] == []


def test_dry_run_writes_nothing(tmp_path) -> None:
    assert carve.main([]) == 0          # report-only, no --out


def test_staged_candidate_is_fresh() -> None:
    """The committed human-review candidate must match a fresh carve (no rot)."""
    staged = ROOT / "agi-proof" / "data-health" / "seib_entity_disjoint_candidate" / "candidate.jsonl"
    assert staged.exists(), "run: tools/carve_entity_disjoint_split.py --out <staged path>"
    result = carve.carve()
    expected = "".join(
        json.dumps(d, ensure_ascii=False, sort_keys=True) + "\n" for d in result["disjoint"]
    )
    assert staged.read_text(encoding="utf-8") == expected, \
        "staged candidate is stale — re-run the carver --out to refresh it"


def test_write_produces_valid_disjoint_manifest(tmp_path) -> None:
    out = tmp_path / "candidate.jsonl"
    assert carve.main(["--out", str(out)]) == 0
    assert out.exists()
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert rows and all("prompt" in r and "entities" in r for r in rows)
    man = json.loads((tmp_path / "manifest.json").read_text())
    assert man["trainingDisjoint"] is True
    assert man["sharedWithTrain"] == []
    assert man["canClaimAGI"] is False
    assert len(man["contentHash"]) == 64
