# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for the data asset registry (Phase 1)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_data_registry as bdr  # noqa: E402


def test_registry_is_deterministic() -> None:
    assert bdr.serialize(bdr.build_registry()) == bdr.serialize(bdr.build_registry())


def test_registry_covers_all_data_manifests() -> None:
    reg = bdr.build_registry()
    paths = {a["path"] for a in reg["assets"]}
    for mp in ROOT.glob("data/*/manifest.json"):
        assert mp.parent.relative_to(ROOT).as_posix() in paths


def test_every_asset_has_a_manifest_anchor() -> None:
    reg = bdr.build_registry()
    assert reg["assets"], "registry should not be empty"
    for a in reg["assets"]:
        assert len(a["manifestSha256"]) == 64        # sha256 hex
        assert a["kind"] in {"benchmark", "training", "dataset"}
        assert a["rows"] >= 0


def test_summary_counts_match_assets() -> None:
    reg = bdr.build_registry()
    assets = reg["assets"]
    assert reg["summary"]["nAssets"] == len(assets)
    assert reg["summary"]["nSealed"] == sum(1 for a in assets if a["sealed"])


def test_committed_registry_is_current() -> None:
    assert bdr.main(["--check"]) == 0
