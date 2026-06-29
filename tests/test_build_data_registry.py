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


def test_lineage_edges_are_real_declared_and_anchored() -> None:
    reg = bdr.build_registry()
    lin = reg["lineage"]
    # registryVersion is a sha256 anchor for pinning eval runs to a corpus/registry state
    assert len(lin["registryVersion"]) == 64
    paths = {a["path"] for a in reg["assets"]}
    rels = {r for _, r in bdr._LINEAGE_FIELDS}
    seen_from = set()
    for e in lin["edges"]:
        # every edge originates from a real registered asset and a known relation
        assert e["from"] in paths
        assert e["rel"] in rels
        assert isinstance(e["to"], str) and e["to"]
        seen_from.add(e["from"])
    assert lin["nEdges"] == len(lin["edges"])
    assert lin["assetsWithDeclaredUpstream"] == len(seen_from)
    assert lin["assetsWithoutDeclaredUpstream"] == len(reg["assets"]) - len(seen_from)


def test_lineage_is_derived_not_invented() -> None:
    # An edge must correspond to a field literally present in the source manifest — the
    # registry never fabricates provenance.
    import json
    reg = bdr.build_registry()
    by_path = {a["path"]: a["manifest"] for a in reg["assets"]}
    for e in reg["lineage"]["edges"]:
        doc = json.loads((ROOT / by_path[e["from"]]).read_text(encoding="utf-8"))
        declared = {str(v).strip() for k, v in doc.items() if isinstance(v, str)}
        declared |= {str(x).strip() for v in doc.values() if isinstance(v, list) for x in v}
        assert e["to"] in declared, f"edge target {e['to']!r} not declared in {e['from']} manifest"


def test_committed_registry_is_current() -> None:
    assert bdr.main(["--check"]) == 0
