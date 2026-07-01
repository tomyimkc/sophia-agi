#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the cross-harness operator-layer manifest + DRY-RUN installer.

Deterministic, offline — no model, no network, no new dependencies.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import install_sophia_layer as isl  # noqa: E402

MANIFEST_PATH = ROOT / "packaging" / "operator_manifest.json"


def test_manifest_is_well_formed_json() -> None:
    with MANIFEST_PATH.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert manifest["schema"] == "sophia.operator.manifest.v1"
    assert isinstance(manifest["surfaces"], list)
    assert isinstance(manifest["harnesses"], dict)


def test_validate_manifest_ok_all_sources_exist() -> None:
    ok, problems = isl.validate_manifest()
    assert ok, problems
    assert problems == []


def test_at_least_three_surfaces_and_four_harnesses() -> None:
    manifest = isl.load_manifest()
    assert len(manifest["surfaces"]) >= 3
    assert len(manifest["harnesses"]) >= 4


def test_surface_kinds_cover_skill_mcp_gate() -> None:
    manifest = isl.load_manifest()
    kinds = {s["kind"] for s in manifest["surfaces"]}
    assert {"skill", "mcp", "gate"} <= kinds


def test_validate_catches_missing_source() -> None:
    bad = isl.load_manifest()
    bad["surfaces"] = list(bad["surfaces"]) + [
        {"id": "phantom", "kind": "skill", "source": "skills/portable/does-not-exist/SKILL.md"}
    ]
    ok, problems = isl.validate_manifest(bad)
    assert not ok
    assert any("does-not-exist" in p for p in problems)


def test_dry_run_plan_lists_actions_and_writes_nothing() -> None:
    harness_dirs = [
        ROOT / ".claude",
        ROOT / ".cursor",
        ROOT / ".grok",
        ROOT / ".agents",
    ]
    existing = [d for d in harness_dirs if d.exists()]
    before = {d: _tree_mtimes(d) for d in existing}

    plan = isl.plan_install("claude")
    assert isinstance(plan, list)
    assert len(plan) >= 1
    # Every action declares a source, a target, and is explicitly dry-run.
    for action in plan:
        assert action["source"]
        assert action["target"]
        assert action["action"] == "would-copy (dry-run)"

    after = {d: _tree_mtimes(d) for d in existing}
    assert before == after, "dry-run must not mutate any harness directory"


def test_offline_invariants_pass() -> None:
    ok, detail = isl.offline_invariants()
    assert ok, detail
    assert detail["surfaces"] >= 3
    assert detail["harnesses"] >= 4
    assert detail["plan_ok"]


def _tree_mtimes(d: Path) -> dict[str, float]:
    """Snapshot path -> mtime for every entry under d (used to prove no writes)."""
    return {str(p.relative_to(d)): p.stat().st_mtime for p in sorted(d.rglob("*"))}


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} install-sophia-layer tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
