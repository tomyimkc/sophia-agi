#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Projection makes a promoted forged skill routable by the agent harness."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import skills  # noqa: E402
from tools import project_forge_to_registry as proj  # noqa: E402

ACCEPTED_ENTRY = {
    "skill_id": "skill.destructive-command",
    "task_id": "destructive-command",
    "description": "flag destructive delete intent",
    "skill_dir": "skills/generated/destructive-command",
    "promotion_status": "accepted",
    "best_validation": {"accuracy": 1.0},
    "promotion": {"admitted": [{"name": "contains:delete"}]},
}
REJECTED_ENTRY = {
    "skill_id": "skill.vague", "task_id": "vague", "promotion_status": "rejected",
    "promotion": {"admitted": []},
}


def _setup(tmp: Path):
    reg = tmp / "registry"
    reg.mkdir(parents=True)
    (reg / "forge_index.json").write_text(json.dumps({
        "schema": "sophia.skill_forge.registry.v1",
        "skills": [ACCEPTED_ENTRY, REJECTED_ENTRY],
    }), encoding="utf-8")
    proj.REGISTRY = reg
    proj.FORGE_INDEX = reg / "forge_index.json"
    return reg


def test_projection_creates_routable_spec():
    with tempfile.TemporaryDirectory() as t:
        reg = _setup(Path(t))
        rc = proj.build(check=False)
        assert rc == 0
        out = reg / "skill-destructive-command.json"
        assert out.exists(), "projected spec not written"
        spec = json.loads(out.read_text())
        assert spec["schema"] == proj.PROJECTED_SCHEMA
        assert "delete" in spec["triggers"], spec["triggers"]
        # rejected skills are never projected
        assert not (reg / "skill-vague.json").exists()
        # ...and the projected spec is selectable by the hybrid router
        best = skills.select("please delete the whole database", skill_dir=reg)
        assert best and best["name"] == "skill.destructive-command", best


def test_check_mode_flags_stale_then_clean():
    with tempfile.TemporaryDirectory() as t:
        _setup(Path(t))
        assert proj.build(check=True) == 1   # nothing written yet -> stale
        assert proj.build(check=False) == 0  # write
        assert proj.build(check=True) == 0   # now clean


def test_demotion_prunes_projection():
    with tempfile.TemporaryDirectory() as t:
        reg = _setup(Path(t))
        proj.build(check=False)
        assert (reg / "skill-destructive-command.json").exists()
        # forge skill demoted -> projection should be pruned
        (reg / "forge_index.json").write_text(json.dumps({
            "schema": "sophia.skill_forge.registry.v1",
            "skills": [{**ACCEPTED_ENTRY, "promotion_status": "rejected"}],
        }), encoding="utf-8")
        proj.build(check=False)
        assert not (reg / "skill-destructive-command.json").exists()


def main() -> int:
    test_projection_creates_routable_spec()
    test_check_mode_flags_stale_then_clean()
    test_demotion_prunes_projection()
    print("test_project_forge_to_registry: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
