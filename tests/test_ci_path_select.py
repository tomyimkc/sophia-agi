#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for tools/ci_path_select.py — fail-open CI lane selection."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ci_path_select import HEAVY, classify, select_lanes  # noqa: E402


def test_fast_ci_always_required() -> None:
    for paths in ([], ["docs/x.md"], ["tools/y.py"], ["weird.bin"]):
        assert "fast-ci" in select_lanes(paths)["required"]


def test_python_change_runs_full_suite() -> None:
    sel = select_lanes(["tools/runpod_train.py"])
    assert sel["full"] and not sel["skippable"]
    assert select_lanes(["requirements-lora.txt"])["full"]
    assert select_lanes(["pyproject.toml"])["full"]


def test_data_change_runs_full_suite() -> None:
    assert select_lanes(["training/local_sophia_7b/mlx/train.jsonl"])["full"]
    assert select_lanes(["data/team_agents_benchmark/manifest.json"])["full"]


def test_docs_only_skips_all_heavy() -> None:
    sel = select_lanes(["docs/Home.md", "README.md"])
    assert sel["required"] == ["fast-ci"]
    assert set(sel["skippable"]) == set(HEAVY)


def test_skill_registry_json_runs_only_skill_lanes() -> None:
    sel = select_lanes(["skills/registry/runpod-eta-estimate.json"])
    assert set(sel["required"]) == {"fast-ci", "validate-core", "test"}
    assert "validate-reasoning" in sel["skippable"]


def test_workflow_yaml_skips_heavy() -> None:
    sel = select_lanes([".github/workflows/ci.yml"])
    assert sel["required"] == ["fast-ci"]


def test_unknown_path_fails_open() -> None:
    assert select_lanes(["some/unknown.bin"])["full"]
    assert select_lanes([])["full"]  # empty diff -> fail-open


def test_mixed_takes_union_and_fails_open_to_full() -> None:
    # docs + python -> python forces full
    assert select_lanes(["docs/x.md", "agent/foo.py"])["full"]
    # docs + skill-json -> union = validate-core + test (not full)
    sel = select_lanes(["docs/x.md", "skills/registry/z.json"])
    assert set(sel["required"]) == {"fast-ci", "validate-core", "test"}


def test_classify_examples() -> None:
    assert classify("agent/x.py") == "pycode"
    assert classify("skills/registry/a.json") == "skill_json"
    assert classify("skills/core.py") == "pycode"
    assert classify("wiki/concept/x.md") == "wiki"
    assert classify(".github/workflows/ci.yml") == "workflow"
    assert classify("docs/x.md") == "docs"


def main() -> int:
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("test_ci_path_select: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
