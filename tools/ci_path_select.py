#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Decide which CI lanes a change actually needs — to skip needless heavy runs.

FAIL-OPEN by design: the cheap always-on lane (fast-ci: lint_claims, attribution,
compileall, dependency-free unit tests) is ALWAYS required, and any path we don't
recognise — or any Python / data / requirements change — forces the FULL heavy
suite. We only mark heavy lanes skippable when the whole diff is provably
irrelevant to them (pure docs, skill-registry JSON, or workflow YAML). A
misclassification therefore costs CI minutes, never correctness.

Heavy lanes (match .github/workflows/ci.yml job names):
  validate-core · validate-reasoning · validate-build · validate-safety · test

    python3 tools/ci_path_select.py docs/x.md skills/registry/y.json
    git diff --name-only origin/main... | python3 tools/ci_path_select.py --stdin --json
"""
from __future__ import annotations

import argparse
import json
import sys

ALWAYS = ["fast-ci"]  # cheap gate, never skipped
HEAVY = ["validate-core", "validate-reasoning", "validate-build", "validate-safety", "test"]

# Python/source dirs whose changes need the FULL heavy suite.
PY_PREFIXES = (
    "agent/", "tools/", "tests/", "provenance_bench/", "selfextend/",
    "sophia_contract/", "sophia_mcp/", "gateway/", "okf/", "scripts/",
)
DATA_PREFIXES = ("data/", "training/", "eval/", "constitution/", "benchmark/")


def classify(path: str) -> str:
    p = path.strip()
    if p.startswith("./"):
        p = p[2:]
    if not p:
        return "other"
    # Python package code, deps → full suite
    if p.endswith((".py",)) and (p.startswith(PY_PREFIXES) or "/" not in p):
        return "pycode"
    if p in ("pyproject.toml",) or (p.startswith("requirements") and p.endswith(".txt")):
        return "pycode"
    if p.startswith("skills/") and p.endswith(".py"):
        return "pycode"
    # Data / packs / benchmarks / constitution → full suite (feeds decontam, gates, benches)
    if p.startswith(DATA_PREFIXES):
        return "data"
    # Skill-registry cards: only the skills tests need them
    if p.startswith("skills/registry/") and p.endswith(".json"):
        return "skill_json"
    # Wiki content: OKF/wiki validation lives in validate-core
    if p.startswith("wiki/"):
        return "wiki"
    # Pure docs (markdown/prose) → cheap lane only
    if p.startswith("docs/") or p.endswith((".md", ".rst", ".txt")):
        return "docs"
    # Workflow YAML: the changed workflow runs itself; no python correctness impact
    if p.startswith(".github/workflows/"):
        return "workflow"
    return "other"  # unknown → fail-open


# Which heavy lanes each category needs.
_NEEDS = {
    "pycode": set(HEAVY),
    "data": set(HEAVY),
    "other": set(HEAVY),                 # fail-open
    "skill_json": {"validate-core", "test"},  # test_skills runs in both
    "wiki": {"validate-core"},
    "docs": set(),
    "workflow": set(),
}


def select_lanes(paths: list[str]) -> dict:
    cats: dict[str, list[str]] = {}
    needed: set[str] = set()
    for path in paths:
        c = classify(path)
        cats.setdefault(c, []).append(path.strip())
        needed |= _NEEDS.get(c, set(HEAVY))
    if not paths:  # nothing to diff → fail-open
        needed = set(HEAVY)
    lanes = {lane: True for lane in ALWAYS}
    for lane in HEAVY:
        lanes[lane] = lane in needed
    skippable = [lane for lane in HEAVY if not lanes[lane]]
    full = set(needed) == set(HEAVY)
    return {
        "lanes": lanes,
        "required": ALWAYS + [lane for lane in HEAVY if lanes[lane]],
        "skippable": skippable,
        "categories": cats,
        "full": full,
        "note": "fail-open: unknown/py/data paths force the full suite; only docs/skill-json/workflow diffs skip heavy lanes; fast-ci always runs",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="changed file paths")
    ap.add_argument("--stdin", action="store_true", help="read newline-separated paths from stdin")
    ap.add_argument("--json", action="store_true", help="emit JSON (default: human summary)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = list(args.paths)
    if args.stdin:
        paths += [ln for ln in sys.stdin.read().splitlines() if ln.strip()]
    sel = select_lanes(paths)
    if args.json:
        print(json.dumps(sel, indent=2))
    else:
        print(f"required: {', '.join(sel['required'])}")
        print(f"skippable: {', '.join(sel['skippable']) or '(none — full suite)'}")
        print(f"full suite: {sel['full']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
