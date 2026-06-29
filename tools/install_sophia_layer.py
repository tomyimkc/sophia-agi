#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""DRY-RUN validator/installer for Sophia's cross-harness operator surfaces.

Reads packaging/operator_manifest.json (skills, the verifier gate, the MCP
server) and reports, per harness, where each surface *would* be installed.

Inspired by affaan-m/ECC: one manifest + per-harness adapters, portable across
Claude Code, Codex, Cursor, Gemini, OpenCode, Zed and Copilot. This module is
DRY-RUN ONLY: it computes and prints planned actions but writes nothing to disk.
Real copying and per-harness adapter formats are OPEN (see the docs). The whole
module is deterministic and offline — no model, no network, no new dependencies.

canClaimAGI stays false.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "packaging" / "operator_manifest.json"

SCHEMA_ID = "sophia.operator.manifest.v1"
VALID_KINDS = {"skill", "mcp", "gate"}


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    """Load and parse the operator manifest as JSON."""
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _target_for(surface: dict, harness_id: str, harness: dict) -> str:
    """Compute the repo-relative target path for a surface under a harness.

    Skills install into the harness `skillsDir`; mcp/gate install into a flat
    `operator/` folder beside the skills dir so non-skill surfaces are explicit.
    """
    source = surface["source"]
    kind = surface.get("kind")
    name = Path(source).name
    if kind == "skill":
        skills_dir = harness.get("skillsDir", f".{harness_id}/skills")
        # Skills are directory-shaped (each has its own SKILL.md), so target the dir.
        return f"{skills_dir}/{surface['id']}/{name}"
    base = harness.get("skillsDir", f".{harness_id}/skills")
    parent = str(Path(base).parent)
    return f"{parent}/operator/{name}"


def validate_manifest(manifest: dict | None = None, root: Path = ROOT) -> tuple[bool, list[str]]:
    """Validate the manifest. Returns (ok, problems).

    Fails if: schema id is wrong; any surface source is missing on disk; a
    surface has an unknown kind; or a harness entry is malformed.
    """
    problems: list[str] = []
    if manifest is None:
        try:
            manifest = load_manifest()
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive
            return False, [f"manifest unreadable: {exc}"]

    if manifest.get("schema") != SCHEMA_ID:
        problems.append(f"schema must be {SCHEMA_ID!r}, got {manifest.get('schema')!r}")

    surfaces = manifest.get("surfaces")
    if not isinstance(surfaces, list) or not surfaces:
        problems.append("surfaces must be a non-empty list")
        surfaces = []
    seen_ids: set[str] = set()
    for i, surface in enumerate(surfaces):
        if not isinstance(surface, dict):
            problems.append(f"surface[{i}] is not an object")
            continue
        sid = surface.get("id")
        if not sid:
            problems.append(f"surface[{i}] missing id")
        elif sid in seen_ids:
            problems.append(f"duplicate surface id {sid!r}")
        else:
            seen_ids.add(sid)
        kind = surface.get("kind")
        if kind not in VALID_KINDS:
            problems.append(f"surface {sid!r} has invalid kind {kind!r}")
        source = surface.get("source")
        if not source:
            problems.append(f"surface {sid!r} missing source")
        elif not (root / source).exists():
            problems.append(f"surface {sid!r} source does not exist: {source}")

    harnesses = manifest.get("harnesses")
    if not isinstance(harnesses, dict) or not harnesses:
        problems.append("harnesses must be a non-empty object")
        harnesses = {}
    for hid, hcfg in harnesses.items():
        if not isinstance(hcfg, dict):
            problems.append(f"harness {hid!r} entry is not an object")
            continue
        if not hcfg.get("skillsDir"):
            problems.append(f"harness {hid!r} missing skillsDir")

    return (not problems), problems


def plan_install(harness_id: str, manifest: dict | None = None, root: Path = ROOT) -> list[dict]:
    """Compute (but do not perform) the install plan for one harness.

    Returns a list of action dicts: each has surface id, kind, source, the
    computed target path, and whether the source currently exists. Writes
    NOTHING to disk.
    """
    if manifest is None:
        manifest = load_manifest()
    harnesses = manifest.get("harnesses", {})
    if harness_id not in harnesses:
        raise KeyError(f"unknown harness {harness_id!r}; known: {sorted(harnesses)}")
    harness = harnesses[harness_id]
    plan: list[dict] = []
    for surface in manifest.get("surfaces", []):
        source = surface["source"]
        plan.append(
            {
                "id": surface["id"],
                "kind": surface.get("kind"),
                "source": source,
                "target": _target_for(surface, harness_id, harness),
                "source_exists": (root / source).exists(),
                "action": "would-copy (dry-run)",
            }
        )
    return plan


def render_plan(harness_id: str, plan: list[dict]) -> str:
    """Render a human-readable dry-run plan."""
    lines = [f"DRY-RUN install plan for harness {harness_id!r} ({len(plan)} surface(s)):"]
    for a in plan:
        mark = "OK " if a["source_exists"] else "MISSING"
        lines.append(f"  [{mark}] {a['kind']:<5} {a['id']}")
        lines.append(f"          source: {a['source']}")
        lines.append(f"          target: {a['target']}")
    lines.append("No files written (dry-run only).")
    return "\n".join(lines)


def offline_invariants(root: Path = ROOT) -> tuple[bool, dict]:
    """Self-contained offline invariants used by __main__ and the tests."""
    detail: dict = {}
    try:
        manifest = load_manifest()
        detail["manifest_loads"] = True
    except (OSError, json.JSONDecodeError) as exc:
        return False, {"manifest_loads": False, "error": str(exc)}

    ok, problems = validate_manifest(manifest, root=root)
    detail["validate_ok"] = ok
    detail["problems"] = problems
    detail["surfaces"] = len(manifest.get("surfaces", []))
    detail["harnesses"] = len(manifest.get("harnesses", []))

    # A plan can be computed for every harness and lists every surface.
    plan_ok = True
    per_harness: dict[str, int] = {}
    for hid in manifest.get("harnesses", {}):
        plan = plan_install(hid, manifest, root=root)
        per_harness[hid] = len(plan)
        if len(plan) != detail["surfaces"]:
            plan_ok = False
    detail["plan_per_harness"] = per_harness
    detail["plan_ok"] = plan_ok

    checks = [
        detail["manifest_loads"],
        ok,
        detail["surfaces"] >= 3,
        detail["harnesses"] >= 4,
        plan_ok,
    ]
    return all(checks), detail


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DRY-RUN Sophia cross-harness operator-layer installer.")
    p.add_argument("--harness", help="harness id to plan an install for (e.g. claude)")
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="dry-run only (default and only supported mode; writes nothing)",
    )
    p.add_argument("--validate", action="store_true", help="validate the manifest and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = load_manifest()

    if args.validate or not args.harness:
        ok, problems = validate_manifest(manifest)
        if problems:
            print("Manifest problems:")
            for prob in problems:
                print(f"  - {prob}")
        print(f"validate_manifest: {'OK' if ok else 'FAIL'}")
        if args.validate:
            return 0 if ok else 1

    if args.harness:
        plan = plan_install(args.harness, manifest)
        print(render_plan(args.harness, plan))

    ok, detail = offline_invariants()
    print(f"\nINVARIANTS {'PASS' if ok else 'FAIL'}: "
          f"{detail['surfaces']} surfaces, {detail['harnesses']} harnesses")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
