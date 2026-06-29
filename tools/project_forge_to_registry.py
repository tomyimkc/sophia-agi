#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Project promoted Skill Forge skills into routable harness registry specs.

The forge writes verifier evidence to ``skills/registry/forge_index.json``, but the
agent harness router (``agent/skills.py::load_all``) deliberately SKIPS that index
(it is bookkeeping, not an executable skill spec). So a promoted forged skill is
never selectable by the agent. This closes that gap: for every ``promotion_status ==
"accepted"`` entry in the forge index, emit a Layer-B skill spec
(``skills/registry/<task>.json``) whose ``whenToUse`` / ``triggers`` are derived from
the forge description and the validated verifier rules. Now a forged skill is
discoverable by ``select`` / ``select_ranked`` like any hand-written one.

Idempotent: re-running regenerates the same files. Use ``--check`` in CI to fail if
projections are stale (the ci-artifact-drift pattern).

Run:  python tools/project_forge_to_registry.py [--check]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "skills" / "registry"
FORGE_INDEX = REGISTRY / "forge_index.json"

# Marker so projected specs are recognisable + safe to overwrite/prune.
PROJECTED_SCHEMA = "sophia.skill_forge.projected.v1"
_STOP = set("the a an of to is are be and or detect intent text output".split())


def _task_slug(skill_id: str, task_id: str) -> str:
    return task_id or re.sub(r"^skill\.", "", skill_id)


def _triggers_from(description: str, rules: list[dict]) -> list[str]:
    """Salient trigger tokens: description words + the tokens each verifier rule keys on."""
    trig: list[str] = []
    for w in re.findall(r"[a-z0-9]+", (description or "").lower()):
        if len(w) > 2 and w not in _STOP and w not in trig:
            trig.append(w)
    for rule in rules or []:
        name = str(rule.get("name", ""))
        params = rule.get("params", {}) or {}
        tok = params.get("token")
        if tok and str(tok).lower() not in trig:
            trig.append(str(tok).lower())
        elif name.startswith("contains:"):
            t = name.split(":", 1)[1].lower()
            if t and t not in trig:
                trig.append(t)
    return trig[:12]


def _spec_for(entry: dict) -> dict | None:
    if entry.get("promotion_status") != "accepted":
        return None
    skill_id = entry.get("skill_id", "")
    task = _task_slug(skill_id, entry.get("task_id", ""))
    desc = entry.get("description") or f"forged classifier for {task}"
    rules = []
    promo = entry.get("promotion") or {}
    for adm in (promo.get("admitted") or []):
        rules.append({"name": adm.get("name", ""), "params": {}})
    # richer rule params live in the per-skill manifest; load if present
    manifest = ROOT / (entry.get("skill_dir") or "") / "manifest.json"
    if manifest.exists():
        try:
            rules = json.loads(manifest.read_text(encoding="utf-8")).get("rules", rules)
        except Exception:
            pass
    acc = (entry.get("best_validation") or {}).get("accuracy")
    when = (f"Use when the goal is to {desc} (forged, verifier-gated). "
            f"Run the forged skill {skill_id} via Sophia Gateway for a provenance-stamped verdict.")
    return {
        "schema": PROJECTED_SCHEMA,
        "name": skill_id,
        "whenToUse": when,
        "triggers": _triggers_from(desc, rules),
        "requiredTools": ["skillforge", "gateway"],
        "workflow": [
            f"Load the forged skill from {entry.get('skill_dir', 'skills/generated/<task>')}.",
            f"Call {skill_id} through Sophia Gateway so the output is verified + provenance-stamped.",
            "Return the boolean verdict; abstain (hold) if the Gateway rejects.",
        ],
        "ioSchema": {"input": {"text": "string"}, "output": {"answer": "boolean"}},
        "verification": [
            f"forge_index best_validation accuracy >= threshold ({acc if acc is not None else 'n/a'}).",
            "Gateway verifier accepted the call.",
        ],
        "commonFailures": [
            "Paraphrases outside the validated example distribution.",
            "Treating a held (abstained) verdict as a negative.",
        ],
        "examples": [
            {"input": f"text relevant to {task}", "output": "answer: true/false with skillforge:// source"}
        ],
        "_source": {"forge_skill_id": skill_id, "skill_dir": entry.get("skill_dir")},
    }


def _projected_path(skill_id: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", skill_id.lower()).strip("-")
    return REGISTRY / f"{slug}.json"


def _rel(path: Path) -> str:
    """Display path relative to ROOT when possible; tolerate redirected REGISTRY (tests)."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build(check: bool = False) -> int:
    if not FORGE_INDEX.exists():
        print("forge→registry: no forge_index.json — nothing to project.")
        return 0
    index = json.loads(FORGE_INDEX.read_text(encoding="utf-8"))
    wanted: dict[Path, dict] = {}
    for entry in index.get("skills", []):
        spec = _spec_for(entry)
        if spec is not None:
            wanted[_projected_path(spec["name"])] = spec

    # existing projected files (so we can prune ones whose forge skill was demoted/removed)
    existing = {}
    for p in REGISTRY.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("schema") == PROJECTED_SCHEMA:
            existing[p] = data

    stale: list[str] = []
    for path, spec in wanted.items():
        new = json.dumps(spec, indent=2, ensure_ascii=False) + "\n"
        if not path.exists() or path.read_text(encoding="utf-8") != new:
            stale.append(_rel(path))
            if not check:
                path.write_text(new, encoding="utf-8")
    for path in existing:
        if path not in wanted:
            stale.append(f"{_rel(path)} (prune: forge skill no longer accepted)")
            if not check:
                path.unlink()

    if check and stale:
        print("forge→registry: STALE projections (run without --check to regenerate):")
        for s in stale:
            print(f"  - {s}")
        return 1
    if stale:
        print(f"forge→registry: wrote/updated {len(stale)} projection(s):")
        for s in stale:
            print(f"  - {s}")
    else:
        print(f"forge→registry: OK — {len(wanted)} projection(s) up to date.")
    return 0


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail if any projection is stale (CI)")
    args = ap.parse_args(argv)
    return build(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
