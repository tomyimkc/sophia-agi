#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Unified skill index — one manifest every layer can read.

The repo has three sibling skill systems that don't talk to each other:
  A. Agent Skills      .claude/skills/*/SKILL.md   (model-chosen by description)
  B. Harness skills    skills/registry/*.json      (token-router via agent/skills.py)
  C. Skill Forge       skills/registry/forge_index.json + skills/generated/

This builds ``skills/registry/index.json``: a single ``{id, layer, description,
triggers, body_ref, verifier_ref, reliability}`` row per skill, generated from all
three sources. The Claude Code agent, the harness router, and the forge can then
select from ONE surface — "author a skill once, discover it everywhere".

Generated artifact (do not hand-edit). Use ``--check`` in CI to fail on drift
(the ci-artifact-drift pattern).

Run:  python tools/build_skill_index.py [--check]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAUDE_SKILLS = ROOT / ".claude" / "skills"
REGISTRY = ROOT / "skills" / "registry"
FORGE_INDEX = REGISTRY / "forge_index.json"
INDEX = REGISTRY / "index.json"

# NOTE: must contain "registry" so agent/skills.py::_is_metadata_file treats index.json
# as bookkeeping (skipped by the router's load_all), not as an executable skill spec.
SCHEMA = "sophia.skill_registry.unified_index.v1"
PROJECTED_SCHEMA = "sophia.skill_forge.projected.v1"


def _fm_description(md: str) -> str:
    if not md.startswith("---"):
        return ""
    end = md.find("\n---", 3)
    fm = md[3 : end if end != -1 else len(md)]
    out: list[str] = []
    capturing = False
    for ln in fm.splitlines():
        m = re.match(r"\s*description:\s*(.*)$", ln)
        if m:
            capturing = True
            rest = m.group(1).strip()
            if rest and rest not in (">", "|", ">-", "|-", ">+", "|+"):
                out.append(rest)
            continue
        if capturing:
            if re.match(r"\s*\w[\w-]*:\s", ln) and not ln.startswith((" ", "\t")):
                break
            if ln.strip() == "" and out:
                break
            out.append(ln.strip())
    return " ".join(x for x in out if x).strip()


def _triggers_from_text(text: str, limit: int = 12) -> list[str]:
    stop = set("the a an of to is are be and or use when whenever before after this that".split())
    out: list[str] = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if len(w) > 2 and w not in stop and w not in out:
            out.append(w)
        if len(out) >= limit:
            break
    return out


def collect() -> list[dict]:
    rows: list[dict] = []

    # Layer A — Agent Skills
    if CLAUDE_SKILLS.exists():
        for p in sorted(CLAUDE_SKILLS.glob("*/SKILL.md")):
            md = p.read_text(encoding="utf-8", errors="replace")
            if "\x00GITCRYPT" in md[:32]:
                rows.append({"id": p.parent.name, "layer": "A", "description": "(encrypted — unlock to index)",
                             "triggers": [], "body_ref": str(p.relative_to(ROOT)),
                             "verifier_ref": None, "reliability": None, "encrypted": True})
                continue
            desc = _fm_description(md)
            rows.append({
                "id": p.parent.name, "layer": "A", "description": desc,
                "triggers": _triggers_from_text(desc), "body_ref": str(p.relative_to(ROOT)),
                "verifier_ref": None, "reliability": None,
            })

    # Layer B — harness registry specs (skip bookkeeping/index/projected handled below)
    forge_dirs = set()
    if REGISTRY.exists():
        for p in sorted(REGISTRY.glob("*.json")):
            data = json.loads(p.read_text(encoding="utf-8"))
            schema = str(data.get("schema", ""))
            if p.name in ("index.json", "forge_index.json"):
                continue
            if schema == PROJECTED_SCHEMA:
                rows.append({
                    "id": data.get("name", p.stem), "layer": "C", "description": data.get("whenToUse", ""),
                    "triggers": data.get("triggers", []), "body_ref": str(p.relative_to(ROOT)),
                    "verifier_ref": "grounding",
                    "reliability": None,
                    "source": (data.get("_source") or {}).get("skill_dir"),
                })
                forge_dirs.add((data.get("_source") or {}).get("forge_skill_id"))
                continue
            if "whenToUse" not in data:
                continue
            rows.append({
                "id": data.get("name", p.stem), "layer": "B", "description": data.get("whenToUse", ""),
                "triggers": data.get("triggers", []), "body_ref": str(p.relative_to(ROOT)),
                "verifier_ref": ("verifier" if data.get("verification") else None), "reliability": None,
            })

    # Layer C — forged skills not yet projected (so they still show up in the index)
    if FORGE_INDEX.exists():
        fidx = json.loads(FORGE_INDEX.read_text(encoding="utf-8"))
        for e in fidx.get("skills", []):
            if e.get("promotion_status") != "accepted" or e.get("skill_id") in forge_dirs:
                continue
            rows.append({
                "id": e.get("skill_id"), "layer": "C", "description": e.get("description", ""),
                "triggers": _triggers_from_text(e.get("description", "")),
                "body_ref": e.get("skill_dir"), "verifier_ref": e.get("verifier_ref", "grounding"),
                "reliability": (e.get("best_validation") or {}).get("accuracy"),
            })

    rows.sort(key=lambda r: (r["layer"], str(r["id"])))
    return rows


def build(check: bool = False) -> int:
    rows = collect()
    doc = {"schema": SCHEMA, "count": len(rows),
           "layers": {"A": "agent-skill (.claude)", "B": "harness-registry", "C": "skill-forge"},
           "skills": rows}
    new = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
    cur = INDEX.read_text(encoding="utf-8") if INDEX.exists() else ""
    if new == cur:
        print(f"skill-index: OK — {len(rows)} skill(s), index.json up to date.")
        return 0
    if check:
        print("skill-index: STALE — run `python tools/build_skill_index.py` to regenerate "
              "skills/registry/index.json.")
        return 1
    INDEX.write_text(new, encoding="utf-8")
    print(f"skill-index: wrote skills/registry/index.json ({len(rows)} skills across layers A/B/C).")
    return 0


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail if index.json is stale (CI)")
    args = ap.parse_args(argv)
    return build(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
