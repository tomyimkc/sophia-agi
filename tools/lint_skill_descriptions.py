#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Skill-description linter — fail the build if a skill's trigger surface is too thin.

Because of progressive disclosure, an Agent Skill's ``description`` (and a harness
skill's ``whenToUse`` + ``triggers``) is the ENTIRE surface the router/model sees at
selection time. A vague description never triggers. This deterministic, offline
linter scores each skill's trigger-richness and fails on thin ones, exactly the way
``tools/lint_claims.py`` fails on overclaims.

Checks (per skill):
  - description present and >= MIN_DESC_CHARS;
  - contains at least one action/trigger verb ("use when", "run", "before", ...);
  - contains at least MIN_TRIGGER_TOKENS distinctive trigger tokens;
  - has at least one token NOT shared with every sibling skill (distinctiveness);
  - (Agent Skills only) names a slash alias "/<name>" — recommended, warn-only.

Run:  python tools/lint_skill_descriptions.py [--check]
Exit: 0 = all skills pass, 1 = at least one thin description.
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

MIN_DESC_CHARS = 80
MIN_TRIGGER_TOKENS = 4

ACTION_CUES = (
    "use when", "use whenever", "run", "run before", "before", "after", "whenever",
    "trigger", "fires on", "apply", "invoke", "when you", "when the user",
)

# Common words that do not make a description distinctive.
STOP = set("""
a an the and or of to in on for with without this that these those is are be use using used
your you it its as at by from into out up down skill skills repo project code agent agents when
""".split())


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP and len(t) > 2}


def _frontmatter_description(md: str) -> str:
    """Extract the YAML frontmatter ``description:`` (handles >- folded blocks)."""
    if not md.startswith("---"):
        return ""
    end = md.find("\n---", 3)
    fm = md[3 : end if end != -1 else len(md)]
    lines = fm.splitlines()
    out: list[str] = []
    capturing = False
    for ln in lines:
        m = re.match(r"\s*description:\s*(.*)$", ln)
        if m:
            capturing = True
            rest = m.group(1).strip()
            if rest and rest not in (">", "|", ">-", "|-", ">+", "|+"):
                out.append(rest)
            continue
        if capturing:
            # folded block continues while indented; a new top-level key ends it
            if re.match(r"\s*\w[\w-]*:\s", ln) and not ln.startswith((" ", "\t")):
                break
            if ln.strip() == "" and out:
                break
            out.append(ln.strip())
    return " ".join(x for x in out if x).strip()


def _collect() -> list[dict]:
    """Return [{kind, name, description, trigger_tokens, path}] for every skill."""
    skills: list[dict] = []
    # Layer A — Agent Skills
    if CLAUDE_SKILLS.exists():
        for p in sorted(CLAUDE_SKILLS.glob("*/SKILL.md")):
            md = p.read_text(encoding="utf-8", errors="replace")
            if md.lstrip().startswith("\x00GITCRYPT") or "\x00GITCRYPT" in md[:32]:
                continue  # locked/encrypted; can't lint ciphertext
            desc = _frontmatter_description(md)
            name = p.parent.name
            skills.append({
                "kind": "agent-skill", "name": name, "description": desc,
                "trigger_tokens": _tokens(desc), "path": str(p.relative_to(ROOT)),
                "wants_slash": True,
            })
    # Layer B — harness registry specs
    if REGISTRY.exists():
        for p in sorted(REGISTRY.glob("*.json")):
            data = json.loads(p.read_text(encoding="utf-8"))
            schema = str(data.get("schema", ""))
            if schema.startswith("sophia.") and "registry" in schema:
                continue  # index/forge bookkeeping, not a skill
            if "whenToUse" not in data:
                continue
            desc = str(data.get("whenToUse", ""))
            trig = set(t.lower() for t in data.get("triggers", []))
            skills.append({
                "kind": "registry-skill", "name": data.get("name", p.stem),
                "description": desc, "trigger_tokens": _tokens(desc) | trig,
                "explicit_triggers": trig,
                "path": str(p.relative_to(ROOT)), "wants_slash": False,
            })
    return skills


def _lint_one(sk: dict, shared: set[str]) -> tuple[list[str], list[str]]:
    """Validate one skill by KIND.

    Agent Skills (Layer A) are prose-only and seen via progressive disclosure, so
    their ``description`` must itself be long enough AND carry an action cue.
    Registry skills (Layer B) route on an explicit ``triggers`` array, so that
    array (not prose cues) is their trigger surface; ``whenToUse`` need only be a
    real sentence. Both kinds must be distinguishable from their siblings.
    """
    errors: list[str] = []
    warnings: list[str] = []
    desc = sk["description"]
    low = desc.lower()

    if sk["kind"] == "agent-skill":
        if len(desc) < MIN_DESC_CHARS:
            errors.append(f"description too short ({len(desc)}<{MIN_DESC_CHARS} chars)")
        if not any(cue in low for cue in ACTION_CUES):
            errors.append("no action/trigger cue (e.g. 'use when', 'run before', 'whenever')")
        if sk.get("wants_slash") and f"/{sk['name']}" not in desc:
            warnings.append(f"recommend naming the slash alias '/{sk['name']}' in the description")
    else:  # registry-skill: the triggers array is the trigger surface
        if len(desc) < 40:
            errors.append(f"whenToUse too short ({len(desc)}<40 chars)")
        if not sk.get("explicit_triggers"):
            errors.append("no 'triggers' array (the router's trigger surface) — add one")
        elif len(sk["explicit_triggers"]) < MIN_TRIGGER_TOKENS:
            errors.append(f"too few triggers ({len(sk['explicit_triggers'])}<{MIN_TRIGGER_TOKENS})")

    if len(sk["trigger_tokens"]) < MIN_TRIGGER_TOKENS:
        errors.append(f"too few distinctive trigger tokens ({len(sk['trigger_tokens'])}<{MIN_TRIGGER_TOKENS})")
    if sk["trigger_tokens"] and sk["trigger_tokens"] <= shared:
        errors.append("no token unique to this skill (indistinguishable from siblings)")
    return errors, warnings


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="(default behavior) exit nonzero on any thin description")
    ap.parse_args(argv)

    skills = _collect()
    if not skills:
        print("SKILL-DESC LINTER: no skills found (locked repo?) — nothing to lint.")
        return 0

    # A token is "shared" if it appears in 2+ skills; distinctiveness needs a non-shared token.
    counts: dict[str, int] = {}
    for sk in skills:
        for t in sk["trigger_tokens"]:
            counts[t] = counts.get(t, 0) + 1
    shared = {t for t, c in counts.items() if c >= 2}

    failures = 0
    warned = 0
    for sk in sorted(skills, key=lambda s: (s["kind"], s["name"])):
        errors, warnings = _lint_one(sk, shared)
        for w in warnings:
            warned += 1
            print(f"  warn  {sk['kind']}:{sk['name']}: {w}")
        if errors:
            failures += 1
            for e in errors:
                print(f"  FAIL  {sk['kind']}:{sk['name']} ({sk['path']}): {e}")

    n = len(skills)
    if failures:
        print(f"SKILL-DESC LINTER: {failures}/{n} skill(s) have thin trigger surfaces. "
              f"Sharpen the description (see .claude/skills/git-discipline for the gold standard).")
        return 1
    print(f"SKILL-DESC LINTER: OK — {n} skill(s) have trigger-rich descriptions "
          f"({warned} non-fatal warning(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
