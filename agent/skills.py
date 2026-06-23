"""Reusable skill registry for the Sophia agent harness.

A skill is a data file (skills/registry/*.json) describing WHEN to use it, the
TOOLS it needs, a step-by-step WORKFLOW, an IO SCHEMA, a VERIFICATION method,
COMMON FAILURES, and EXAMPLES. The harness injects the selected skill's workflow
and verification into the planner/executor prompts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agent.config import ROOT

SKILL_DIR = ROOT / "skills" / "registry"

REQUIRED_FIELDS = (
    "name",
    "whenToUse",
    "requiredTools",
    "workflow",
    "ioSchema",
    "verification",
    "commonFailures",
    "examples",
)


def validate_skill(skill: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in skill:
            errors.append(f"missing field: {field}")
    for list_field in ("requiredTools", "workflow", "verification", "commonFailures", "examples"):
        if list_field in skill and not isinstance(skill[list_field], list):
            errors.append(f"{list_field} must be a list")
    if "workflow" in skill and isinstance(skill["workflow"], list) and not skill["workflow"]:
        errors.append("workflow must be non-empty")
    if "ioSchema" in skill and not isinstance(skill["ioSchema"], dict):
        errors.append("ioSchema must be an object")
    return errors


def load_all(skill_dir: Path = SKILL_DIR) -> dict[str, dict[str, Any]]:
    skills: dict[str, dict[str, Any]] = {}
    if not skill_dir.exists():
        return skills
    for path in sorted(skill_dir.glob("*.json")):
        skill = json.loads(path.read_text(encoding="utf-8"))
        # ``skills/registry`` now also holds Skill Forge bookkeeping such as
        # ``forge_index.json``. Those files are registries/manifests, not
        # executable Agent Harness skill specs, so do not try to validate/load
        # them as skills. This keeps the registry extensible without breaking
        # older harness code that globbed every JSON file.
        if _is_metadata_file(skill):
            continue
        errors = validate_skill(skill)
        if errors:
            raise ValueError(f"invalid skill {path.name}: {errors}")
        skills[skill["name"]] = skill
    return skills


def _is_metadata_file(data: dict[str, Any]) -> bool:
    schema = str(data.get("schema", ""))
    return bool(schema and schema.startswith("sophia.") and "registry" in schema)


def get(name: str, skill_dir: Path = SKILL_DIR) -> dict[str, Any] | None:
    return load_all(skill_dir).get(name)


def list_skills(skill_dir: Path = SKILL_DIR) -> list[dict[str, str]]:
    return [{"name": s["name"], "whenToUse": s["whenToUse"]} for s in load_all(skill_dir).values()]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def select(goal: str, skill_dir: Path = SKILL_DIR, *, min_score: int = 1) -> dict[str, Any] | None:
    """Pick the best-matching skill for a goal, or None.

    Scores each skill by overlap between the goal and the skill's triggers /
    name / whenToUse text.
    """
    goal_tokens = _tokens(goal)
    best: dict[str, Any] | None = None
    best_score = 0
    for skill in load_all(skill_dir).values():
        triggers = set(t.lower() for t in skill.get("triggers", []))
        name_tokens = _tokens(skill["name"].replace("-", " "))
        when_tokens = _tokens(skill.get("whenToUse", ""))
        score = 0
        for token in goal_tokens:
            if token in triggers:
                score += 3
            elif token in name_tokens:
                score += 2
            elif token in when_tokens:
                score += 1
        # multiword trigger phrases
        lowered_goal = goal.lower()
        for trigger in triggers:
            if " " in trigger and trigger in lowered_goal:
                score += 3
        if score > best_score:
            best, best_score = skill, score
    return best if best_score >= min_score else None
