"""Sophia Skills — core registry + the ``@sophia_skill`` decorator.

A *skill* is a thin, friendly, **fail-closed** wrapper over one or more Sophia MCP
tools. The decorator:

- registers the skill by name (auto-registration when its module is imported);
- guarantees every skill returns a plain ``dict`` and **never raises** into the
  caller — on any error it returns ``{"ok": False, "verdict": "held",
  "failClosed": True, ...}`` (abstain, don't fabricate);
- records lightweight provenance (which MCP tools the skill is allowed to use).

This module has no third-party dependencies and is import-safe in CI.
"""
from __future__ import annotations

import functools
from typing import Any, Callable

# name -> wrapped skill callable
SKILLS: dict[str, Callable[..., dict]] = {}
# name -> {summary, uses}
_META: dict[str, dict] = {}


def sophia_skill(
    name: str | None = None,
    *,
    summary: str = "",
    uses: tuple[str, ...] = (),
) -> Callable[[Callable[..., dict]], Callable[..., dict]]:
    """Register a fail-closed skill.

    ``uses`` lists the MCP tools the skill is expected to call (provenance/audit).
    """

    def deco(fn: Callable[..., dict]) -> Callable[..., dict]:
        skill_name = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(**kwargs: Any) -> dict:
            try:
                out = fn(**kwargs)
                if not isinstance(out, dict):
                    out = {"value": out}
                out.setdefault("verdict", "ok")
                return {"skill": skill_name, "ok": True, **out}
            except Exception as e:  # fail-closed: abstain, never propagate
                return {
                    "skill": skill_name,
                    "ok": False,
                    "verdict": "held",
                    "error": f"{type(e).__name__}: {e}",
                    "failClosed": True,
                }

        wrapper.sophia_skill = skill_name  # type: ignore[attr-defined]
        SKILLS[skill_name] = wrapper
        _META[skill_name] = {"summary": summary, "uses": list(uses)}
        return wrapper

    return deco


def list_skills() -> dict:
    """Return ``{name: {summary, uses}}`` for every registered skill (sorted)."""
    return {n: _META[n] for n in sorted(SKILLS)}


def run_skill(name: str, /, **kwargs: Any) -> dict:
    """Invoke a skill by name. Unknown names fail closed (held)."""
    fn = SKILLS.get(name)
    if fn is None:
        return {
            "skill": name,
            "ok": False,
            "verdict": "held",
            "error": f"unknown skill '{name}'",
            "failClosed": True,
            "available": sorted(SKILLS),
        }
    return fn(**kwargs)
