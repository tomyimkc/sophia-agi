# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Audit + permission substrate for MCP tools.

Every tool call is appended to an audit log; mutating/high-risk tools require an
explicit approval env flag. This is the safety floor that must sit under any
write/exec tool. The audit log lives under agent/memory/ (gitignored).
"""

from __future__ import annotations

import functools
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
AUDIT_LOG = ROOT / "agent" / "memory" / "mcp_audit.jsonl"

# default risk per tool; anything not listed is treated as "low" (read-only)
TOOL_RISK = {
    "sophia_export_corpus": "medium",  # writes training/corpus.jsonl
    "sophia_wiki_upsert": "medium",    # writes an agent-owned OKF wiki page
}

APPROVE_ENV = "SOPHIA_MCP_APPROVE_WRITES"  # set to "1" to allow medium/high tools


def _summarize(value: Any, limit: int = 200) -> Any:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text[:limit] + ("…" if len(text) > limit else "")


def audit_log(entry: dict[str, Any], *, path: Path = AUDIT_LOG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.now(timezone.utc).isoformat(), **entry}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def check_permission(tool: str, *, approved: bool | None = None) -> tuple[bool, str | None]:
    risk = TOOL_RISK.get(tool, "low")
    if risk == "low":
        return True, None
    allowed = approved if approved is not None else os.environ.get(APPROVE_ENV) == "1"
    if allowed:
        return True, None
    return False, f"tool '{tool}' is risk={risk}; set {APPROVE_ENV}=1 to allow"


def audited(tool: str, *, risk: str | None = None, audit_path: Path = AUDIT_LOG) -> Callable:
    """Decorator: enforce permission, run, and append an audit record.

    Wraps an impl returning a dict; on permission denial returns a structured
    error dict instead of executing.
    """
    if risk is not None:
        TOOL_RISK[tool] = risk

    def decorator(func: Callable[..., dict]) -> Callable[..., dict]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> dict:
            ok, error = check_permission(tool)
            if not ok:
                audit_log({"tool": tool, "ok": False, "denied": True, "error": error, "args": _summarize(args)}, path=audit_path)
                return {"error": error, "denied": True}
            try:
                result = func(*args, **kwargs)
            except Exception as exc:  # structured error, still audited
                audit_log({"tool": tool, "ok": False, "error": repr(exc), "args": _summarize(args)}, path=audit_path)
                return {"error": repr(exc)}
            audit_log({
                "tool": tool,
                "ok": not (isinstance(result, dict) and result.get("error")),
                "risk": TOOL_RISK.get(tool, "low"),
                "args": _summarize(args),
                "resultSummary": _summarize(result),
            }, path=audit_path)
            return result

        return wrapper

    return decorator
