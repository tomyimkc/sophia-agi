# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Approval gate — the active, blocking sibling of the (observe-only) Sentinel.

Distilled from AgentArk's Sentinel approval gates. When enabled, a high-risk
governed tool call is **not dispatched**; instead it is enqueued for human
review and the caller gets a fail-closed ``approval_required`` hold. This is the
same posture the gateway already uses for kill-switch / BLP holds.

Opt-in and default-off: the gate is inert unless ``SOPHIA_MCP_APPROVAL=1`` (so
the served surface is byte-identical by default). When on, only tools whose name
is in :data:`REQUIRES_APPROVAL` are held; everything else passes through. The
queue is an append-only JSONL under ``agent/memory/`` (gitignored), storing the
action *class* and an arg digest — never raw secret-bearing payloads.

Nothing here executes the tool. Enqueue → hold. A reviewer drains the queue
out-of-band; this module only records the request and reports the hold.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPROVAL_QUEUE = ROOT / "agent" / "memory" / "approval_queue.jsonl"

APPROVE_ENV = "SOPHIA_MCP_APPROVAL"  # set to "1" to ACTIVATE the gate (default off)

# Tools that require human approval when the gate is active. These are the
# side-effecting / external-egress tools the gateway already governs.
REQUIRES_APPROVAL = (
    "sophia_wiki_upsert",
    "sophia_export_corpus",
    "sophia_openclaw_infer",
)


def approval_enabled() -> bool:
    """The gate is inert unless explicitly activated (default off)."""
    return os.environ.get(APPROVE_ENV) == "1"


def requires_approval(tool_id: str) -> bool:
    """True iff the gate is active AND this tool is on the approval list."""
    return approval_enabled() and tool_id in REQUIRES_APPROVAL


def _arg_digest(args: dict) -> str:
    """A stable, non-reversible digest of the call args (no raw payload stored)."""
    try:
        blob = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        blob = str(args)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def enqueue(tool_id: str, args: dict, *, role: str | None = None,
            queue_path: Path | None = None) -> dict:
    """Append an approval request and return a fail-closed ``approval_required``
    hold (the gateway's ``_held`` shape). Stores only the action class + arg
    digest + key names — never raw arg values (secret-safety).
    """
    path = queue_path or APPROVAL_QUEUE
    digest = _arg_digest(args)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "tool": tool_id,
        "role": role,
        "argDigest": digest,
        "argKeys": sorted(args.keys()) if isinstance(args, dict) else [],
        "status": "pending",
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Even if we cannot persist the request, we still HOLD (fail-closed):
        # the worst case is a held call with no queue entry, never a silent dispatch.
        pass
    return {
        "result": None,
        "held_reason": "approval_required",
        "reasons": [f"tool '{tool_id}' requires human approval (queued: {digest})"],
        "suggested_fix": f"a reviewer must approve request {digest} in the approval queue",
        "_governance": {"verdict": "held", "held_reason": "approval_required",
                        "argDigest": digest},
    }


def pending(queue_path: Path | None = None) -> "list[dict]":
    """Return the pending approval requests (for a reviewer / report). Fail-open read."""
    path = queue_path or APPROVAL_QUEUE
    out: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") == "pending":
                out.append(rec)
    except FileNotFoundError:
        return out
    except Exception:
        return out
    return out
