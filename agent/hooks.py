# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Lifecycle hook bus — deterministic, named steering events (Stage A).

Sophia already enforces its guarantees *inside* ``gateway/interceptor.py`` (the
fail-closed pipeline) and ``agent/dataflow/interpreter.py`` (the CaMeL-style
constrained executor). What was missing was a **named, reusable lifecycle
abstraction** so those guarantees can be (a) externally legible, (b) extended
without editing the interceptor body, and (c) snapshotted before context
compaction so the audit trail survives long-horizon runs.

This module is that abstraction. It is dependency-free, offline, and
deterministic — the steering discipline lives in code the harness runs, never in
a prompt instruction (the central lesson the design imports from the Claude Code
steering model: a real guardrail must be enforcement, not a "never do this"
string).

Events
------
  - ``SESSION_START``   : a session/run begins.
  - ``PRE_TOOL_USE``    : before a tool executes. A handler may **block**
                          (fail-closed) — the canonical guardrail point.
  - ``POST_TOOL_USE``   : after a tool executes / is verified. Observe only.
  - ``PRE_COMPACT``     : before context compaction. Handlers persist a durable
                          snapshot (belief-graph / provenance delta).
  - ``SESSION_END``     : a session/run ends.

Decision contract (fail-closed)
-------------------------------
A ``PRE_TOOL_USE`` handler returns a :class:`HookDecision` (or ``None`` = allow).
If *any* handler blocks, the action is blocked. A handler that *raises* is treated
as a block on ``PRE_TOOL_USE`` (fail-closed: a broken guardrail must not silently
open) but as a no-op on observe-only events (availability over a missed
annotation, matching the repo's existing "loud but non-fatal" audit convention).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class HookEvent(str, Enum):
    SESSION_START = "SessionStart"
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    PRE_COMPACT = "PreCompact"
    SESSION_END = "SessionEnd"


#: Events at which a handler is permitted to *block* the pending action.
BLOCKING_EVENTS = frozenset({HookEvent.PRE_TOOL_USE})


@dataclass(frozen=True)
class HookContext:
    """Immutable payload passed to every handler.

    ``event``    the lifecycle event being dispatched.
    ``tool_id``  the tool about to run / that ran (None for session/compact events).
    ``args``     the call arguments (read-only view; handlers must not mutate).
    ``payload``  free-form, event-specific extra data (verdict, snapshot target...).
    """

    event: "HookEvent"
    tool_id: "str | None" = None
    args: dict = field(default_factory=dict)
    payload: dict = field(default_factory=dict)


@dataclass(frozen=True)
class HookDecision:
    """A handler's verdict on a pending action.

    ``allow``   False blocks the action (fail-closed) on a blocking event.
    ``reason``  human/audit explanation, always set when blocking.
    ``source``  the handler name, for audit attribution.
    """

    allow: bool = True
    reason: str = ""
    source: str = ""


# A handler maps a context to a decision (or None == allow / observe-only).
Handler = Callable[["HookContext"], "HookDecision | None"]


@dataclass(frozen=True)
class DispatchResult:
    """Aggregate outcome of dispatching one event to all its handlers."""

    event: "HookEvent"
    allowed: bool
    decisions: tuple = ()      # tuple[HookDecision] from handlers that returned one
    blocked_by: "str | None" = None
    reason: str = ""

    @property
    def blocked(self) -> bool:
        return not self.allowed


class HookBus:
    """Register handlers per event and dispatch deterministically.

    Handlers run in registration order. The bus is fail-closed on blocking
    events: an exception in a ``PRE_TOOL_USE`` handler counts as a block, so a
    crashing guardrail can never let an action through.
    """

    def __init__(self) -> None:
        self._handlers: dict = {e: [] for e in HookEvent}

    def register(self, event: "HookEvent", handler: "Handler", *, name: "str | None" = None) -> "HookBus":
        """Register ``handler`` for ``event``. ``name`` is used for audit/blocked_by;
        defaults to the handler's ``__name__``. Returns self for chaining."""
        if event not in self._handlers:
            raise ValueError(f"unknown hook event {event!r}")
        handler_name = name or getattr(handler, "__name__", repr(handler))
        self._handlers[event].append((handler_name, handler))
        return self

    def handlers(self, event: "HookEvent") -> "list[str]":
        return [n for n, _ in self._handlers.get(event, [])]

    def dispatch(self, ctx: "HookContext") -> "DispatchResult":
        """Run every handler for ``ctx.event``.

        On a blocking event the first handler that returns ``allow=False`` (or
        raises) short-circuits and blocks. On an observe-only event every handler
        runs and exceptions are swallowed (loud-but-non-fatal), never blocking.
        """
        event = ctx.event
        is_blocking = event in BLOCKING_EVENTS
        decisions: list = []
        for name, handler in self._handlers.get(event, []):
            try:
                decision = handler(ctx)
            except Exception as exc:  # noqa: BLE001 - intentional fail-closed/observe split
                if is_blocking:
                    return DispatchResult(
                        event=event, allowed=False,
                        decisions=tuple(decisions),
                        blocked_by=name,
                        reason=f"handler {name!r} raised (fail-closed): {exc!r}",
                    )
                continue  # observe-only: a broken annotator must not stall the run
            if decision is None:
                continue
            if isinstance(decision, bool):  # tolerate a bare bool for ergonomics
                decision = HookDecision(allow=decision, source=name)
            if not decision.source:
                decision = HookDecision(allow=decision.allow, reason=decision.reason, source=name)
            decisions.append(decision)
            if is_blocking and not decision.allow:
                return DispatchResult(
                    event=event, allowed=False,
                    decisions=tuple(decisions),
                    blocked_by=decision.source,
                    reason=decision.reason or f"blocked by {decision.source}",
                )
        return DispatchResult(event=event, allowed=True, decisions=tuple(decisions))


# ----------------------------------------------------------------------------- #
# Ready-made handlers for the common Sophia guardrails.
# ----------------------------------------------------------------------------- #
def make_provenance_pretool_guard(*, require_provenance: bool = True) -> "Handler":
    """A ``PRE_TOOL_USE`` guard that fails closed when a side-effecting call lacks
    declared provenance/clearance.

    This expresses, as a *named hook*, the same discipline the interceptor already
    enforces inline: a write/external tool must carry a clearance and the caller
    must not be anonymous. It only *adds* a fail-closed check; it never relaxes
    the interceptor.
    """

    def _guard(ctx: "HookContext") -> "HookDecision | None":
        if not require_provenance:
            return None
        side_effects = ctx.payload.get("side_effects", "read")
        if side_effects in ("write", "external"):
            clearance = ctx.payload.get("clearance")
            role = ctx.payload.get("role")
            if not clearance or not role:
                return HookDecision(
                    allow=False,
                    reason=(f"fail-closed: side-effecting tool {ctx.tool_id!r} "
                            f"requires both role and clearance "
                            f"(role={role!r}, clearance={clearance!r})"),
                )
        return None

    return _guard


def make_precompact_snapshot(sink: "Callable[[dict], Any]") -> "Handler":
    """A ``PRE_COMPACT`` handler that hands the snapshot payload to ``sink`` (e.g.
    append-to-jsonl) so the belief-graph / provenance delta survives compaction.

    ``sink`` receives the ``ctx.payload`` dict (plus the event name). Exceptions
    are swallowed by the bus on this observe-only event, so a failing sink cannot
    stall a run — but should itself log loudly per repo convention.
    """

    def _snapshot(ctx: "HookContext") -> None:
        sink({"event": ctx.event.value, **ctx.payload})
        return None

    return _snapshot


def make_conscience_pretool_guard(*, default_high_impact: bool = True) -> "Handler":
    """A PRE_TOOL_USE guard backed by :mod:`agent.conscience_enforcement`.

    Imported lazily to avoid a hard dependency cycle. High-impact tool calls with
    conscience verdict block/abstain/escalate/retrieve/clarify/revise are blocked
    fail-closed unless the caller explicitly marks them low-impact.
    """
    from agent.conscience_enforcement import make_conscience_pretool_guard as _mk
    return _mk(default_high_impact=default_high_impact)


__all__ = [
    "HookEvent",
    "BLOCKING_EVENTS",
    "HookContext",
    "HookDecision",
    "HookBus",
    "DispatchResult",
    "Handler",
    "make_provenance_pretool_guard",
    "make_precompact_snapshot",
    "make_conscience_pretool_guard",
]
