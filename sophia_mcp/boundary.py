# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Caller identity + kill-switch surface for the served MCP boundary.

MCP/stdio has **no authenticated caller on the wire** — every tool call arrives
anonymous, and the call *arguments* are attacker-controllable. So identity must be
provisioned out-of-band (environment), never read from tool args, or the gateway's
authz / BLP no-read-up checks are theatre an attacker can forge.

  - ``caller_identity()`` → (role, clearance) from ``SOPHIA_MCP_ROLE`` /
    ``SOPHIA_MCP_CLEARANCE``; defaults to anonymous + UNCLASSIFIED.
  - ``kill_switch_engaged()`` → an operator surface to halt all gated calls via
    ``SOPHIA_MCP_KILL_SWITCH`` (env) or a sentinel file, before any side effect.
  - ``gateway_enabled()`` → whether served write/external tools route through the
    fail-closed gateway (opt-in: ``SOPHIA_MCP_GATEWAY=1``; default off, so the
    server's existing behavior is unchanged unless explicitly enabled).
"""

from __future__ import annotations

import os
from pathlib import Path

from sophia_contract import blp

DEFAULT_CLEARANCE = "UNCLASSIFIED"
_TRUTHY = {"1", "true", "yes", "on"}


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def caller_identity() -> "tuple[str | None, str]":
    """Resolve (role, clearance) for a served call from the environment only.

    Tool ARGS never confer identity (they are untrusted). An unset/invalid
    clearance falls back to the least-privileged level, fail-closed.
    """
    role = os.environ.get("SOPHIA_MCP_ROLE", "").strip() or None
    clearance = os.environ.get("SOPHIA_MCP_CLEARANCE", "").strip() or DEFAULT_CLEARANCE
    if not blp.is_level(clearance):
        clearance = DEFAULT_CLEARANCE
    return role, clearance


def kill_switch_engaged() -> bool:
    """True if a human has engaged the operator kill switch (env or sentinel file).

    This is the *served-surface* switch, checked before dispatch so no side effect
    runs while engaged. It is independent of (and additional to) the contract's own
    programmatic ``engage_kill_switch``.
    """
    if _flag("SOPHIA_MCP_KILL_SWITCH"):
        return True
    sentinel = os.environ.get("SOPHIA_MCP_KILL_SWITCH_FILE", "").strip()
    if sentinel and Path(sentinel).exists():
        return True
    return False


def gateway_enabled() -> bool:
    """Whether served write/external tools route through the fail-closed gateway."""
    return _flag("SOPHIA_MCP_GATEWAY")
