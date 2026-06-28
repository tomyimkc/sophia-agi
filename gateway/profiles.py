# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deployment profiles for the gateway.

``hardened_gateway()`` returns a Gateway configured for an exposed/production
deployment: egress output-guard ON, a finite call budget (anti-DoS / unbounded
consumption, OWASP LLM10), and the acceptable-use refusal screen advertised to
callers. The default ``Gateway()`` stays permissive for local/offline tests; this
is the opinionated profile you deploy behind.

Pair with ``sophia_mcp`` approval gates (``SOPHIA_MCP_APPROVAL=1``) for
side-effecting tools — that lives in the MCP server, not here.
"""

from __future__ import annotations

import os

from gateway.interceptor import Gateway

# Conservative default ceiling for a single session; override via env.
DEFAULT_CALL_BUDGET = int(os.environ.get("SOPHIA_GATEWAY_CALL_BUDGET", "200"))


# Default tamper-evident audit trail location for a hardened deployment.
DEFAULT_AUDIT_LOG = os.environ.get("SOPHIA_GATEWAY_AUDIT_LOG", "agent/memory/gateway_audit.jsonl")


def hardened_gateway(*, system_prompt: "str | None" = None,
                     canaries: "list[str] | None" = None,
                     call_budget: "int | None" = None,
                     audit_log: "str | None" = DEFAULT_AUDIT_LOG, **kwargs) -> Gateway:
    """Build a production-hardened Gateway (fail-closed + leak guard + budget +
    hash-chained audit trail).

    ``system_prompt`` enables verbatim-echo detection; ``canaries`` (the minted
    private canary set) enables confirmed-leak blocking; ``audit_log`` enables the
    tamper-evident trail (pass ``None`` to disable). All optional.
    """
    return Gateway(
        output_guard=True,
        system_prompt=system_prompt,
        canaries=canaries,
        call_budget=DEFAULT_CALL_BUDGET if call_budget is None else call_budget,
        audit_log=audit_log,
        **kwargs,
    )


def enforce_acceptable_use(context: "dict | None" = None) -> dict:
    """Return a conscience-context dict with the acceptable-use refusal screen ON.

    Use at the input boundary:
        decision = conscience_check(user_text, context=enforce_acceptable_use())
    """
    ctx = dict(context or {})
    ctx["enforceAcceptableUse"] = True
    return ctx


__all__ = ["hardened_gateway", "enforce_acceptable_use", "DEFAULT_CALL_BUDGET",
           "DEFAULT_AUDIT_LOG"]
