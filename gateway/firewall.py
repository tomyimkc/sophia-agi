# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""MCP firewall (P1) — tool descriptions and outputs are untrusted input.

Two defenses, both fail-closed:
  - injection scan: tool *descriptions* (scanned at registration) and call *arguments*
    (scanned per call) are screened for prompt-injection / exfiltration markers; a hit
    quarantines the tool (risk_tier=high) or refuses the call.
  - taint: a tool's OUTPUT is low-integrity (Biba) until verified; the gateway tags it so
    it cannot be laundered into a high-integrity sink without passing the verify router.

Deterministic, dependency-free. Reuses the spirit of agent/dataflow without importing it,
so the gateway stays standalone.
"""

from __future__ import annotations

import re

# Markers of an attempt to subvert the agent via tool text/args (a documented MCP attack
# surface). Conservative + case-insensitive; extend as the red-team grows.
INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|prior|above) (instructions|prompt)",
    r"disregard (all |the )?(previous|prior|system)",
    r"you are now",
    r"new instructions:",
    r"system prompt",
    r"reveal (your |the )?(system )?(prompt|instructions|secrets?)",
    r"exfiltrat",
    r"send (the |all )?(secrets?|api[ _]?keys?|credentials?|env)",
    r"curl\s+http",
    r"<\s*script",
    r"base64\s*\(",
]
_RX = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def scan_text(text: str) -> "list[str]":
    """Return the injection patterns that match ``text`` (empty == clean)."""
    t = text or ""
    return [rx.pattern for rx in _RX if rx.search(t)]


def scan_args(args: dict) -> "list[str]":
    hits: list = []
    for v in (args or {}).values():
        if isinstance(v, str):
            hits += scan_text(v)
        elif isinstance(v, (list, tuple)):
            for x in v:
                if isinstance(x, str):
                    hits += scan_text(x)
    return sorted(set(hits))


def scan_tool_description(description: str) -> dict:
    hits = scan_text(description)
    return {"clean": not hits, "hits": hits}


def taint_label(verifier_ref: str, side_effects: str) -> str:
    """Integrity of a tool's output before verification. Unverified or external output
    is UNTRUSTED (no-write-up); a real verifier raises it to VERIFIED downstream."""
    if verifier_ref == "none" or side_effects == "external":
        return "untrusted"
    return "checked"
