# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Canary tokens for leak detection.

A canary is a unique, meaningless string embedded in the *private* system prompt
and/or the *private* training mixture. It must NEVER appear in a public artifact.
If a canary later shows up in:

  - model output  → the model is leaking its system prompt / memorized training
    data (OWASP LLM07 / LLM02);
  - the public web or the public repo → the private recipe has been exfiltrated.

Canaries are deterministic given a seed so they can be regenerated and watched,
but they are derived from a SECRET seed (env ``SOPHIA_CANARY_SEED``) so an
attacker cannot predict them. Keep the seed in ``.env`` (gitignored), never here.
"""

from __future__ import annotations

import hashlib
import os
import re

CANARY_PREFIX = "SOPHIA-CANARY"
# Matches any canary token regardless of seed, for the output/web monitor.
CANARY_RX = re.compile(rf"{CANARY_PREFIX}-[0-9a-f]{{16}}", re.IGNORECASE)


def make_canary(label: str, *, seed: str | None = None) -> str:
    """Return a deterministic canary token for ``label`` under the secret seed.

    ``label`` distinguishes call sites (e.g. ``"system_prompt"``, ``"train_mix"``)
    so a leak tells you *which* surface lost containment.
    """
    seed = seed if seed is not None else os.environ.get("SOPHIA_CANARY_SEED", "")
    if not seed:
        raise RuntimeError(
            "SOPHIA_CANARY_SEED is not set; refusing to mint a guessable canary. "
            "Put a random secret in .env (gitignored)."
        )
    digest = hashlib.sha256(f"{seed}:{label}".encode()).hexdigest()[:16]
    return f"{CANARY_PREFIX}-{digest}"


def scan_for_canaries(text: str) -> "list[str]":
    """Return every canary-shaped token found in ``text`` (empty == clean)."""
    return sorted({m.group(0) for m in CANARY_RX.finditer(text or "")})


def contains_known_canary(text: str, known: "list[str]") -> "list[str]":
    """Return which of the caller's ``known`` canaries appear in ``text``.

    Use this in the output guard / web monitor when you have the exact set of
    minted canaries and want a confirmed (not just shaped) leak signal.
    """
    found = set(scan_for_canaries(text))
    return sorted(c for c in known if c in found)


__all__ = ["CANARY_PREFIX", "CANARY_RX", "make_canary", "scan_for_canaries", "contains_known_canary"]
