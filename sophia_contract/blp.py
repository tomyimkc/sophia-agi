# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Bell-LaPadula classification for the contract wire vocabulary.

The contract exposes four ordered levels — the stable enum aihk-os pins against:

    UNCLASSIFIED < CONFIDENTIAL < SECRET < TOP_SECRET

Two rules, both fail-closed and **never silently resolved by downgrading**:

  - **no write down** (record time): a derived claim must be classified at least as
    high as its most-classified parent. Writing high-provenance content into a
    lower-labelled claim leaks it — a ``BLP_VIOLATION``.
  - **no read up** (verify/publish time): a request may only act on a claim whose
    level its clearance dominates. Reading above clearance is a ``BLP_VIOLATION``.

(Internally agent.security.labels carries a richer lattice with a Biba integrity
axis; the contract deliberately publishes only this confidentiality enum so the
wire shape stays small and stable.)
"""

from __future__ import annotations

# Order is the contract; index = dominance rank. Do not reorder without a MAJOR bump.
BLP_LEVELS = ("UNCLASSIFIED", "CONFIDENTIAL", "SECRET", "TOP_SECRET")
_RANK = {name: i for i, name in enumerate(BLP_LEVELS)}


def is_level(value: str) -> bool:
    return value in _RANK


def rank(level: str) -> int:
    if level not in _RANK:
        raise ValueError(f"unknown blp_level {level!r}; valid: {BLP_LEVELS}")
    return _RANK[level]


def dominates(a: str, b: str) -> bool:
    """True when level ``a`` is at least as classified as ``b`` (a >= b)."""
    return rank(a) >= rank(b)


def write_down_violation(claim_level: str, parent_levels: "list[str]") -> "str | None":
    """no-write-down: a claim must dominate every parent. Returns a reason string on
    violation, else None."""
    for p in parent_levels:
        if not dominates(claim_level, p):
            return (f"no-write-down: claim labelled {claim_level} derives from a "
                    f"{p} parent (a derived claim must be at least as classified as "
                    f"its sources)")
    return None


def read_up_violation(clearance: str, claim_level: str) -> "str | None":
    """no-read-up: the caller's clearance must dominate the claim. Returns a reason
    string on violation, else None."""
    if not dominates(clearance, claim_level):
        return (f"no-read-up: clearance {clearance} may not act on a {claim_level} "
                f"claim")
    return None
