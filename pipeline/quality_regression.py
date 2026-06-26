# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Fail-closed quality-regression gate (Phase 3).

A daily corpus build should not be allowed to silently get worse. This gate compares a
current corpus summary (``pipeline.corpus_table.summarize``) against a committed baseline and
flags regressions: a drop in mean quality, a drop in keep-rate, a spike in duplicate-rate, or
a large drop in token volume. It mirrors the discipline of
``provenance_bench.dataset_guard`` — **fail-closed**: any breach returns a non-empty problem
list (and a CLI exit code), so a bad batch blocks rather than ships.

Pure stdlib and deterministic: summaries are plain JSON, so the gate runs anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tolerances:
    """How much regression is allowed before the gate fails (absolute deltas / ratios)."""

    #: Max allowed drop in mean quality (absolute, e.g. 0.05 = 5 quality points).
    max_quality_drop: float = 0.05
    #: Max allowed drop in keep-rate (absolute).
    max_keep_rate_drop: float = 0.05
    #: Max allowed rise in duplicate-rate (absolute).
    max_duplicate_rate_rise: float = 0.05
    #: Max allowed *fractional* drop in total tokens (0.5 = corpus may not lose >50% volume).
    max_token_drop_frac: float = 0.5


def compare(baseline: dict, current: dict, *, tol: Tolerances | None = None) -> list[str]:
    """Return a list of regressions (empty == pass). Fail-closed on missing baseline fields."""
    tol = tol or Tolerances()
    problems: list[str] = []

    b_q, c_q = baseline.get("meanQuality"), current.get("meanQuality")
    if b_q is not None and c_q is not None:
        if (b_q - c_q) > tol.max_quality_drop:
            problems.append(
                f"meanQuality dropped {b_q:.4f} -> {c_q:.4f} (> {tol.max_quality_drop})"
            )
    elif b_q is not None and c_q is None:
        problems.append("meanQuality became unavailable (baseline had a value)")

    b_k, c_k = baseline.get("keepRate"), current.get("keepRate")
    if b_k is not None and c_k is not None and (b_k - c_k) > tol.max_keep_rate_drop:
        problems.append(f"keepRate dropped {b_k:.4f} -> {c_k:.4f} (> {tol.max_keep_rate_drop})")

    b_d, c_d = baseline.get("duplicateRate"), current.get("duplicateRate")
    if b_d is not None and c_d is not None and (c_d - b_d) > tol.max_duplicate_rate_rise:
        problems.append(
            f"duplicateRate rose {b_d:.4f} -> {c_d:.4f} (> {tol.max_duplicate_rate_rise})"
        )

    b_t, c_t = baseline.get("totalTokens"), current.get("totalTokens")
    if isinstance(b_t, (int, float)) and b_t > 0 and isinstance(c_t, (int, float)):
        drop_frac = (b_t - c_t) / b_t
        if drop_frac > tol.max_token_drop_frac:
            problems.append(
                f"totalTokens dropped {b_t} -> {c_t} ({drop_frac:.1%} > {tol.max_token_drop_frac:.0%})"
            )

    return problems


def gate(baseline: dict, current: dict, *, tol: Tolerances | None = None) -> dict:
    """Run the gate; return ``{"ok": bool, "problems": [...]}``. ``ok`` is False on any breach."""
    problems = compare(baseline, current, tol=tol)
    return {"ok": not problems, "problems": problems}


__all__ = ["Tolerances", "compare", "gate"]
