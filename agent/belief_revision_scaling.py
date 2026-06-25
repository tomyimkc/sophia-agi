# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Belief-revision scaling stress-test — honor the limitations ledger honestly.

The failure ledger flags that minimal-change belief revision (AGM) is hard and classic
Truth-Maintenance Systems did not scale. Rather than hand-wave, this measures it: build a
synthetic OKF graph of size N with periodic contradictions, run the revise-or-abstain
policy, and record conflicts resolved + wall-time, sweeping N to expose the real cost curve
(and where it would break). Correctness (the right beliefs are retracted) is checked
separately and deterministically; timing is reported, not asserted.

    from agent.belief_revision_scaling import scaling_sweep
    scaling_sweep([100, 500, 1000])   # [{n, conflicts, retracted, seconds}, ...]
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from agent.belief_revision_policy import resolve_conflicts
from okf.page import Page


def make_pages(n: int, *, contradiction_every: int = 10) -> "list[Page]":
    """N self-grounded sourced facts; every ``contradiction_every``-th is a higher-tier
    (axiom) claim that contradicts its predecessor — so the predecessor must be retracted."""
    pages: list[Page] = []
    for i in range(n):
        meta: dict = {"id": f"f{i}", "pageType": "concept", "authorConfidence": "attributed"}
        if i > 0 and i % contradiction_every == 0:
            meta["contradicts"] = [f"f{i - 1}"]
            meta["beliefTier"] = "axiom"   # the newer, higher-tier belief wins
        pages.append(Page(path=Path(f"f{i}.md"), meta=meta))
    return pages


def measure(n: int, *, contradiction_every: int = 10) -> "dict[str, Any]":
    pages = make_pages(n, contradiction_every=contradiction_every)
    start = time.perf_counter()
    report = resolve_conflicts(pages)
    elapsed = time.perf_counter() - start
    return {
        "n": n,
        "conflicts": report["conflictCount"],
        "retracted": len(report["retracted"]),
        "seconds": round(elapsed, 4),
    }


def scaling_sweep(sizes, *, contradiction_every: int = 10) -> "list[dict]":
    """Run ``measure`` across graph sizes to expose the revision cost curve."""
    return [measure(n, contradiction_every=contradiction_every) for n in sizes]


__all__ = ["make_pages", "measure", "scaling_sweep"]
