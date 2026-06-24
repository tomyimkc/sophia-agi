# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Graded forgetting curve — graceful, measurable confidence decay over disuse.

Catastrophic forgetting is abrupt and silent; human memory fades *gracefully* and
*predictably*. This module gives the belief graph a forgetting curve: a fact's effective
confidence decays with time since it was last reinforced (queried / re-sourced), following
a spaced-repetition retention model — and each reinforcement raises its stability, so
often-used facts decay slowly and unused ones fade. A faded fact is not deleted (it stays
retrievable in the graph); it simply drops below the assertion threshold, so the gate stops
*confidently* asserting it. That is graceful forgetting, and it is auditable.

Pure, deterministic math — elapsed time is passed in (no wall-clock), so it is testable and
reproducible.

    from agent.forgetting_curve import retention, decayed_confidence, faded
    retention(days_since=90, reinforcements=0)   # low — long unused, never reinforced
    retention(days_since=90, reinforcements=5)   # high — reinforcement stabilized it
"""

from __future__ import annotations

import math
from typing import Any


def stability(reinforcements: int, *, base_days: float = 30.0, per_reinforcement: float = 30.0) -> float:
    """Memory stability (days) — grows with each reinforcement, so used facts decay slower."""
    return base_days + per_reinforcement * max(0, int(reinforcements))


def retention(days_since: float, reinforcements: int = 0, **kw) -> float:
    """Spaced-repetition retention in [0, 1]: exp(-Δt / stability)."""
    s = stability(reinforcements, **kw)
    return math.exp(-max(0.0, float(days_since)) / s)


def decayed_confidence(base_rank: float, days_since: float, reinforcements: int = 0, **kw) -> float:
    """A fact's confidence rank scaled by its current retention (graceful decay)."""
    return round(float(base_rank) * retention(days_since, reinforcements, **kw), 4)


def forgetting_report(facts, *, assert_threshold: float = 1.0, **kw) -> "dict[str, Any]":
    """Decay a set of facts and report which have *faded* below the assertion threshold.

    ``facts``: iterable of {id, rank, daysSince, reinforcements}. A faded fact stays in the
    graph (retrievable) but the gate should no longer *assert* it — graceful, not catastrophic.
    """
    rows = []
    faded_ids = []
    for f in facts:
        d = decayed_confidence(f["rank"], f["daysSince"], f.get("reinforcements", 0), **kw)
        is_faded = d < assert_threshold
        rows.append({"id": f["id"], "baseRank": f["rank"], "daysSince": f["daysSince"],
                     "reinforcements": f.get("reinforcements", 0), "decayedRank": d, "faded": is_faded})
        if is_faded:
            faded_ids.append(f["id"])
    rows.sort(key=lambda r: r["decayedRank"])
    return {
        "schema": "sophia.forgetting_curve.v1",
        "assertThreshold": assert_threshold,
        "total": len(rows),
        "fadedCount": len(faded_ids),
        "faded": faded_ids,
        "rows": rows,
    }


def faded(facts, *, assert_threshold: float = 1.0, **kw) -> "list[str]":
    """Ids whose decayed confidence has fallen below the assertion threshold."""
    return forgetting_report(facts, assert_threshold=assert_threshold, **kw)["faded"]


__all__ = ["stability", "retention", "decayed_confidence", "forgetting_report", "faded"]
