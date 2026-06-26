# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Turn an eval-ladder report into a single search objective — fail-closed.

The RunPod LoRA pipeline emits ``eval_ladder_adapter.json`` (schema ``sophia.eval_ladder.v2``)
with four rungs — base · base+gate · adapter · adapter+gate — each carrying per-suite and
channel (format/content/combined) pass rates. The autonomous search needs ONE number to
minimize. The honest objective is the **uplift the trained adapter+gate delivers over the
untrained base** on the combined channel:

    uplift = score(adapter+gate) - score(base)

We return ``objective_for_min = -uplift`` (the loop minimizes), plus the raw rung scores and
the base→base+gate→adapter→adapter+gate progression for auditing. Fail-closed: if any rung is
missing or malformed, we return ``ok=False`` with ``objective_for_min=+inf`` so the search
treats it as a failed trial rather than inventing a score.
"""
from __future__ import annotations

from typing import Any

_RUNGS = ("base", "base+gate", "adapter", "adapter+gate")


def _rung_score(rung: dict, channel: str) -> float | None:
    """Combined/content/overall score_pct for a rung, or None if absent."""
    summary = rung.get("summary")
    if not isinstance(summary, dict):
        return None
    if channel in ("format", "content", "combined"):
        ch = summary.get("channels", {}).get(channel, {})
        if isinstance(ch, dict) and "score_pct" in ch:
            return float(ch["score_pct"])
    # fall back to the rung's headline score_pct
    if "score_pct" in summary:
        return float(summary["score_pct"])
    return None


def parse_objective(report: dict, *, channel: str = "combined") -> "dict[str, Any]":
    """Extract the search objective from an eval-ladder report. Fail-closed."""
    if not isinstance(report, dict) or report.get("schema") != "sophia.eval_ladder.v2":
        return {"ok": False, "reason": "not a sophia.eval_ladder.v2 report",
                "objective_for_min": float("inf")}
    rungs = {r.get("rung"): r for r in report.get("rungs", []) if isinstance(r, dict)}
    missing = [name for name in _RUNGS if name not in rungs]
    scores = {name: _rung_score(rungs[name], channel) for name in _RUNGS if name in rungs}
    if "base" not in scores or "adapter+gate" not in scores \
            or scores["base"] is None or scores["adapter+gate"] is None:
        return {"ok": False, "reason": f"missing/blank rungs: {missing or 'base/adapter+gate'}",
                "rung_scores": scores, "objective_for_min": float("inf")}

    uplift = round(scores["adapter+gate"] - scores["base"], 4)
    gate_only = (round(scores["base+gate"] - scores["base"], 4)
                 if scores.get("base+gate") is not None else None)
    adapter_only = (round(scores["adapter"] - scores["base"], 4)
                    if scores.get("adapter") is not None else None)
    return {
        "ok": True,
        "channel": channel,
        "rung_scores": {k: scores.get(k) for k in _RUNGS},
        "uplift_combined": uplift,            # adapter+gate over base (higher is better)
        "gate_only_uplift": gate_only,        # isolates the gate's contribution
        "adapter_only_uplift": adapter_only,  # isolates the adapter's contribution
        "objective_for_min": round(-uplift, 4),   # the loop MINIMIZES this
        "missing_rungs": missing,
    }


__all__ = ["parse_objective"]
