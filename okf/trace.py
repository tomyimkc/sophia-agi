# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Surface a provenance-colored retrieval trace from multi_hop_recall results.

A recall-only engine's search trace answers "why was this retrieved?" (which entity
bridged the hop). Sophia's trace answers that AND "how much may I trust the chain that
found it?" — every hit carries the effective confidence rank of its own page and the
``provenanceFloor`` of the whole surfacing path, so a result reached through a weak
(legendary / anachronism / none_extant) bridge is visibly flagged rather than read at
face value.

Public API:
    trace_records(hits)        -> list[dict]   (machine-readable, JSON-safe)
    format_trace(query, hits)  -> str          (human-readable, provenance-colored)
"""

from __future__ import annotations

from okf.extract import RecallHit, is_capped
from okf.schema import CONFIDENCE_RANK  # re-exported via __all__ (see below)

# Rank -> short label, derived from the schema ladder so a drift there shows here.
_RANK_LABEL = {
    0: "none/anachronism",
    1: "legendary/disputed",
    2: "compiled/layered",
    3: "attributed",
    4: "consensus",
}


def _why(hit: RecallHit) -> str:
    """One-line account of how the hit surfaced (the retrieval reason)."""
    if hit.hops == 0:
        return "direct lexical match"
    # path = (seed_page, entity1, entity2, ...) — the last element is the reaching entity.
    bridges = " → ".join(hit.path[1:]) if len(hit.path) > 1 else ""
    seed = hit.path[0] if hit.path else "?"
    return f"{hit.hops}-hop via {seed} → [{bridges}]"


def trace_records(hits) -> "list[dict]":
    """Machine-readable trace: one JSON-safe dict per hit, in surfaced order."""
    out: list[dict] = []
    for rank, h in enumerate(hits, 1):
        out.append({
            "rank": rank,
            "eventId": h.event.id,
            "page": h.event.page_id,
            "hops": h.hops,
            "score": round(h.score, 4),
            "authorConfidence": h.event.author_confidence,
            "effectiveRank": h.event.confidence_rank,
            "provenanceFloor": h.provenance_floor,
            "capped": h.capped,
            "path": list(h.path),
            "why": _why(h),
            "text": h.event.text,
        })
    return out


def format_trace(query: str, hits, *, width: int = 64) -> str:
    """Human-readable, provenance-colored trace (plain text, terminal-safe)."""
    lines = [f'recall trace for: "{query}"  ({len(hits)} hit(s))', "─" * width]
    if not hits:
        lines.append("  (no hits)")
        return "\n".join(lines)
    for rec in trace_records(hits):
        flag = "⚑ CAPPED" if rec["capped"] else "  ok    "
        floor_lbl = _RANK_LABEL.get(rec["provenanceFloor"], "?")
        snippet = rec["text"]
        if len(snippet) > width:
            snippet = snippet[: width - 1] + "…"
        lines.append(
            f"{rec['rank']:>2}. {flag}  {rec['page']:<26} "
            f"floor={rec['provenanceFloor']}({floor_lbl})  {rec['why']}"
        )
        lines.append(f"      {snippet}")
    capped = sum(1 for r in trace_records(hits) if r["capped"])
    lines.append("─" * width)
    lines.append(f"  {capped}/{len(hits)} hit(s) rest on weak provenance (capped).")
    return "\n".join(lines)


# Re-export for callers that want the threshold without importing extract directly.
__all__ = ["trace_records", "format_trace", "is_capped", "CONFIDENCE_RANK"]
