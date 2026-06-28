# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Self-improving corpus loop — turn answering failures into an enrichment worklist.

Lifelong learning is not only retention; it is deciding *what to learn next*. Every time
the grounded agent abstains for lack of a route, abstains for lack of a grounded source,
or falls back because a page is thin, that is a **knowledge gap** the corpus could close.
This module logs those gaps (append-only) and turns them — plus the recall audit's thin
pages — into a frequency-ranked worklist: the corpus grows where it is actually queried,
not by hand-guessing.

    from agent.knowledge_gap_log import log_gap, gap_worklist, load_gaps
    log_gap("when did X happen?", target="x", policy="grounded_fallback", path=LEDGER)
    gap_worklist(load_gaps(LEDGER), thin_targets=audit["thinTargets"])
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Policies that signal the corpus came up short for a real query. Includes the grounded-search
# reflexes (`agent.grounded_search`): a hedged or abstained search is a perception gap the
# corpus could close, so it feeds the same enrichment worklist as the grounded agent's gaps.
GAP_POLICIES = {
    "abstain_no_route", "abstain_no_source", "grounded_fallback", "fallback_gated_abstain",
    "grounded_search_abstain", "grounded_search_hedge", "grounded_search_ungrounded",
}


def is_gap(policy: str) -> bool:
    return policy in GAP_POLICIES


def log_gap(query: str, *, target, policy: str, path, by: str = "grounded_agent") -> "dict | None":
    """Append a knowledge-gap record (only if ``policy`` indicates a gap). Returns the
    record written, or None if the policy was a clean grounded answer."""
    if not is_gap(policy):
        return None
    rec = {
        "event": "knowledge_gap",
        "at": datetime.now(timezone.utc).isoformat(),
        "by": by,
        "query": query,
        "target": target,
        "policy": policy,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def load_gaps(path) -> "list[dict]":
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def gap_worklist(gaps, thin_targets=()) -> "dict[str, Any]":
    """Frequency-ranked enrichment worklist from logged gaps + audit thin pages.

    A target queried often but routed to a thin/absent page is high priority. Audit thin
    targets seed the list even if not yet queried (known-weak pages)."""
    by_target = Counter(g.get("target") for g in gaps if g.get("target"))
    no_route = sum(1 for g in gaps if g.get("policy") == "abstain_no_route")
    items = []
    for target in set(by_target) | set(thin_targets):
        items.append({
            "target": target,
            "gapHits": by_target.get(target, 0),
            "auditThin": target in set(thin_targets),
        })
    # Most-queried first; audit-thin breaks ties (known-weak even if unqueried).
    items.sort(key=lambda x: (x["gapHits"], x["auditThin"]), reverse=True)
    return {
        "schema": "sophia.knowledge_gap_worklist.v1",
        "candidateOnly": True,
        "totalGaps": len(gaps),
        "unroutableQueries": no_route,
        "distinctTargets": len(items),
        "worklist": items,
    }


__all__ = ["GAP_POLICIES", "is_gap", "log_gap", "load_gaps", "gap_worklist"]
