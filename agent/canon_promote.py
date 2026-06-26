# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Canon promotion — human sign-off that elevates a reviewed draft into consolidated memory.

The self-correction loop ends at a `needsReview` draft (a gap stub auto-filled from a trusted
source, `agent/source_fill.py`). The final step is **deliberately human**: a reviewer approves
the draft, and only then is it elevated from the quarantined ``draft`` tier into the agent's
consolidated ``memory`` tier — where it outranks drafts and grounds future answers without the
low-confidence hedge a stub carries.

Two invariants keep this safe:
  - **Human-gated.** ``promote`` requires an explicit ``approver`` — there is no auto-canon path;
    the agent surfaces candidates (`pending_reviews`) but a person signs off.
  - **Re-gated.** Promotion re-runs the provenance gate (`agent/wiki_store`) on the page, so a
    draft that no longer passes (e.g. an edit introduced a forbidden attribution) is rejected at
    the boundary rather than laundered into canon.

The hand-authored canonical wiki (`wiki/`) is never touched — promotion writes to the
agent-owned memory tier (`agent/memory/wiki/`), which `index_by_id` ranks below canonical but
above drafts. Deterministic and offline.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_drafts() -> "list":
    """Pages currently in the draft tier (read directly so a memory shadow can't hide them)."""
    from okf import load_pages

    from agent import wiki_store

    draft_dir = Path(wiki_store.DRAFT_DIR)
    return list(load_pages(draft_dir)) if draft_dir.exists() else []


def pending_reviews() -> "list[dict]":
    """Draft pages awaiting review, each with the provenance + live gate status a reviewer needs.

    Sorted gate-failing first (those need attention before they could ever be promoted), then by
    id. A reviewer reads this to decide promote/reject without trusting the stub blindly.
    """
    from agent import wiki_store

    out: list[dict] = []
    for p in _load_drafts():
        if not p.meta.get("needsReview"):
            continue
        ok, reasons = wiki_store.gate(p.meta, p.body)
        out.append({
            "id": p.id,
            "pageType": p.meta.get("pageType"),
            "provenance": p.meta.get("provenance"),
            "authorConfidence": p.meta.get("authorConfidence"),
            "attributedAuthor": p.meta.get("attributedAuthor"),
            "sources": list(p.meta.get("sources") or []),
            "doNotAttributeTo": list(p.meta.get("doNotAttributeTo") or []),
            "gatePasses": ok,
            "gateReasons": reasons,
        })
    out.sort(key=lambda r: (r["gatePasses"], r["id"]))  # failing (False) first
    return out


def promote(page_id: str, *, approver: str, remove_draft: bool = True) -> dict:
    """Elevate a reviewed draft into the memory tier on human sign-off. Re-gated, fail-closed.

    Requires a non-empty ``approver`` (the human signing off). Clears ``needsReview``, stamps
    ``reviewedBy`` / ``reviewedAt`` / ``reviewStatus``, and writes to the memory tier via the
    gated `wiki_store.upsert`. Removes the draft file on success so the page isn't duplicated
    across tiers. Never raises; returns a structured result.
    """
    if not approver or not str(approver).strip():
        return {"ok": False, "id": page_id, "reason": "approver required (human sign-off)"}

    from agent import wiki_store

    draft = next((p for p in _load_drafts() if p.id == page_id), None)
    if draft is None:
        return {"ok": False, "id": page_id, "reason": "no draft to promote"}

    meta = dict(draft.meta)
    meta["needsReview"] = False          # explicit: upsert merges, so we must overwrite not drop
    meta["reviewedBy"] = str(approver)
    meta["reviewedAt"] = _now()
    meta["reviewStatus"] = "approved"

    result = wiki_store.upsert(page_id, meta=meta, body=draft.body, tier="memory")
    if not result.get("ok"):
        return {"ok": False, "id": page_id, "reason": "gate rejected on promotion",
                "reasons": result.get("reasons")}

    if remove_draft:
        Path(wiki_store.DRAFT_DIR, f"{page_id}.md").unlink(missing_ok=True)
    return {"ok": True, "id": page_id, "promotedTo": "memory", "path": result.get("path"),
            "approver": str(approver)}


def reject(page_id: str, *, approver: str, reason: str = "", remove_draft: bool = True) -> dict:
    """Reject a draft on review — remove it from the draft tier (it never reaches canon)."""
    if not approver or not str(approver).strip():
        return {"ok": False, "id": page_id, "reason": "approver required"}
    from agent import wiki_store

    path = Path(wiki_store.DRAFT_DIR, f"{page_id}.md")
    existed = path.exists()
    if remove_draft:
        path.unlink(missing_ok=True)
    return {"ok": True, "id": page_id, "rejected": True, "existed": existed,
            "approver": str(approver), "reason": reason}


__all__ = ["pending_reviews", "promote", "reject"]
