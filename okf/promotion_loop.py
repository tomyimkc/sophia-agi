# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Projection → boundary promotion loop (candidate queue, human-gated commit).

Bulk projection yields ``PromotionCandidate`` records. This module stages them in a
pending JSONL queue (``promoted: false``) and optionally commits human-approved rows
to the wiki boundary via ``wiki_store.upsert``. Bulk/projection output is never
shipped raw.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from okf.projection import PromotionCandidate, ProjectionResult

ROOT = Path(__file__).resolve().parents[1]
PENDING_PATH = ROOT / "training" / "feedback" / "pending_projection_candidates.jsonl"

_FAIL_CLOSED_VERDICTS = frozenset({"block", "abstain", "escalate", "retrieve", "clarify"})


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _candidate_key(row: dict) -> tuple:
    return (str(row.get("nodeId", "")), str(row.get("submittedAt", ""))[:10])


def submit_projection_candidates(
    projection: ProjectionResult,
    *,
    path: Path | None = None,
    source: str = "okf_projection",
) -> dict:
    """Append projection-passing candidates to the pending queue (deduped by nodeId+day)."""
    out = path or PENDING_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    existing = _read_jsonl(out)
    seen = {_candidate_key(r) for r in existing}
    submitted = 0
    now = datetime.now(timezone.utc).isoformat()

    for cand in projection.promoted:
        row = {
            "schema": "sophia.projection_candidate.v1",
            "nodeId": cand.node_id,
            "meta": cand.meta,
            "bodyPreview": cand.body[:500],
            "checks": cand.checks,
            "source": source,
            "candidateOnly": projection.candidateOnly,
            "promoted": False,
            "submittedAt": now,
        }
        key = _candidate_key(row)
        if key in seen:
            continue
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        seen.add(key)
        submitted += 1

    return {
        "schema": "sophia.submit_projection_candidates.v1",
        "candidateOnly": True,
        "path": str(out),
        "submitted": submitted,
        "pendingTotal": len(existing) + submitted,
        "promotedCount": len(projection.promoted),
    }


def commit_approved_candidate(
    node_id: str,
    *,
    path: Path | None = None,
    tier: str = "draft",
    reviewer: str = "human",
    note: str = "",
) -> dict:
    """Upsert one human-approved pending candidate to the wiki boundary."""
    from agent import wiki_store

    pending = path or PENDING_PATH
    rows = _read_jsonl(pending)
    target: dict | None = None
    for row in rows:
        if row.get("nodeId") == node_id and row.get("promoted") is True:
            target = row
            break
    if target is None:
        return {"ok": False, "error": f"no promoted pending candidate for nodeId '{node_id}'"}

    meta = dict(target.get("meta") or {})
    meta.setdefault("pageType", "concept")
    meta["projectionApproved"] = True
    meta["reviewer"] = reviewer
    if note:
        meta["reviewNote"] = note

    body_path = pending.parent / f"_body_{node_id}.md"
    body = body_path.read_text(encoding="utf-8") if body_path.exists() else str(target.get("bodyPreview", ""))
    result = wiki_store.upsert(node_id, meta=meta, body=body, tier=tier)
    return {"ok": bool(result.get("ok")), "wiki": result, "nodeId": node_id}


__all__ = [
    "PENDING_PATH",
    "submit_projection_candidates",
    "commit_approved_candidate",
    "_FAIL_CLOSED_VERDICTS",
]
