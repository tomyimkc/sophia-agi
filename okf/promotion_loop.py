# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Projection → boundary promotion loop (candidate queue, human-gated commit).

Bulk projection yields ``PromotionCandidate`` records. This module stages them in a
pending JSONL queue (``promoted: false``) and optionally commits human-approved rows
to the wiki boundary via ``wiki_store.upsert``. Bulk/projection output is never
shipped raw. Default-deny: unapproved candidates are never committed.
"""

from __future__ import annotations

import hashlib
import json
import re
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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _candidate_key(row: dict) -> tuple:
    return (str(row.get("nodeId", "")), str(row.get("submittedAt", ""))[:10])


def _body_store_path(pending: Path, node_id: str, day: str = "") -> Path:
    """Filename-safe, per-row body path under the pending dir.

    ``node_id`` can originate from content/user input, so it is sanitized to a
    restricted slug (no path separators or ``..``) plus a short stable hash to
    avoid slug collisions. A ``day`` segment keys the body to the specific
    pending row (deduped by nodeId+day), so a resubmission on a different day
    cannot overwrite an earlier approved body.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(node_id)).strip("._-")[:64] or "node"
    digest = hashlib.sha1(str(node_id).encode("utf-8")).hexdigest()[:8]
    stem = f"_body_{slug}_{digest}_{day}" if day else f"_body_{slug}_{digest}"
    return pending.parent / f"{stem}.md"


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

    day = now[:10]
    for cand in projection.promoted:
        body_path = _body_store_path(out, cand.node_id, day)
        row = {
            "schema": "sophia.projection_candidate.v1",
            "nodeId": cand.node_id,
            "meta": cand.meta,
            "bodyPreview": cand.body[:500],
            "bodyPath": str(body_path),
            "checks": cand.checks,
            "source": source,
            "candidateOnly": projection.candidateOnly,
            "promoted": False,
            "approved": False,
            "committed": False,
            "submittedAt": now,
        }
        key = _candidate_key(row)
        if key in seen:
            continue
        body_path.write_text(cand.body, encoding="utf-8")
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


def approve_projection_candidate(
    node_id: str,
    *,
    path: Path | None = None,
    reviewer: str = "human",
    note: str = "",
) -> dict:
    """Human gate: mark one pending candidate approved (still not wiki-committed)."""
    pending = path or PENDING_PATH
    rows = _read_jsonl(pending)
    found = False
    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        if row.get("nodeId") != node_id:
            continue
        if row.get("committed"):
            return {
                "ok": True,
                "alreadyApproved": True,
                "nodeId": node_id,
                "committed": True,
            }
        row["approved"] = True
        row["promoted"] = True
        row["reviewer"] = reviewer
        row["reviewNote"] = note
        row["approvedAt"] = now
        found = True
        break
    if not found:
        return {"ok": False, "error": f"no pending candidate for nodeId '{node_id}'"}
    _write_jsonl(pending, rows)
    return {"ok": True, "nodeId": node_id, "approved": True}


def commit_approved_candidate(
    node_id: str,
    *,
    path: Path | None = None,
    tier: str = "draft",
    reviewer: str = "human",
    note: str = "",
) -> dict:
    """Upsert one human-approved pending candidate to the wiki boundary (default-deny)."""
    from agent import wiki_store

    pending = path or PENDING_PATH
    rows = _read_jsonl(pending)
    target: dict | None = None
    for row in rows:
        if row.get("nodeId") != node_id:
            continue
        target = row
        break
    if target is None:
        return {"ok": False, "error": f"no pending candidate for nodeId '{node_id}'", "defaultDeny": True}

    if not target.get("approved") and not target.get("promoted"):
        return {
            "ok": False,
            "error": "candidate not approved — default-deny",
            "defaultDeny": True,
            "nodeId": node_id,
        }

    if target.get("committed"):
        return {
            "ok": True,
            "idempotent": True,
            "nodeId": node_id,
            "alreadyCommitted": True,
        }

    meta = dict(target.get("meta") or {})
    meta.setdefault("pageType", "concept")
    meta["projectionApproved"] = True
    meta["reviewer"] = reviewer or target.get("reviewer", "human")
    if note or target.get("reviewNote"):
        meta["reviewNote"] = note or target.get("reviewNote", "")

    # Read the exact body persisted for THIS pending row (path stored at submit),
    # falling back to a legacy/computed path then the inline preview.
    stored = target.get("bodyPath")
    body_path = Path(stored) if stored else _body_store_path(
        pending, node_id, str(target.get("submittedAt", ""))[:10]
    )
    body = body_path.read_text(encoding="utf-8") if body_path.exists() else str(target.get("bodyPreview", ""))
    result = wiki_store.upsert(node_id, meta=meta, body=body, tier=tier)
    if not result.get("ok"):
        return {"ok": False, "wiki": result, "nodeId": node_id}

    for row in rows:
        if row.get("nodeId") == node_id:
            row["committed"] = True
            row["committedAt"] = datetime.now(timezone.utc).isoformat()
            break
    _write_jsonl(pending, rows)
    return {"ok": True, "wiki": result, "nodeId": node_id, "committed": True}


def promotion_loop_report(path: Path | None = None) -> dict:
    """Summarize pending queue for agi-proof artifacts."""
    pending = path or PENDING_PATH
    rows = _read_jsonl(pending)
    submitted = len(rows)
    approved = sum(1 for r in rows if r.get("approved") or r.get("promoted"))
    committed = sum(1 for r in rows if r.get("committed"))
    return {
        "schema": "sophia.promotion_loop_report.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "path": str(pending),
        "submitted": submitted,
        "approved": approved,
        "committed": committed,
        "defaultDeny": True,
    }


__all__ = [
    "PENDING_PATH",
    "submit_projection_candidates",
    "approve_projection_candidate",
    "commit_approved_candidate",
    "promotion_loop_report",
    "_FAIL_CLOSED_VERDICTS",
]
