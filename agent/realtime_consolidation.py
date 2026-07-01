# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Slow, gated consolidation loop — turn verifier-certified beliefs into a
reversible weight-update candidate.

The fast loop (:mod:`agent.realtime_grounding`) writes fact-checked live data to an
external belief store and never touches weights. This module is the second speed:
it consolidates only the ``ingested`` (verifier-certified) belief rows into GRPO/DPO
training rows and records each batch as a **reversibility-ledger** entry — a
mergeable LoRA delta logged against its provenance hash, so a later-falsified fact
can be un-merged. That fuses knowledge-editing reversibility, replay, and the
claim-gate audit trail into one auditable pipeline.

Two disciplines are enforced here so the output is honest:

  * Habit-not-fact: emitted targets teach the *routing/epistemic habit* (route,
    epistemic_status, verdict, confidence, needed_sources, valid_until) rather than a
    bare ground-truth fact — the same contract ``tools/lint_training_rows.py`` checks.
  * Self-decontamination: every emitted prompt is re-checked against the eval surface
    (:mod:`agent.streaming_decontam`) and dropped if it near-duplicates a held-out
    prompt, and the drop count is reported (no silent truncation).

The actual GPU training is the ``--live`` seam only (shell out to ``tools/run_rlvr.py``
on RunPod via GitHub Actions); this module's default is a dry run that changes no
weights. Every record carries ``candidateOnly=True`` / ``level3Evidence=False``.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent import streaming_decontam as sd

SCHEMA = "sophia.realtime_consolidation.v1"
LEDGER_SCHEMA = "sophia.reversibility_ledger.v1"
ROW_SCHEMA = "sophia.realtime_training_row.v1"
REWARD_PROVENANCE = "realtime_grounding_gate"

GPU_SEAM_NOTE = (
    "dry run: no weights changed. The live weight update is the seam only — "
    "verifier-certified rows are handed to tools/run_rlvr.py (DPO/GRPO) on RunPod "
    "via GitHub Actions, never trained locally, and the resulting LoRA delta is "
    "logged in the reversibility ledger so it can be un-merged."
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def reward_for(belief_row: dict[str, Any]) -> float:
    """Bounded [-1, 1] reward from the verifier — never a self-score.

    Ingested (verifier-accepted) rows pay their weakest-link confidence; anything
    not verifier-accepted pays nothing. Mirrors the verified-trace reward clamp.
    """
    if belief_row.get("ingestState") != "ingested" or belief_row.get("verdict") != "accepted":
        return 0.0
    return round(max(-1.0, min(1.0, float(belief_row.get("confidence", 0.0)))), 4)


def to_training_row(belief_row: dict[str, Any]) -> dict[str, Any]:
    """Build one habit-shaped GRPO/DPO training row from a verified belief.

    The assistant target is a routing/epistemic structure (not a bare fact), so it
    satisfies the source-discipline habit contract. It carries the reward and the
    provenance needed to trace and reverse it.
    """
    claim = str(belief_row.get("claim", ""))
    needed = sorted({
        ref for ref in (
            str(s.get("url") or s.get("id") or s.get("domain") or "")
            for s in belief_row.get("sources", []) if s
        ) if ref
    }) or ["(deterministic verifier; no external source)"]
    target = {
        "route": "grounded-external",
        "epistemic_status": "externally-verified",
        "verdict": belief_row.get("verdict", "accepted"),
        "confidence": belief_row.get("confidence", 0.0),
        "needed_sources": needed,
        "valid_until": belief_row.get("validUntil") or None,
        "reason": "assert only with cited live evidence; re-verify before validUntil; abstain if evidence is stale",
    }
    return {
        "schema": ROW_SCHEMA,
        "messages": [
            {"role": "user", "content": f"A live source states: {claim}\nDecide whether to assert this and how to qualify it."},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ],
        "reward": reward_for(belief_row),
        "rewardProvenance": REWARD_PROVENANCE,
        "metadata": {
            "task_family": "realtime_grounding",
            "claimId": belief_row.get("claimId", ""),
            "nonconformity": belief_row.get("nonconformity"),
            "sourceTimestamp": belief_row.get("sourceTimestamp", ""),
        },
        "candidateOnly": True,
        "level3Evidence": False,
    }


def _provenance_hash(rows: list[dict[str, Any]]) -> str:
    blob = json.dumps([r.get("metadata", {}).get("claimId", "") for r in rows], sort_keys=True)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def ledger_entry(delta_id: str, rows: list[dict[str, Any]], *, based_on_spec: str, created_at: str | None = None) -> dict[str, Any]:
    """A reversible, un-mergeable record of a consolidation batch (a LoRA delta)."""
    rewards = [float(r.get("reward", 0.0)) for r in rows]
    return {
        "schema": LEDGER_SCHEMA,
        "deltaId": delta_id,
        "basedOnSpec": based_on_spec,
        "nRows": len(rows),
        "claimIds": [r.get("metadata", {}).get("claimId", "") for r in rows],
        "provenanceHash": _provenance_hash(rows),
        "meanReward": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
        "mergeState": "pending",  # pending -> merged -> reverted
        "canRevert": True,
        "createdAt": created_at or _now_iso(),
        "candidateOnly": True,
        "level3Evidence": False,
    }


def revert(ledger_path: str | Path, delta_id: str, *, reverted_at: str | None = None) -> dict[str, Any]:
    """Flip a ledger delta to ``reverted`` (un-merge a later-falsified batch)."""
    path = Path(ledger_path)
    if not path.exists():
        return {"ok": False, "reason": "no ledger", "deltaId": delta_id}
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # tolerate a malformed/partial ledger line
    found = False
    for e in entries:
        if e.get("deltaId") == delta_id:
            e["mergeState"] = "reverted"
            e["revertedAt"] = reverted_at or _now_iso()
            found = True
    path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + ("\n" if entries else ""), encoding="utf-8")
    return {"ok": found, "reason": "reverted" if found else "deltaId not found", "deltaId": delta_id}


def _read_ingested(belief_store_path: str | Path) -> list[dict[str, Any]]:
    path = Path(belief_store_path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # tolerate a malformed/partial store line
        if row.get("ingestState") == "ingested":
            out.append(row)
    return out


def consolidate(
    belief_store_path: str | Path,
    *,
    out_dir: str | Path,
    based_on_spec: str,
    delta_id: str,
    root: Path | None = None,
    eval_prompts: set[str] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Consolidate verifier-certified beliefs into a dry-run training candidate.

    Reads ``ingested`` belief rows, emits habit-shaped training rows (self-decontam'd
    against the eval surface), writes ``rows.jsonl`` + appends a reversibility-ledger
    entry under ``out_dir``, and returns a report. Changes no weights.
    """
    if eval_prompts is None:
        eval_prompts = sd.eval_surface(root)
    ingested = _read_ingested(belief_store_path)
    rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for b in ingested:
        row = to_training_row(b)
        prompt = row["messages"][0]["content"]
        check = sd.content_decontam(prompt, eval_prompts)
        if not check["ok"]:
            dropped.append({"claimId": b.get("claimId", ""), "reason": check["reason"]})
            continue
        rows.append(row)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    entry = ledger_entry(delta_id, rows, based_on_spec=based_on_spec, created_at=created_at)
    ledger_path = out / "reversibility_ledger.jsonl"
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return {
        "schema": SCHEMA,
        "candidateOnly": True,
        "level3Evidence": False,
        "dryRun": True,
        "nIngested": len(ingested),
        "nRows": len(rows),
        "nDropped": len(dropped),
        "dropped": dropped,
        "rowsPath": str(rows_path),
        "ledgerPath": str(ledger_path),
        "ledgerEntry": entry,
        "basedOnSpec": based_on_spec,
        "note": GPU_SEAM_NOTE,
    }


__all__ = [
    "SCHEMA",
    "LEDGER_SCHEMA",
    "ROW_SCHEMA",
    "REWARD_PROVENANCE",
    "GPU_SEAM_NOTE",
    "reward_for",
    "to_training_row",
    "ledger_entry",
    "revert",
    "consolidate",
]
