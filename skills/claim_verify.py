# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Skill: claim_verify_and_record — record with provenance, then verify (fail-closed)."""
from __future__ import annotations

from skills.core import sophia_skill
from skills.mcp_bridge import call


@sophia_skill(
    "claim_verify_and_record",
    summary="Record a claim with provenance, then verify it through the governance contract. Only 'accepted' may publish.",
    uses=("record_claim", "verify_claim"),
)
def claim_verify_and_record(
    *, content: str, idempotency_key: str, sources=None, clearance: str = "UNCLASSIFIED"
) -> dict:
    rec = call("record_claim", idempotency_key=idempotency_key, content=content, sources=sources or [])
    cid = rec.get("claim_id")
    if not cid:
        return {"verdict": "held", "reason": "claim was not recorded", "detail": rec, "failClosed": True}
    verdict = call("verify_claim", claim_id=cid, clearance=clearance)
    v = verdict.get("verdict", "held")
    return {
        "verdict": v,
        "claimId": cid,
        "publishable": v == "accepted",
        "confidence": verdict.get("confidence"),
        "reasons": verdict.get("reasons", []),
        "detail": verdict,
    }
