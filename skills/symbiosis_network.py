# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Mycorrhizal symbiosis network — verified nutrient exchange between agents.

Agents exchange only verifier-passing facts via a small JSON protocol. Nutrients
never bypass the boundary; failed verification is held, not forwarded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from skills.core import sophia_skill

SCHEMA = "sophia.symbiosis.nutrient.v1"


@dataclass
class Nutrient:
    """One verified fact packet exchanged across the mycelial network."""

    claim: str
    evidence: str
    donor_id: str
    verifier_ref: str = "provenance_faithful"
    tradition: str | None = None
    createdAt: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "claim": self.claim,
            "evidence": self.evidence,
            "donorId": self.donor_id,
            "verifierRef": self.verifier_ref,
            "tradition": self.tradition,
            "createdAt": self.createdAt,
        }


def _verify_nutrient(nutrient: Nutrient) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not nutrient.claim.strip():
        reasons.append("empty claim")
    if not nutrient.evidence.strip():
        reasons.append("empty evidence")
    try:
        from agent.verifiers import provenance_faithful

        blob = f"{nutrient.claim}\n{nutrient.evidence}"
        pf = provenance_faithful()(blob, None, {})
        if not pf.get("passed"):
            reasons.extend(pf.get("reasons") or [])
    except Exception as exc:
        reasons.append(f"verifier error: {type(exc).__name__}")
    return (not reasons, reasons)


def exchange_nutrient(nutrient: Nutrient) -> dict[str, Any]:
    """Accept or hold one nutrient packet (fail-closed)."""
    ok, reasons = _verify_nutrient(nutrient)
    out = nutrient.to_dict()
    out["ok"] = ok
    out["verdict"] = "accepted" if ok else "held"
    out["reasons"] = reasons
    out["candidateOnly"] = True
    return out


def broadcast_nutrients(nutrients: list[Nutrient]) -> dict[str, Any]:
    """Fan-out exchange; returns accepted vs held tallies."""
    results = [exchange_nutrient(n) for n in nutrients]
    accepted = [r for r in results if r.get("verdict") == "accepted"]
    held = [r for r in results if r.get("verdict") != "accepted"]
    return {
        "schema": "sophia.symbiosis.broadcast.v1",
        "candidateOnly": True,
        "acceptedCount": len(accepted),
        "heldCount": len(held),
        "accepted": accepted,
        "held": held,
    }


@sophia_skill(
    name="symbiosis_exchange",
    summary="Exchange a verified nutrient packet across the symbiosis network.",
    uses=("sophia_gate_check",),
)
def symbiosis_exchange(
    *,
    claim: str,
    evidence: str,
    donor_id: str = "agent_local",
    tradition: str | None = None,
) -> dict[str, Any]:
    nutrient = Nutrient(claim=claim, evidence=evidence, donor_id=donor_id, tradition=tradition)
    return exchange_nutrient(nutrient)
