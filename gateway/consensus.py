"""Verified-consensus (P5): adjudicate competing answers by VERIFICATION, not a vote.

Given several candidate outputs (from different agents/skills/models), each is run through
the gateway's verify router. The winner is chosen among the *verified* candidates (highest
confidence), not by majority — so three confidently-wrong agents cannot outvote one
correct, verifiable one. If none verify, the result is held (fail-closed).
"""

from __future__ import annotations

from gateway.verify_router import verify_output


def verified_consensus(gateway, candidates: "list[dict]", *, verifier_ref: str = "grounding",
                       blp_level: str = "UNCLASSIFIED", clearance: str = "UNCLASSIFIED",
                       topic: str = "consensus") -> dict:
    """candidates: [{id, output}]. Returns the chosen verified answer + the per-candidate
    verdicts. Adjudication = verification, not vote."""
    verdicts: list = []
    for i, cand in enumerate(candidates):
        v, cid = verify_output(
            gateway.contract, verifier_ref=verifier_ref, output=cand["output"],
            tool_id=cand.get("id", f"cand{i}"), args={}, blp_level=blp_level, role=None,
            clearance=clearance, idempotency_key=f"consensus:{topic}:{i}")
        verdicts.append({"id": cand.get("id", f"cand{i}"), "verdict": v.get("verdict"),
                         "confidence": v.get("confidence", 0.0), "provenance_id": cid,
                         "output": cand["output"]})
    accepted = [v for v in verdicts if v["verdict"] == "accepted"]
    if not accepted:
        return {"topic": topic, "decided": False, "verdict": "held", "held_reason": "needs_human",
                "reason": "no candidate verified", "candidates": [
                    {k: c[k] for k in ("id", "verdict", "confidence")} for c in verdicts]}
    winner = max(accepted, key=lambda v: v["confidence"])
    return {
        "topic": topic, "decided": True, "winner": winner["id"],
        "answer": winner["output"], "provenance_id": winner["provenance_id"],
        "acceptedCount": len(accepted), "totalCandidates": len(candidates),
        "adjudication": "by verification (not majority vote)",
        "candidates": [{k: c[k] for k in ("id", "verdict", "confidence")} for c in verdicts],
    }
