"""Verify a tool's OUTPUT, routing to the right verifier (Universal Verify, idea #8).

Returns ``(verdict, claim_id)`` where ``verdict`` is a governance-contract Verdict
(only ``accepted`` may be returned to the agent). Every path records a Claim first, so
even a low-trust pass is provenance-stamped.

  - grounding : the output must carry real ``sources`` (fail-closed: none -> held(no_source))
  - env:<kind>: verify by EXECUTING the candidate (selfextend.env_verifier)
  - deterministic / none : recorded with the tool as source; 'none' is flagged low-trust
"""

from __future__ import annotations

from selfextend.env_verifier import verify_by_execution
from sophia_contract.models import build_verdict


def _sources_from(output) -> "list":
    if isinstance(output, dict):
        s = output.get("sources")
        if isinstance(s, list):
            return s
    return []


def verify_output(contract, *, verifier_ref: str, output, tool_id: str, args: dict,
                  blp_level: str, role: "str | None", clearance: str,
                  idempotency_key: str) -> "tuple[dict, str | None]":
    # Execution-verified paths: the environment decides; still record for provenance.
    if verifier_ref.startswith("env:"):
        kind = verifier_ref.split(":", 1)[1]
        candidate = output.get("candidate") if isinstance(output, dict) else str(output)
        spec = args.get("verify_spec", args)
        res = verify_by_execution(kind, str(candidate), spec)
        claim = contract.record_claim({
            "idempotency_key": idempotency_key, "content": str(output),
            "sources": [tool_id], "blp_level": blp_level, **({"role": role} if role else {})})
        cid = claim.get("claim_id") if "error" not in claim else None
        verdict = build_verdict(
            "accepted" if res["passed"] else "rejected",
            confidence=1.0 if res["passed"] else 0.9,
            reasons=[f"executed via {verifier_ref}: {'pass' if res['passed'] else 'fail'}"],
            cited_evidence=[{"id": tool_id, "status": "ok"}],
            suggested_fix=None if res["passed"] else "tool output failed execution check",
            roi_estimate={"founder_minutes_saved": 5.0 if res["passed"] else 0.0,
                          "basis": "execution-verified" if res["passed"] else "rejected by execution"})
        return verdict, cid

    # Source/grounding paths: route through the contract's own verify pipeline.
    if verifier_ref == "grounding":
        sources = _sources_from(output)               # require real sources -> fail-closed
    else:                                              # deterministic / none: tool as source
        sources = [tool_id]
    req = {"idempotency_key": idempotency_key, "content": _content(output),
           "sources": sources, "blp_level": blp_level}
    if role:
        req["role"] = role
    claim = contract.record_claim(req)
    if "error" in claim:
        return ({"verdict": "rejected", "confidence": 0.0,
                 "reasons": [claim["error"]["message"]], "cited_evidence": [],
                 "suggested_fix": "fix the recorded claim", "_error": claim["error"]}, None)
    verdict = contract.verify_claim({"claim_id": claim["claim_id"], **({"role": role} if role else {})},
                                    clearance=clearance)
    if verifier_ref == "none" and verdict.get("verdict") == "accepted":
        verdict.setdefault("reasons", []).append("no output verifier (low trust)")
    return verdict, claim.get("claim_id")


def _content(output) -> str:
    if isinstance(output, dict):
        return str(output.get("answer") or output.get("text") or output)
    return str(output)
