"""Real long-horizon autonomy on the governance contract — with recovery.

Executes an N-step plan, gating every step (record -> verify). A step that is HELD
(e.g., missing a source) triggers a bounded repair (attach the repair source and
re-gate) — recovery. A REJECTED step (a refuted source) is unrecoverable drift and
halts the run. The measured ``effectiveHorizon`` = consecutive accepted steps before
drift, with the kill switch + decision log as the safety substrate. Deterministic;
the harness for a real long-horizon capability number (which needs a live backend +
time to be a capability claim, not just mechanism).
"""

from __future__ import annotations

from sophia_contract.service import SophiaContract


def run_long_horizon(steps: "list[dict]", *, contract: "SophiaContract | None" = None,
                     max_repairs: int = 1) -> dict:
    """steps: each {id, content, sources[], blp_level?, repair_sources?[]}.
    Returns the plan outcome incl. effectiveHorizon and recoveries."""
    contract = contract or SophiaContract()
    completed = recoveries = 0
    drifted_at = None
    trace: list = []

    for i, step in enumerate(steps):
        key = f"lh:{step.get('id', i)}"
        sources = list(step.get("sources", []))
        verdict = None
        for attempt in range(max_repairs + 1):
            claim = contract.record_claim({
                "idempotency_key": f"{key}:{attempt}", "content": step["content"],
                "sources": sources, "blp_level": step.get("blp_level", "UNCLASSIFIED")})
            if "error" in claim:
                verdict = {"verdict": "rejected", "reason": claim["error"]["code"]}
                break
            verdict = contract.verify_claim({"claim_id": claim["claim_id"]})
            if verdict["verdict"] == "accepted":
                break
            if verdict["verdict"] == "held" and step.get("repair_sources") and attempt < max_repairs:
                sources = sources + list(step["repair_sources"])  # repair, then re-gate
                recoveries += 1
                continue
            break

        trace.append({"step": step.get("id", i), "verdict": verdict["verdict"],
                      "held_reason": verdict.get("held_reason")})
        if verdict["verdict"] == "accepted":
            completed += 1
        else:
            drifted_at = step.get("id", i)
            break

    return {
        "plannedSteps": len(steps),
        "completedSteps": completed,
        "effectiveHorizon": completed,
        "recoveries": recoveries,
        "driftedAt": drifted_at,
        "trace": trace,
        "interpretation": ("Consecutive gate-accepted steps before unrecoverable drift; "
                           "held steps were repaired and retried (recovery). Mechanism only — "
                           "a capability number needs a live backend + real tasks."),
    }
