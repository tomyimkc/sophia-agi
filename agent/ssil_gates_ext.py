"""Extension gates for SSIL — G1 (value key) and G3 (capability key).

Two-key promotion: a self-modification needs an independent **capability** key (G3:
a meta-verified verifier exists for the task's core distinction — the gain is not a
fluke of an unverifiable task) AND a **value** key (G1: the proposal clears the
hard-floor public moral standard; gray-zone concerns escalate). Both reuse real
in-process Sophia modules; both return the standard {verdict, reasons, metrics}
shape so the orchestrator aggregates them like any other gate.

Deterministic, offline. Fail-closed: block -> reject, gray-zone / abstain ->
quarantine, clear -> promote.
"""
from __future__ import annotations

from typing import Any


def g1_moral_gate(proposal_text: str, *, context: dict | None = None) -> dict[str, Any]:
    """G1 value key — hard-floor public-standard check on the proposal itself."""
    from agent.public_standard_gate import check_public_standard

    d = check_public_standard(proposal_text, context=context or {}).to_dict()
    violations = d.get("violations", [])
    gray = d.get("grayZone", [])
    if d.get("verdict") == "block" or violations:
        verdict, reasons = "reject", tuple(f"public-standard violation: {v}" for v in (violations or ["blocked"]))
    elif gray:
        verdict, reasons = "quarantine", tuple(f"gray-zone concern (escalate): {g}" for g in gray)
    else:
        verdict, reasons = "promote", ("clears hard-floor public moral standard",)
    return {
        "schema": "sophia.ssil_g1_decision.v1", "gate": "G1", "candidateOnly": True, "level3Evidence": False,
        "verdict": verdict, "reasons": list(reasons),
        "metrics": {"publicStandardVerdict": d.get("verdict"), "violations": violations, "grayZone": gray,
                    "unmetDuties": d.get("unmetDuties", [])},
    }


def g3_capability_gate(verification_task: dict, *, seed: int = 0, min_precision: float = 0.9, min_recall: float = 0.8) -> dict[str, Any]:
    """G3 capability key — synthesize + meta-verify a verifier for the task's core
    distinction. Admitted (non-abstaining) with cleared floors -> promote; abstain
    (no trustworthy check) -> quarantine. Trust comes only from measured validation.
    """
    from agent.verifier_synthesis import synthesize

    res = synthesize(verification_task, seed=seed, min_precision=min_precision, min_recall=min_recall, meta_verify=True)
    admitted = len(res.admitted)
    test = res.test_stats
    if res.abstained or admitted == 0:
        verdict, reasons = "quarantine", ("no meta-verified verifier cleared the floor; task unverifiable",)
    else:
        verdict, reasons = "promote", (f"meta-verified verifier admitted ({admitted}); test precision/recall cleared floor",)
    return {
        "schema": "sophia.ssil_g3_decision.v1", "gate": "G3", "candidateOnly": True, "level3Evidence": False,
        "verdict": verdict, "reasons": list(reasons),
        "metrics": {"admitted": admitted, "abstained": res.abstained,
                    "testPrecision": getattr(test, "precision", None), "testRecall": getattr(test, "recall", None),
                    "splits": res.splits},
    }


def routing_verification_task(min_sources: int = 2, *, reps: int = 10) -> dict:
    """Build a G3 verification task: is the decision's CORE predicate (the
    well-sourced threshold, ``independent_sources >= min_sources``) meta-verifiable in
    principle? Labels come from the task's oracle definition on a clean grid (no
    observational noise) — a structural capability check that a trustworthy verifier
    EXISTS, distinct from G4's measurement of the candidate's gain on noisy held-out
    data. Both keys are required: G3 says the rule-class is verifiable, G4 says this
    candidate actually helps. (Integer threshold → the template library fits it
    cleanly and meta-verification clears the precision/recall floor.)
    """
    import random

    examples = [{"answer": s, "label": s >= min_sources} for s in list(range(0, 5)) * reps]
    random.Random(12345).shuffle(examples)  # representative fit/val/test splits
    return {"task_id": "routing_wellsourced_predicate_verifiable", "examples": examples}
