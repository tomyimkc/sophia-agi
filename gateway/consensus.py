# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Verified-consensus (P5): adjudicate competing answers by VERIFICATION, not a vote.

Given several candidate outputs (from different agents/skills/models), each is run through
the gateway's verify router. The winner is chosen among the *verified* candidates (highest
confidence), not by majority — so three confidently-wrong agents cannot outvote one
correct, verifiable one. If none verify, the result is held (fail-closed).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gateway.verify_router import verify_output


# --------------------------------------------------------------------------- #
# Judge-consensus policy — the ≥2-family judge gate as a declared, versioned
# object instead of convention spread across tools. The policy only *evaluates*
# whether a judged result clears the bar; it never scores anything itself.
# --------------------------------------------------------------------------- #

POLICY_SCHEMA = "sophia.judge_consensus_policy.v1"

#: Agreement metrics this repo reports. Cohen's κ is the pre-registered default;
#: Gwet's AC1 is the prevalence-paradox fallback (must be labelled as such).
AGREEMENT_METRICS = ("cohen_kappa", "gwet_ac1")


@dataclass(frozen=True)
class JudgeConsensusPolicy:
    """Declared quorum policy for LLM-judge certification.

    Encodes the no-overclaim gate's consensus half: how many *distinct* judge
    families, judge ≠ subject lineage, minimum seeds, and the inter-judge
    agreement floor. ``evaluate`` is fail-closed: any missing input is a failure,
    never a pass. Changing these numbers is a claim-contract change — do it in a
    reviewed commit, not at a call site."""

    min_families: int = 2
    require_distinct_families: bool = True
    judge_not_subject: bool = True
    min_seeds: int = 3
    agreement_metric: str = "cohen_kappa"
    min_agreement: float = 0.40
    tie_rule: str = "abstain"       # judges split → no verdict, never a coin flip
    on_missing: str = "abstain"     # any absent input → abstain (fail-closed)
    version: str = "1.0.0"
    notes: tuple = field(default_factory=tuple)

    def validate(self) -> "list[str]":
        problems = []
        if self.min_families < 2:
            problems.append("min_families < 2 breaks judge-family triangulation")
        if self.min_seeds < 3:
            problems.append("min_seeds < 3 cannot support a CI (no-overclaim gate)")
        if self.agreement_metric not in AGREEMENT_METRICS:
            problems.append(f"unknown agreement metric {self.agreement_metric!r}")
        if not (0.0 < self.min_agreement <= 1.0):
            problems.append("min_agreement must be in (0, 1]")
        if self.tie_rule != "abstain" or self.on_missing != "abstain":
            problems.append("tie/missing must abstain (fail-closed) in this repo")
        return problems

    def evaluate(self, *, families: "list[str]", seeds: "list[int] | list[str]",
                 agreement: "float | None", agreement_metric: "str | None" = None,
                 judge_lineages: "list[str] | None" = None,
                 subject_lineage: "str | None" = None) -> dict:
        """Does a judged result clear this policy? Returns {ok, failures[]} —
        failures name the exact clause, so a NO-GO is self-explaining."""
        failures = self.validate()
        fams = [f for f in (families or []) if f]
        n_fams = len(set(fams)) if self.require_distinct_families else len(fams)
        if n_fams < self.min_families:
            failures.append(f"families: {n_fams} distinct < required {self.min_families}")
        if len(set(seeds or [])) < self.min_seeds:
            failures.append(f"seeds: {len(set(seeds or []))} < required {self.min_seeds}")
        metric = agreement_metric or self.agreement_metric
        if agreement is None:
            failures.append("agreement: not reported (fail-closed abstain)")
        elif metric not in AGREEMENT_METRICS:
            failures.append(f"agreement metric {metric!r} not recognised")
        elif agreement < self.min_agreement:
            failures.append(f"agreement: {metric}={agreement} < floor {self.min_agreement}")
        elif metric != self.agreement_metric:
            failures.append(
                f"agreement metric substituted ({metric} for {self.agreement_metric}) — "
                f"allowed only if labelled as a fallback in the published artifact")
        if self.judge_not_subject:
            if subject_lineage is None or not judge_lineages:
                failures.append("judge/subject lineages not declared (fail-closed abstain)")
            elif any(subject_lineage == lin for lin in judge_lineages):
                failures.append(f"judge lineage equals subject lineage ({subject_lineage})")
        return {"schema": POLICY_SCHEMA, "version": self.version,
                "ok": not failures, "failures": failures}

    def to_dict(self) -> dict:
        return {"schema": POLICY_SCHEMA, "version": self.version,
                "minFamilies": self.min_families,
                "requireDistinctFamilies": self.require_distinct_families,
                "judgeNotSubject": self.judge_not_subject,
                "minSeeds": self.min_seeds,
                "agreementMetric": self.agreement_metric,
                "minAgreement": self.min_agreement,
                "tieRule": self.tie_rule, "onMissing": self.on_missing,
                "notes": list(self.notes)}


#: The VALIDATED-grade policy — mirrors the no-overclaim gate in
#: SESSION-HANDOVER/measurement-thesis (≥2 families, ≥3 seeds, κ ≥ 0.40).
VALIDATION_POLICY = JudgeConsensusPolicy(notes=(
    "matches the published no-overclaim gate; Gwet AC1 permitted only as a "
    "labelled prevalence-paradox fallback",
))


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
