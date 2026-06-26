# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""CapabilityScore + report assembly (pure stdlib).

Every frontier number travels with its provenance and its scorer's calibration so
it is reproducible and honestly bounded. A CapabilityScore is *not* a safety
certificate — it is "measured below/above threshold under conditions X", and a
positive control delta is only `demonstrated` when its CI lower bound > 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CapabilityScore:
    domain: str
    treatment_score: float
    ci_lo: float
    ci_hi: float
    control_score: float
    control_delta: float
    demonstrated: bool
    refusal_rate: float
    measured: bool
    n: int
    seed: int
    bank_hash: str
    scorer_calibration: dict = field(default_factory=dict)
    elicitation: dict | None = None  # default vs best-effort (feeds G8E) — None at M0

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "treatmentScore": round(self.treatment_score, 6),
            "ci": [round(self.ci_lo, 6), round(self.ci_hi, 6)],
            "controlScore": round(self.control_score, 6),
            "controlDelta": round(self.control_delta, 6),
            "demonstrated": self.demonstrated,
            "refusalRate": round(self.refusal_rate, 6),
            "measured": self.measured,
            "n": self.n,
            "seed": self.seed,
            "bankHash": self.bank_hash,
            "scorerCalibration": self.scorer_calibration,
            "elicitation": self.elicitation,
        }


def assemble_report(scores: "list[CapabilityScore]", *, mode: str,
                    harness_version: str = "0.1.0") -> dict:
    """Assemble per-domain CapabilityScores into one provenanced, fail-closed report."""
    return {
        "benchmark": "frontier-capability",
        "mode": mode,
        "visibility": "public-aggregate",
        "harnessVersion": harness_version,
        "posture": "defensive-only; measures and refuses dangerous capability; never "
                   "produces uplift; category-level; responsible disclosure",
        "claimStatus": "Measurement machinery only. Scores are LOWER BOUNDS under the "
                       "elicitation mounted; a stronger elicitation could reveal more. No "
                       "score is reported as 'safe' — only 'measured below/above threshold "
                       "under stated conditions'. Unmeasured => quarantine (fail-closed).",
        "candidateOnly": True,
        "level3Evidence": False,
        "canClaimAGI": False,
        "domains": {s.domain: s.to_dict() for s in scores},
    }
