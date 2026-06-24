"""Skill: self_extend_probe — offline self-extending-flywheel coverage probe.

Honest scope: this runs the OFFLINE flywheel (synthesize -> validate on a held-out
split -> coverage) on a small deterministic domain. It is a *candidate/offline*
mechanism demo, NOT a capability claim — results carry candidateOnly/level3Evidence.
"""
from __future__ import annotations

from skills.core import sophia_skill

# default deterministic domain: (text, is_correct) pairs, stratified (>=2 of each class)
_DEMO_DOMAIN = {
    "arithmetic": [("2+2=4", True), ("3+1=4", True), ("2+2=5", False), ("9-1=7", False)],
}


@sophia_skill(
    "self_extend_probe",
    summary="Offline self-extension probe: synthesize+validate a verifier on a held-out split and report coverage. Candidate/offline, not a capability claim.",
    uses=("selfextend.flywheel.run_flywheel",),
)
def self_extend_probe(*, domain: dict | None = None, threshold: float = 0.8) -> dict:
    from selfextend.flywheel import run_flywheel  # imported lazily; fail-closed if unavailable

    res = run_flywheel(domain or _DEMO_DOMAIN, threshold=threshold)
    return {
        "verdict": "candidate",
        "candidateOnly": True,
        "level3Evidence": False,
        "detail": res,
    }
