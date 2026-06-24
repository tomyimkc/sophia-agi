"""Constitutional via-negativa gate for Sophia.

The constitution is external policy, not model preference. This v1 implements
machine-checkable negative rules plus positive duties in a deterministic/offline
form. It intentionally complements, not replaces, provenance and deontic gates.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# Prefer constitution v2 (adds publicStandardLinks + two distinct moral theories)
# when present; fall back to v1 so older checkouts keep working. v2 is a strict
# superset of the v1 prohibitions/duties shapes, so the gate logic is unchanged.
_CONSTITUTION_V2 = ROOT / "constitution" / "constitution.v2.json"
_CONSTITUTION_V1 = ROOT / "constitution" / "constitution.v1.json"
DEFAULT_CONSTITUTION = _CONSTITUTION_V2 if _CONSTITUTION_V2.exists() else _CONSTITUTION_V1
SAFE_AGI_BOUNDARY_RE = re.compile(r"\b(?:not\s+proven\s+AGI|AGI-candidate|candidate\s+proof|not\s+claimed\s+as\s+proven\s+AGI|canClaimAGI\s*[=:]\s*false)\b", re.I)
AGI_OVERCLAIM_RE = re.compile(r"\b(?:Sophia\s+(?:is|has\s+achieved|is\s+proven)\s+(?:an?\s+)?AGI|proven\s+AGI|achieved\s+AGI|certified\s+AGI)\b", re.I)


@dataclass(frozen=True)
class ConstitutionalFinding:
    rule_id: str
    severity: str
    verdict: str
    reason: str
    matched: str = ""


@dataclass(frozen=True)
class ConstitutionalDecision:
    schema: str = "sophia.constitutional_gate.v1"
    verdict: str = "accepted"  # accepted|held|rejected
    reason: str = "no constitutional issue detected"
    findings: tuple[ConstitutionalFinding, ...] = ()
    duties: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "verdict": self.verdict,
            "reason": self.reason,
            "findings": [f.__dict__ for f in self.findings],
            "duties": list(self.duties),
        }


def load_constitution(path: str | Path = DEFAULT_CONSTITUTION) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_constitution(text: str, *, constitution: dict[str, Any] | None = None, context: dict[str, Any] | None = None) -> ConstitutionalDecision:
    constitution = constitution or load_constitution()
    context = context or {}
    txt = text or ""
    findings: list[ConstitutionalFinding] = []

    if AGI_OVERCLAIM_RE.search(txt) and not SAFE_AGI_BOUNDARY_RE.search(txt) and context.get("canClaimAGI") is not True:
        findings.append(ConstitutionalFinding("no_agi_overclaim", "critical", "rejected", "AGI status overclaim while canClaimAGI is not true", AGI_OVERCLAIM_RE.search(txt).group(0)))

    low = txt.lower()
    for rule in constitution.get("prohibitions", []):
        rid = str(rule.get("id", "unknown"))
        if rid == "no_agi_overclaim" and (any(f.rule_id == rid for f in findings) or SAFE_AGI_BOUNDARY_RE.search(txt)):
            continue
        for pat in rule.get("patterns", []):
            if str(pat).lower() in low:
                findings.append(ConstitutionalFinding(rid, str(rule.get("severity", "high")), "rejected", str(rule.get("rule", "prohibited by constitution")), str(pat)))
                break

    if findings:
        verdict = "rejected" if any(f.severity == "critical" for f in findings) else "held"
        return ConstitutionalDecision(verdict=verdict, reason="constitutional prohibition triggered", findings=tuple(findings), duties=tuple(d.get("id", "") for d in constitution.get("duties", [])))
    return ConstitutionalDecision(duties=tuple(d.get("id", "") for d in constitution.get("duties", [])))


__all__ = ["ConstitutionalFinding", "ConstitutionalDecision", "load_constitution", "check_constitution"]
