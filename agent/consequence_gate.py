# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""ConsequenceGate — the 8th conscience path.

Sophia's OKF belief graph is already a GO board in the sense Tom names: a node
that other claims ``derivesFrom`` is a stone with territory, and retracting it
can flip (orphan) every claim transitively grounded in it. The cascade + abstain
machinery already exists — ``okf.revision.revise`` places the stone,
``okf.revision.claims_to_abstain`` is the captured group, ``okf.counterfactual.
reduced_without`` is the contracted board. This module is the named, audited
runtime consumer of that operation: it does not invent new graph primitives; it
*exposes* the consequence of a retraction as a conscience verdict and a
retractable-edge audit record.

Honest scope (mirrors ``agent.verification_mcts``): this reasons over the
deterministic, already-grounded ``derivesFrom`` graph. It invents no facts. The
abstain set it reports is the set of *existing* claims that lose their
provenance ground — derivations, not predictions about a future world state.
A candidate "move" (retraction target) that does not resolve to a node cannot be
bounded, so the gate fails closed: ``abstain``, never a silent no-op (the same
invariant as ``okf.revision.revise`` for unknown targets).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from okf.counterfactual import counterfactual_remove
from okf.graph import Graph, resolve
from okf.revision import revise

# Conservative hand-pick default; the live value is loaded from
# ``config/consequence.json`` (tunable without code change — TODO: tune against a
# real benchmark pack). Kept as a module constant so the safe default is visible
# and importable even if the config file is absent/invalid.
FLIP_SEVERITY_ESCALATE = 0.15
_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "consequence.json"


def _load_consequence_config() -> dict[str, Any]:
    """Load consequence params from ``config/consequence.json``.

    Fail-safe: if the file is missing OR malformed in ANY way, fall back to the
    conservative module defaults. A malformed config must NEVER raise into the gate
    path — raising would crash import-time initialization and weaken the gate by
    failing open via exception. So we catch the full universe of malformed inputs:
    missing/unreadable file (OSError), bad UTF-8 (UnicodeDecodeError), syntactically
    invalid JSON (json.JSONDecodeError), valid JSON that is not an object
    (raw.get AttributeError on a list/str/null), or a value that is not a real
    number (float() ValueError/TypeError, incl. NaN/Inf which float() accepts but
    would silently break the threshold comparison). Any of these -> module default.
    """
    try:
        raw = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        value = float(raw["flipSeverityEscalate"])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, KeyError, ValueError, AttributeError):
        return {"flipSeverityEscalate": FLIP_SEVERITY_ESCALATE}
    # float("nan")/float("inf") parse but make the >= threshold meaningless; a
    # non-finite/non-normal threshold is malformed for our purposes -> default.
    import math
    if not math.isfinite(value) or not (0.0 <= value <= 1.0):
        return {"flipSeverityEscalate": FLIP_SEVERITY_ESCALATE}
    return {"flipSeverityEscalate": value}


_CONFIG = _load_consequence_config()
# |abstain set| / |graph| at or above which the gate forces ``escalate`` rather
# than ``allow``. A retraction that would orphan >= this fraction of the belief
# set is not a routine consequence — it needs stronger process before commitment.
flip_severity_escalate = float(_CONFIG["flipSeverityEscalate"])


@dataclass(frozen=True)
class ConsequenceReport:
    """Result of simulating the consequence of placing one retraction "stone".

    The headline fields a runtime gate consults:
    - ``abstainSet``: claims that lose their provenance ground (fail-closed set).
    - ``flipSeverity``: |abstainSet| / |graph| — the structural magnitude of the
      flip. Deterministic; no learned prior.
    - ``verdict``: a *subset* of the conscience verdict vocabulary
      (allow|escalate|abstain) so the kernel's 7-value contract is unchanged.
    - ``audit``: the retractable-edge record (one entry per retraction + cascade).
    """

    schema: str = "sophia.consequence.v1"
    candidateMove: str = ""
    targetId: "str | None" = None
    found: bool = True
    abstainSet: tuple[str, ...] = ()
    flipSeverity: float = 0.0
    verdict: str = "allow"  # allow|escalate|abstain
    reason: str = "no downstream provenance loss"
    detail: dict[str, Any] = field(default_factory=dict)
    audit: tuple[dict[str, Any], ...] = ()
    candidateOnly: bool = True
    level3Evidence: bool = False
    boundary: str = (
        "consequence report is a deterministic derivation over the current OKF "
        "graph; it is not a prediction of a future world state and not AGI proof."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "candidateMove": self.candidateMove,
            "targetId": self.targetId,
            "found": self.found,
            "abstainSet": list(self.abstainSet),
            "abstainCount": len(self.abstainSet),
            "flipSeverity": self.flipSeverity,
            "verdict": self.verdict,
            "reason": self.reason,
            "detail": self.detail,
            "audit": list(self.audit),
            "candidateOnly": self.candidateOnly,
            "level3Evidence": self.level3Evidence,
            "boundary": self.boundary,
        }


def simulate_cascade(
    graph: Graph, move: str, *, by: str = "consequence_gate"
) -> ConsequenceReport:
    """Place ``move`` as a retraction stone on the OKF board; compute the cascade.

    Pure, non-destructive: builds a reduced graph view and reports what would
    lose support — it never mutates ``graph`` or writes a page. If ``move`` does
    not resolve to a node, the consequence is unbounded, so the gate returns
    ``abstain`` (fail-closed), matching ``okf.revision.revise``'s treatment of
    unknown retractions.

    ``flipSeverity`` is structural magnitude only (|abstain|/|graph|). It carries
    no learned/empirical prior — that is a deliberate Variant-A scope boundary;
    adding a provenance-scored empirical prior is a Medium extension that must
    itself pass the fact gate per-sample.
    """
    rid = resolve(graph, move)
    if rid is None:
        return ConsequenceReport(
            candidateMove=move,
            targetId=None,
            found=False,
            verdict="abstain",
            reason=f"unresolved target {move!r}: consequence cannot be bounded",
        )

    rev = revise(graph, [(rid, "consequence_gate_probe")], by=by)
    abstain = tuple(rev.abstain)
    n = max(1, len(graph.nodes))
    severity = len(abstain) / n
    escalate = severity >= flip_severity_escalate
    verdict = "escalate" if escalate else "allow"
    reason = (
        "downstream provenance loss exceeds flip-severity threshold"
        if escalate
        else "bounded downstream provenance loss"
    )
    # The counterfactual_remove report is richer (per-claim before/after); keep it
    # as detail for callers that want the full flip ledger.
    detail = counterfactual_remove(graph, rid)
    return ConsequenceReport(
        candidateMove=move,
        targetId=rid,
        found=True,
        abstainSet=abstain,
        flipSeverity=round(severity, 4),
        verdict=verdict,
        reason=reason,
        detail=detail,
        audit=tuple(rev.audit_log()),
    )


__all__ = ["FLIP_SEVERITY_ESCALATE", "flip_severity_escalate", "ConsequenceReport", "simulate_cascade"]
