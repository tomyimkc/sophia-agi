# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Per-domain, empirically-calibrated competence — measured weakness, not guesses.

The knowledge-gap loop asks "what should the corpus learn next?". Answering that
honestly requires knowing *where the agent is actually weak*, and weakness must be
read off observed behaviour, never hand-assigned. This module consumes graded
outcome records — ``(domain, confidence, correct)`` triples produced by an external
oracle — and, per OKF domain, computes the same calibration evidence the rest of the
Conscience Kernel trusts:

  - ECE (are stated confidences honest?) via :mod:`agent.calibration`,
  - AURC and selective-vs-base risk (does the agent know what it doesn't know?),
  - a split-conformal abstention threshold via :mod:`agent.conformal_gate`.

From that evidence it derives one scalar competence in ``[0, 1]`` per domain that
*rewards* calibration and selective accuracy and *penalises* error and
miscalibration (see :func:`_competence_score` for the exact formula). Domains never
observed fail closed: lowest competence, most conservative (zero) answer threshold —
the agent does not claim skill it has no evidence for.

:func:`learning_priorities` then ranks domains by a measured *deficit* (worst
competence, highest base risk, highest ECE first). That ranked worklist is the
bridge into :mod:`agent.knowledge_gap_log`: "learn next where the data says you are
weakest", not where someone guessed.

    from agent.competence_model import build_competence_model, learning_priorities
    model = build_competence_model(records, alpha=0.1, coverage=0.5)
    model.competence("history"); model.threshold("history")
    learning_priorities(model)  # measured-weakness worklist
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from agent.calibration import calibration_report
from agent.conformal_gate import fit_conformal_policy
from okf.schema import DOMAINS

__all__ = [
    "reliability_diagram",
    "DomainCompetence",
    "CompetenceModel",
    "build_competence_model",
    "learning_priorities",
    "competence_gap_worklist",
]


# --------------------------------------------------------------------------- #
# Record helpers
# --------------------------------------------------------------------------- #
def _conf(rec: dict[str, Any]) -> float:
    """Clamp a record's confidence into ``[0, 1]``."""
    return min(max(float(rec["confidence"]), 0.0), 1.0)


def _nonconformity(rec: dict[str, Any]) -> float:
    """Nonconformity = explicit field, else ``1 - confidence`` (less trust = higher)."""
    if rec.get("nonconformity") is not None:
        return float(rec["nonconformity"])
    return 1.0 - _conf(rec)


def reliability_diagram(records: Sequence[dict[str, Any]], n_bins: int = 10) -> list[dict]:
    """The literal data to plot a reliability diagram, across all supplied records.

    Buckets records by confidence into ``n_bins`` equal-width bins over ``[0, 1]``;
    each non-empty bin reports its confidence range, the mean stated confidence, the
    observed accuracy, and the count. Empty bins are omitted. Deterministic: bins are
    emitted in ascending confidence order and depend only on the inputs.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    bins: dict[int, list[tuple[float, bool]]] = {}
    for rec in records:
        c = _conf(rec)
        b = min(n_bins - 1, int(c * n_bins))
        bins.setdefault(b, []).append((c, bool(rec["correct"])))
    out: list[dict] = []
    for b in sorted(bins):
        items = bins[b]
        count = len(items)
        out.append({
            "binLo": round(b / n_bins, 4),
            "binHi": round((b + 1) / n_bins, 4),
            "meanConfidence": round(sum(c for c, _ in items) / count, 4),
            "accuracy": round(sum(1 for _, ok in items if ok) / count, 4),
            "count": count,
        })
    return out


# --------------------------------------------------------------------------- #
# Competence score
# --------------------------------------------------------------------------- #
def _competence_score(report: dict[str, Any]) -> float:
    """Single competence scalar in ``[0, 1]`` derived from calibration evidence.

    Definition (documented and deterministic):

        accuracy        = 1 - baseRisk          # how often answers are correct
        selectiveSkill  = 1 - selectiveRisk      # correctness among confident answers
        calibration     = 1 - ece                # honesty of stated confidence (clamped >=0)

        competence = 0.40 * accuracy
                   + 0.35 * selectiveSkill
                   + 0.25 * calibration

    The weights make raw accuracy the largest term, reward "knowing what you don't
    know" (selective accuracy at the chosen coverage) almost as much, and dock points
    for miscalibration. An overconfident-but-sometimes-right domain loses on both the
    calibration term and (because confidence is uninformative) the selective term,
    scoring below a well-calibrated peer with the same accuracy. Result clamped to
    ``[0, 1]``.
    """
    accuracy = 1.0 - float(report["baseRisk"])
    selective = 1.0 - float(report["selectiveRisk"])
    calibration = max(0.0, 1.0 - float(report["ece"]))
    score = 0.40 * accuracy + 0.35 * selective + 0.25 * calibration
    return round(min(1.0, max(0.0, score)), 4)


@dataclass(frozen=True)
class DomainCompetence:
    """Empirical competence evidence for one domain. Immutable."""

    domain: str
    n: int
    ece: float
    aurc: float
    baseRisk: float
    selectiveRisk: float
    selectiveBeatsBase: bool
    coverage: float
    threshold: float
    targetCoverage: float
    nCalibration: int
    competence: float
    seen: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "n": self.n,
            "ece": self.ece,
            "aurc": self.aurc,
            "baseRisk": self.baseRisk,
            "selectiveRisk": self.selectiveRisk,
            "selectiveBeatsBase": self.selectiveBeatsBase,
            "coverage": self.coverage,
            "threshold": self.threshold,
            "targetCoverage": self.targetCoverage,
            "nCalibration": self.nCalibration,
            "competence": self.competence,
            "seen": self.seen,
        }


# Fail-closed defaults for an unseen domain: no evidence => lowest competence and the
# most conservative answer threshold (0.0 nonconformity => answer ~nothing).
def _unseen_domain(domain: str, *, alpha: float, coverage: float) -> DomainCompetence:
    return DomainCompetence(
        domain=domain,
        n=0,
        ece=1.0,
        aurc=1.0,
        baseRisk=1.0,
        selectiveRisk=1.0,
        selectiveBeatsBase=False,
        coverage=coverage,
        threshold=0.0,
        targetCoverage=round(1.0 - alpha, 4),
        nCalibration=0,
        competence=0.0,
        seen=False,
    )


@dataclass(frozen=True)
class CompetenceModel:
    """Per-domain competence map with fail-closed lookups."""

    domains: dict[str, DomainCompetence]
    alpha: float
    coverage: float
    _fallback: dict[str, DomainCompetence] = field(default_factory=dict, compare=False)

    def _get(self, domain: str) -> DomainCompetence:
        if domain in self.domains:
            return self.domains[domain]
        # Fail closed: cache a conservative record so repeated lookups are stable.
        if domain not in self._fallback:
            self._fallback[domain] = _unseen_domain(domain, alpha=self.alpha, coverage=self.coverage)
        return self._fallback[domain]

    def competence(self, domain: str) -> float:
        """Competence in ``[0, 1]``. Unseen domains => 0.0 (fail closed)."""
        return self._get(domain).competence

    def threshold(self, domain: str) -> float:
        """Conformal answer threshold (nonconformity). Unseen => 0.0 (most conservative)."""
        return self._get(domain).threshold

    def get(self, domain: str) -> DomainCompetence:
        """Full evidence record for a domain (fail-closed if unseen)."""
        return self._get(domain)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "sophia.competence_model.v1",
            "candidateOnly": True,
            "alpha": self.alpha,
            "coverage": self.coverage,
            "domains": {d: dc.to_dict() for d, dc in sorted(self.domains.items())},
        }


def build_competence_model(
    records: Sequence[dict[str, Any]], *, alpha: float = 0.1, coverage: float = 0.5,
) -> CompetenceModel:
    """Build a per-domain competence model from observed graded outcomes.

    Each record is ``{"domain", "confidence" in [0,1], "correct"}`` (optionally
    ``"nonconformity"``). For every domain with at least one record we compute the
    calibration report (ECE/AURC/base & selective risk) and fit a split-conformal
    answer threshold from the in-domain rows, then derive a single competence score.
    Domains with no records are absent here and resolved fail-closed on lookup.
    """
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_domain.setdefault(str(rec["domain"]), []).append(rec)

    domains: dict[str, DomainCompetence] = {}
    for domain, recs in by_domain.items():
        confidences = [_conf(r) for r in recs]
        correct = [bool(r["correct"]) for r in recs]
        report = calibration_report(confidences, correct, coverage=coverage)

        conformal_rows = [
            {"nonconformity": _nonconformity(r), "correct": bool(r["correct"]), "risk": "normal"}
            for r in recs
        ]
        policy = fit_conformal_policy(conformal_rows, alpha=alpha, risk_bucket="normal")

        domains[domain] = DomainCompetence(
            domain=domain,
            n=report["n"],
            ece=report["ece"],
            aurc=report["aurc"],
            baseRisk=report["baseRisk"],
            selectiveRisk=report["selectiveRisk"],
            selectiveBeatsBase=bool(report["selectiveBeatsBase"]),
            coverage=coverage,
            threshold=policy.threshold,
            targetCoverage=policy.target_coverage,
            nCalibration=policy.n_calibration,
            competence=_competence_score(report),
        )

    return CompetenceModel(domains=domains, alpha=alpha, coverage=coverage)


# --------------------------------------------------------------------------- #
# Measured-weakness worklist
# --------------------------------------------------------------------------- #
def _deficit(dc: DomainCompetence) -> float:
    """Deficit score in ``[0, 1]``: high = weak. Combines low competence, high base
    risk, and high miscalibration. Mirrors the competence weighting so it is the
    measured complement of skill, not an independent heuristic."""
    deficit = (
        0.40 * (1.0 - dc.competence)
        + 0.35 * dc.baseRisk
        + 0.25 * min(1.0, dc.ece)
    )
    return round(min(1.0, max(0.0, deficit)), 4)


def _reasons(dc: DomainCompetence) -> list[str]:
    reasons: list[str] = []
    if not dc.seen:
        reasons.append("unseen-domain-fail-closed")
        return reasons
    if dc.competence < 0.5:
        reasons.append("low-competence")
    if dc.baseRisk > 0.3:
        reasons.append("high-base-risk")
    if dc.ece > 0.1:
        reasons.append("miscalibrated")
    if not dc.selectiveBeatsBase:
        reasons.append("selective-no-better-than-base")
    if dc.n < 10:
        reasons.append("thin-evidence")
    if not reasons:
        reasons.append("adequate")
    return reasons


def learning_priorities(model: CompetenceModel) -> list[dict]:
    """Rank domains by measured deficit — the "what to learn next" worklist.

    Weakest first: highest deficit (low competence / high base risk / high ECE).
    Ties broken by lower competence, then higher base risk, then higher ECE, then
    domain name for determinism. Each item is
    ``{"domain","competence","deficit","reasons":[...]}``. Driven entirely by observed
    records — no hand-tuned domain weights.
    """
    items = []
    for domain, dc in model.domains.items():
        items.append({
            "domain": domain,
            "competence": dc.competence,
            "deficit": _deficit(dc),
            "baseRisk": dc.baseRisk,
            "ece": dc.ece,
            "n": dc.n,
            "reasons": _reasons(dc),
        })
    items.sort(
        key=lambda x: (-x["deficit"], x["competence"], -x["baseRisk"], -x["ece"], x["domain"]),
    )
    return items


def competence_gap_worklist(model: CompetenceModel, thin_targets: Sequence[str] = ()) -> dict[str, Any]:
    """Bridge competence deficits into the knowledge-gap worklist shape.

    Reuses :func:`agent.knowledge_gap_log.gap_worklist` defensively (import-guarded so
    a change there cannot break this module). The weakest domains by measured deficit
    become enrichment targets, seeded alongside any audit-thin targets.
    """
    priorities = learning_priorities(model)
    try:
        from agent.knowledge_gap_log import gap_worklist
    except Exception:  # pragma: no cover - defensive: never break on gap-log changes
        gap_worklist = None

    base: dict[str, Any]
    if gap_worklist is not None:
        try:
            base = dict(gap_worklist([], thin_targets=thin_targets))
        except Exception:  # pragma: no cover
            base = {}
    else:
        base = {}

    base.update({
        "schema": "sophia.competence_gap_worklist.v1",
        "candidateOnly": True,
        "competencePriorities": priorities,
    })
    return base
