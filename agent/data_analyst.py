# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Data Analysis Agent — the curator role that closes the data loop.

This is the agent called for in ``docs/11-Platform/Data-Analysis-Agent-Strategy.md``
(Phase 5): one role that *audits corpus health, scores data-management maturity,
surfaces contamination, and proposes a prioritised curation plan* — instead of those
decisions being scattered across ~30 tools and made ad-hoc.

Sophia discipline (matches every other ``agent/*`` module):

  * **deterministic + offline** — pure analysis over committed artifacts via
    ``tools/data_health_report`` (the DHI) and ``tools/assert_entity_decontam``
    (entity contamination). No GPU, no network, CI-testable.
  * **propose-only / fail-closed** — the analyst NEVER mutates data. It (a) reports,
    (b) proposes a plan; acting happens only through the existing *gated* builders
    (spark_data_refinery, build_*), with humans approving curation (Leiden value 2).
    If an analysis tool or a required artifact is missing it **refuses to emit a
    verdict** rather than guessing (status ``"refused"``).
  * **honest scope** — the DHI is operational/illustrative, never a no-overclaim
    result; ``canClaimAGI`` stays false.

It is also the ``"data"`` team of :mod:`agent.swarm_router` (least-privilege), so a
data/corpus/decontamination task fans out to this analyst like any other expert.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCHEMA = "sophia.data_analyst.v1"
PROPOSE_ONLY = True          # load-bearing: the analyst never mutates data
SCORE_FLOOR = 0.80           # dimensions below this earn a curation action
TARGET = 1.0

# Recommended action per weak dimension (drawn from the strategy doc's phases).
# (recommendation, the gated tool/loop that DOES the work — never the analyst itself).
RECOMMENDATIONS: dict[str, tuple[str, str]] = {
    "coverage": (
        "Expand structured ground-truth records (human-verified) and generate "
        "gate-filtered synthesis; track records (coverage) separately from rows "
        "(volume) so templating never masquerades as coverage.",
        "tools/spark_data_refinery.py + structured-record curation",
    ),
    "mixBalance": (
        "Build toward the target mix: lift the starved families (hk_bilingual, "
        "moral_gate, tool_mcp) and the Chinese share; gate mix drift in CI.",
        "tools/build_*_sft.py + a mix-balance regression gate",
    ),
    "decontamStrength": (
        "Wire the missing decontamination layer (exact → shingle → entity).",
        "tools/assert_entity_decontam.py",
    ),
    "dedupHealth": (
        "Pin dedup/quality thresholds into the manifest so dedup decisions are "
        "reproducible across re-runs.",
        "pipeline/dedup + manifest threshold stamping",
    ),
    "provenanceCompleteness": (
        "Backfill per-row provenance (esp. license) via data passports; aim ≥0.99.",
        "pretraining/data_passport/build_passport.py",
    ),
    "lineage": (
        "Build/refresh the data asset registry so every corpus resolves to a "
        "source→checkpoint→eval anchor.",
        "tools/build_data_registry.py",
    ),
    "reproducibility": (
        "Stamp a content hash into every data manifest and make each artifact "
        "--verify-able.",
        "tools/build_data_registry.py + per-artifact --verify",
    ),
}


@dataclass(frozen=True)
class CurationAction:
    dimension: str
    score: float
    weight: float
    priority: float          # weight * gap — biggest measured leverage first
    recommendation: str
    tool: str

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "score": round(self.score, 4),
            "weight": round(self.weight, 4),
            "priority": round(self.priority, 4),
            "recommendation": self.recommendation,
            "tool": self.tool,
        }


def _refused(reason: str) -> dict:
    return {"status": "refused", "schema": SCHEMA, "reason": reason}


class DataAnalyst:
    """Deterministic, propose-only data curator. See module docstring."""

    def __init__(self, root: Path = ROOT) -> None:
        self.root = root

    # --- audit -------------------------------------------------------------
    def assess(self) -> dict:
        """Run the DHI + entity-contamination audits. Fail-closed on missing tools/data."""
        try:
            from tools import data_health_report as dhr
            from tools import assert_entity_decontam as aed
        except Exception as exc:                       # pragma: no cover - import guard
            return _refused(f"analysis tools unavailable: {exc}")
        if not dhr.MANIFEST.exists():
            return _refused(f"corpus manifest missing: {dhr.MANIFEST}")

        report = dhr.compute_report()
        try:
            entity = aed.audit()
        except Exception as exc:                       # pragma: no cover - defensive
            return _refused(f"entity audit failed: {exc}")

        return {
            "status": "ok",
            "schema": SCHEMA,
            "canClaimAGI": False,
            "proposeOnly": PROPOSE_ONLY,
            "dhi": report["dhi"],
            "weights": report["weights"],
            "dimensions": {k: report["dimensions"][k]["score"] for k in report["dimensions"]},
            "dimensionDetail": report["dimensions"],
            "entityContamination": {
                "nSharedEntities": entity["nSharedEntities"],
                "nEvalPromptsFullyCovered": entity["nEvalPromptsFullyCovered"],
                "sharedEntities": entity["sharedEntities"],
            },
        }

    # --- plan --------------------------------------------------------------
    def curation_plan(self, assessment: dict | None = None) -> dict:
        """Prioritised, propose-only curation plan from the weakest dimensions."""
        a = assessment if assessment is not None else self.assess()
        if a.get("status") != "ok":
            return a

        weights = a["weights"]
        actions: list[CurationAction] = []
        for dim, score in a["dimensions"].items():
            if score >= SCORE_FLOOR:
                continue
            weight = float(weights.get(dim, 0.0))
            gap = TARGET - float(score)
            rec, tool = RECOMMENDATIONS.get(dim, ("Investigate and improve.", "n/a"))
            actions.append(CurationAction(dim, float(score), weight, weight * gap, rec, tool))

        # entity contamination is a first-class action when present
        ec = a["entityContamination"]
        if ec["nEvalPromptsFullyCovered"] > 0:
            actions.append(CurationAction(
                "entityDisjointSplit",
                0.0, 0.0,
                # rank it alongside the strongest dimension action
                priority=max([x.priority for x in actions], default=0.0) + 1e-6,
                recommendation=(
                    f"{ec['nEvalPromptsFullyCovered']} eval prompts are fully entity-covered by "
                    "training — carve a genuinely entity-disjoint held-out split (human-approved)."
                ),
                tool="tools/carve_entity_disjoint_split.py --out <path>",
            ))

        actions.sort(key=lambda x: x.priority, reverse=True)
        return {
            "status": "ok",
            "schema": SCHEMA,
            "canClaimAGI": False,
            "proposeOnly": PROPOSE_ONLY,
            "dhi": a["dhi"],
            "nActions": len(actions),
            "topPriority": actions[0].dimension if actions else None,
            "actions": [x.to_dict() for x in actions],
        }

    def report(self) -> dict:
        """assess() + curation_plan() in one fail-closed object."""
        a = self.assess()
        if a.get("status") != "ok":
            return a
        plan = self.curation_plan(a)
        return {"status": "ok", "schema": SCHEMA, "assessment": a, "plan": plan}


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Data Analysis Agent — audit + propose-only curation plan")
    ap.add_argument("--json", action="store_true", help="emit the full machine-readable report")
    args = ap.parse_args(argv)

    rep = DataAnalyst().report()
    if rep.get("status") != "ok":
        print(f"DATA ANALYST: REFUSED — {rep.get('reason')}")
        return 1
    if args.json:
        print(json.dumps(rep, indent=2, ensure_ascii=False))
    else:
        plan = rep["plan"]
        print(f"DATA ANALYST: DHI={rep['assessment']['dhi']}  topPriority={plan['topPriority']}  "
              f"actions={plan['nActions']} (propose-only)")
        for act in plan["actions"]:
            print(f"  [{act['priority']:.4f}] {act['dimension']:22s} score={act['score']} -> {act['tool']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
