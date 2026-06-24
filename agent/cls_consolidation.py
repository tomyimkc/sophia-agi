"""Complementary Learning Systems: consolidate stable wiki knowledge into weights —
but only through the anti-forgetting plasticity gate.

Brain-inspired continual learning splits a fast, episodic store (hippocampus) from a
slow, consolidated store (neocortex). Map it onto Sophia:

    wiki / OKF graph  =  hippocampus  — every new fact lands here first, instantly,
                                         with zero risk of forgetting (Experiments 1–3)
    model weights     =  neocortex    — slow; only *stable, gate-cleared* knowledge is
                                         ever distilled into it

This module is the consolidation selector. It replays the retention snapshots from
``agent.continual_retention`` to find facts that have stayed grounded and un-weakened
for ``min_stable_snapshots`` in a row, keeps only those that also cleared the provenance
gate, assembles them into an ``UpdateCandidate``, and routes it through the existing
``agent.continual_plasticity`` promotion gate. That gate's protected suites
(``source_discipline``, ``fact_check_false_accept``) are the catastrophic-forgetting
tripwire: any adapter that regresses old knowledge is **rejected**, never promoted.

Distillation/training itself is out of scope here (and out of CI): candidates are
``level3Evidence: false`` until a real run clears the full no-overclaim gate.
"""

from __future__ import annotations

from typing import Any

from agent.continual_plasticity import EvalMetric, UpdateCandidate, evaluate_update
from agent.continual_retention import _origin_confidence

# Suites that must never regress when we move knowledge into weights — the
# anti-forgetting invariant of the whole approach.
PROTECTED_SUITES = ("source_discipline", "fact_check_false_accept")


def stability_streaks(snapshots) -> "dict[str, int]":
    """For each fact, the number of trailing consecutive snapshots in which it stayed
    grounded with confidence at or above its origin confidence.

    A streak of *k* means the fact has been stable across the last *k* learning steps —
    the hippocampal-replay signal that it is safe to consolidate.
    """
    if not snapshots:
        return {}
    origin = _origin_confidence(snapshots)
    streaks: dict[str, int] = {}
    for fid in origin:
        streak = 0
        for snap in reversed(snapshots):
            state = snap.grounded
            if fid in state and state[fid] >= origin[fid]:
                streak += 1
            else:
                break
        streaks[fid] = streak
    return streaks


def select_consolidation_set(streaks: "dict[str, int]", gate_cleared, *, min_stable_snapshots: int = 3) -> "list[str]":
    """Facts eligible to consolidate: stable for >= N snapshots AND gate-cleared."""
    cleared = set(gate_cleared)
    return sorted(fid for fid, k in streaks.items() if k >= min_stable_snapshots and fid in cleared)


def build_candidate(selected, metrics, *, candidate_id: str, verifier_artifacts=()) -> UpdateCandidate:
    """Assemble an UpdateCandidate from a selected fact set and its eval metrics.

    ``metrics`` is a list of (suite, before, after) triples; the protected suites are
    tagged automatically so the plasticity gate enforces the anti-forgetting invariant.
    """
    rows = tuple(
        EvalMetric(suite=s, before=float(b), after=float(a), protected=(s in PROTECTED_SUITES))
        for (s, b, a) in metrics
    )
    return UpdateCandidate(
        id=candidate_id,
        kind="lora_adapter",
        metrics=rows,
        verifier_artifacts=tuple(verifier_artifacts),
        notes=f"CLS consolidation of {len(list(selected))} stable wiki facts",
    )


def consolidate(snapshots, gate_cleared, metrics, *, target_suite: str, candidate_id: str = "cls_consolidation_v1",
                verifier_artifacts=(), min_stable_snapshots: int = 3,
                max_protected_regression: float = 0.01) -> "dict[str, Any]":
    """Select stable, gate-cleared wiki facts and route a distillation candidate through
    the anti-forgetting plasticity gate.

    Returns a report with the selected set and the promotion decision. If nothing is
    stable enough yet, no candidate is built (``decision`` is None) — consolidation
    waits, the wiki keeps serving the knowledge non-parametrically in the meantime.
    """
    streaks = stability_streaks(snapshots)
    selected = select_consolidation_set(streaks, gate_cleared, min_stable_snapshots=min_stable_snapshots)

    decision = None
    if selected:
        candidate = build_candidate(selected, metrics, candidate_id=candidate_id, verifier_artifacts=verifier_artifacts)
        decision = evaluate_update(candidate, target_suite=target_suite,
                                   max_protected_regression=max_protected_regression).to_dict()

    return {
        "schema": "sophia.cls_consolidation.v1",
        "level3Evidence": False,
        "stabilityStreaks": streaks,
        "minStableSnapshots": min_stable_snapshots,
        "selected": selected,
        "selectedCount": len(selected),
        "protectedSuites": list(PROTECTED_SUITES),
        "decision": decision,
        # The headline invariant: weights can only ever gain knowledge the gate proves
        # does not cost old knowledge.
        "antiForgettingEnforced": decision is None or decision["verdict"] != "promote"
        or all(
            row["delta"] >= -max_protected_regression
            for row in decision["metrics"]["metricRows"]
            if row.get("protected")
        ),
    }


__all__ = ["PROTECTED_SUITES", "stability_streaks", "select_consolidation_set", "build_candidate", "consolidate"]
