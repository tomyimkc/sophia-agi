# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Ko-guarded iterative belief revision — the multi-step consumer of ``okf.revise``.

A single ``okf.revision.revise`` call is ko-safe: it batches all retractions and
computes the cascade in one consistent pass. The ko surface (GO-rule cycle
detection) only appears once a caller ITERATES: place retraction A, observe the
abstain set, then reassert or retract something else in response, then maybe go
back. A sequence that revisits a prior abstain state is a ko — an irreducible
oscillation that cannot terminate without new information.

This module is that iterative consumer. It drives ``okf.revise`` round by round
over a caller-supplied retraction schedule, accumulates the abstain-set sequence
(the ko "belief state"), and runs ``ko_detector.detect_ko`` after each round. On a
ko it terminates with verdict ``escalate`` (NEVER ``abstain`` — see the
ko_detector module docstring for the load-bearing distinction: a ko needs a human
or a new source to break the loop, not a silent drop).

Design notes (grounded in the verified ``okf.revise`` contract):

- ``revise`` is non-destructive and rebuilds ``reduced_without(graph, removed)``
  fresh each call. So the loop re-calls ``revise`` on the ORIGINAL graph with the
  CUMULATIVE retraction set for the current round — never threads a reduced graph
  forward (that would lose the original grounding context and corrupt the
  ``is_grounded`` before/after comparison revise relies on).

- A round's retraction set is interpreted as the round's "move": retract exactly
  these targets this round (and reassert everything not in the set). The abstain
  set that round produces IS the belief state for ko purposes. A schedule that
  retracts {A} then {} then {A} again is the canonical ko: the {A, cascade}
  abstain set recurs.

Honest scope: this is a deterministic structural guard over the OKF ``derivesFrom``
graph. It consumes ``okf.revise`` and invents no facts. The abstain sequence is
derived, not predicted. It is candidate-only (``candidateOnly: true``): it earns
``level3Evidence: true`` only after a real run routes retraction decisions through
it with empirical evidence that escalation is the right operator response.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from okf.graph import Graph
from okf.revision import revise
from reasoning.consequence.ko_detector import KOAlert, detect_ko

# NOTE: the config defaults (ko_max_rounds, flip_severity_escalate) are looked up
# LAZILY inside run_revise_loop, not imported at module top. Importing
# agent.consequence_gate here at top level would (a) couple this low-level
# reasoning.* package to the agent.* layer, and (b) create a circular import:
# agent.consequence_gate's loader imports reasoning.consequence.ko_detector, which
# runs this package's __init__, which would re-import this module while it is
# still initializing. Lazy resolution keeps `import reasoning.consequence` free of
# any agent.* dependency and any config-file side effects.


@dataclass(frozen=True)
class ReviseLoopState:
    """Result of running an iterative revise/reassert schedule under a ko guard.

    - ``rounds``: the abstain-set sequence (one ``frozenset[str]`` per round) —
      the ko "belief state" history. Empty for a schedule with zero rounds.
    - ``verdicts``: the per-round consequence verdict
      (``allow``|``escalate``|``abstain``) from ``simulate_cascade``'s vocabulary,
      for the round's abstain set. The FINAL verdict is the loop's recommendation.
    - ``ko``: the ``KOAlert`` if a ko was detected (``None`` otherwise).
    - ``terminated``: True if the loop stopped early — on a ko, or on an
      unresolved retraction target (fail-closed).
    - ``reason``: human-readable termination/continuation reason.
    """

    schema: str = "sophia.consequence.revise_loop.v1"
    rounds: tuple[frozenset[str], ...] = ()
    verdicts: tuple[str, ...] = ()
    ko: "KOAlert | None" = None
    terminated: bool = False
    reason: str = "no rounds executed"
    roundsExecuted: int = 0
    candidateOnly: bool = True
    level3Evidence: bool = False
    boundary: str = (
        "ko-guarded revise loop is a deterministic structural cycle guard over the "
        "OKF derivesFrom graph; it invents no facts and is not AGI proof."
    )

    @property
    def finalVerdict(self) -> str:
        """The loop's recommendation: ``escalate`` on a ko, else the last round's verdict
        (``allow`` for a clean schedule, ``abstain`` if any round failed closed)."""
        if self.ko is not None and self.ko.ko:
            return "escalate"
        return self.verdicts[-1] if self.verdicts else "allow"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "rounds": [sorted(s) for s in self.rounds],
            "verdicts": list(self.verdicts),
            "finalVerdict": self.finalVerdict,
            "ko": self.ko.to_dict() if self.ko is not None else None,
            "terminated": self.terminated,
            "reason": self.reason,
            "roundsExecuted": self.roundsExecuted,
            "candidateOnly": self.candidateOnly,
            "level3Evidence": self.level3Evidence,
            "boundary": self.boundary,
        }


def _round_severity(graph: Graph, abstain: set[str]) -> float:
    """The flip severity of one round's abstain set (|abstain|/|graph|), matching
    ``simulate_cascade``'s structural-magnitude definition. Used only to map the
    round's abstain set onto the ``allow``|``escalate``|``abstain`` verdict
    vocabulary for reporting — the ko decision itself is set-recurrence, not severity."""
    if not abstain:
        return 0.0
    return len(abstain) / max(1, len(graph.nodes))


def run_revise_loop(
    graph: Graph,
    *,
    retraction_schedule: "list[list[str]] | None" = None,
    policy: "Callable[[int, frozenset[str]], list[str] | None] | None" = None,
    max_policy_rounds: int = 64,
    ko_max_rounds: "int | None" = None,
    escalate_threshold: "float | None" = None,
    by: str = "revise_loop",
) -> ReviseLoopState:
    """Run an iterative revise/reassert loop under a ko cycle guard.

    The loop is driven in one of two mutually exclusive modes (exactly one of
    ``retraction_schedule`` / ``policy`` must be given):

    - **Static schedule** (``retraction_schedule``): a pre-baked list of rounds;
      each round is the list of target ids to retract THAT round (a round's targets
      replace the prior round's — i.e. a retraction is "reasserted" by omitting it
      from a later round). This is the replay mode used by tests and benchmarks.

    - **Live policy** (``policy``): a caller-supplied decision callback
      ``policy(round_index, prev_abstain) -> list[str] | None`` invoked once per
      round to obtain that round's targets from the caller's OWN decision logic,
      reacting to the *previous* round's abstain set (empty frozenset for round 0).
      Returning ``None`` ends the loop cleanly. This is the mode a live runtime
      consumer (e.g. ``agent.belief_revision_policy.resolve_conflicts_consequence_aware``)
      uses to route real retraction decisions through the ko guard. The decision is
      INJECTED — this module imports nothing from ``agent.*`` to consume it, so the
      layering contract (no agent.* dependency in ``reasoning.consequence.*``) holds.

    Each round, in either mode:

    1. Calls ``okf.revise(graph, this_round_targets)`` on the ORIGINAL graph (revise
       is non-destructive; we never thread a reduced graph forward).
    2. Records the round's abstain set as the ko "belief state".
    3. Runs ``detect_ko`` over the accumulated state sequence. On a ko → terminate
       with ``finalVerdict = escalate`` (NEVER abstain).
    4. Maps the round's severity to a verdict for reporting (escalate if severity
       >= threshold, else allow; abstain if the round's targets did not resolve).

    ``max_policy_rounds`` bounds the live-policy mode: a policy that neither stops
    (returns ``None``) nor triggers a ko within the cap cannot be bounded, so the
    loop terminates fail-closed (``abstain``). It is ignored in schedule mode.

    ``ko_max_rounds`` defaults to the config value (``config/consequence.json``
    ``koMaxRounds``). ``escalate_threshold`` defaults to the config
    ``flipSeverityEscalate`` (used only for the per-round verdict vocabulary, NOT
    for the ko decision).

    Returns a ``ReviseLoopState``. The input ``graph`` is never mutated (revise is
    non-destructive; this function does not write either).
    """
    if (retraction_schedule is None) == (policy is None):
        raise ValueError(
            "run_revise_loop requires exactly one of retraction_schedule or policy"
        )
    if ko_max_rounds is None:
        # Lazy import: avoids coupling this module's import to agent.* / config I/O.
        from agent.consequence_gate import ko_max_rounds as _cfg_ko_max_rounds
        ko_max_rounds = _cfg_ko_max_rounds
    # Late import to keep the escalate_threshold default resolution lazy too
    # (same reason as ko_max_rounds above).
    if escalate_threshold is None:
        from agent.consequence_gate import flip_severity_escalate
        escalate_threshold = flip_severity_escalate

    rounds: list[frozenset[str]] = []
    verdicts: list[str] = []
    reason = "no rounds executed"

    i = 0
    while True:
        # Source this round's targets from the static schedule or the live policy.
        if retraction_schedule is not None:
            if i >= len(retraction_schedule):
                break
            targets = retraction_schedule[i]
        else:
            if i >= max_policy_rounds:
                # A policy that neither stopped nor ko'd within the cap cannot be
                # bounded -> fail-closed abstain (same posture as a notFound target).
                rounds.append(frozenset())
                verdicts.append("abstain")
                return ReviseLoopState(
                    rounds=tuple(rounds), verdicts=tuple(verdicts),
                    terminated=True, roundsExecuted=i + 1,
                    reason=(
                        f"policy did not terminate within max_policy_rounds="
                        f"{max_policy_rounds} -> fail-closed abstain"
                    ),
                )
            prev_abstain = rounds[-1] if rounds else frozenset()
            targets = policy(i, prev_abstain)
            if targets is None:
                break  # policy signalled clean termination
        rev = revise(graph, [(t, f"{by} round {i}") for t in targets], by=by)
        # An unresolved target is fail-closed: the round's consequence cannot be
        # bounded. Map to abstain and terminate — we cannot reason about the
        # cascade of a target that does not exist.
        if rev.notFound:
            abstain = frozenset()
            rounds.append(abstain)
            verdicts.append("abstain")
            return ReviseLoopState(
                rounds=tuple(rounds), verdicts=tuple(verdicts),
                terminated=True, roundsExecuted=i + 1,
                reason=f"round {i}: unresolved retraction target(s) {sorted(rev.notFound)} -> fail-closed abstain",
            )
        abstain = frozenset(rev.abstain)
        rounds.append(abstain)
        severity = _round_severity(graph, set(abstain))
        verdicts.append("escalate" if severity >= escalate_threshold else "allow")
        # Ko check over the accumulated sequence.
        ko = detect_ko([set(s) for s in rounds], max_rounds=ko_max_rounds)
        if ko.ko:
            return ReviseLoopState(
                rounds=tuple(rounds), verdicts=tuple(verdicts),
                ko=ko, terminated=True, roundsExecuted=i + 1,
                reason=ko.reason,
            )
        reason = f"round {i} complete; no ko across {len(rounds)} round(s)"
        i += 1

    return ReviseLoopState(
        rounds=tuple(rounds), verdicts=tuple(verdicts),
        terminated=False, roundsExecuted=len(rounds), reason=reason,
    )


__all__ = ["ReviseLoopState", "run_revise_loop"]
