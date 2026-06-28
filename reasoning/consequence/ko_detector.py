# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Ko-rule on thoughts — cycle detection over an iterative revision sequence.

Why this exists (and why it is here, not in ``okf.revision``): a single
``okf.revision.revise`` call is ko-safe — it batches all retractions and
computes the cascade in one consistent pass. The ko surface only appears once a
caller iterates: place retraction A, observe the abstain set, then reassert or
retract something else in response. A sequence like

    retract X -> must also drop Y -> reassert Y requires dropping X's rival Z
    -> dropping Z forces X back -> ...

is a GO ko: a loop over belief states that cannot terminate and cannot be
resolved without new information. Left unchecked it would thrash the gate.

The GO rule's analogue: if a revision sequence revisits a *prior abstain state*
within ``KO_MAX_ROUNDS`` steps, it is a ko. The correct response is not to pick
a side (that biases the loop) but to ``escalate`` — exactly the existing
conscience verdict for "needs stronger process / more information". A ko must
NEVER silently ``abstain``: abstain is a terminal "do not assert", whereas a ko
is an irreducible oscillation that a human or a new source must break.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

KO_MAX_ROUNDS = 4  # window in which a revisited belief state counts as a ko


@dataclass(frozen=True)
class KOAlert:
    """Result of ko-detecting over a revision sequence.

    - ``ko``: True iff a belief state recurred within ``KO_MAX_ROUNDS`` steps.
    - ``cycle``: the (first_seen_round, recurring_round) pair, or () if no ko.
    - ``recommendedVerdict``: always ``escalate`` when ``ko`` (ko != abstain).
    """

    schema: str = "sophia.consequence.ko.v1"
    ko: bool = False
    cycle: tuple[int, int] = ()
    rounds: int = 0
    recommendedVerdict: str = "allow"
    reason: str = "no belief-state recurrence within the ko window"
    candidateOnly: bool = True
    level3Evidence: bool = False
    boundary: str = "ko detection is a structural cycle guard, not AGI proof."

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "ko": self.ko,
            "cycle": list(self.cycle),
            "rounds": self.rounds,
            "recommendedVerdict": self.recommendedVerdict,
            "reason": self.reason,
            "candidateOnly": self.candidateOnly,
            "level3Evidence": self.level3Evidence,
            "boundary": self.boundary,
        }


def _state_key(abstain_state: frozenset[str] | set[str]) -> frozenset[str]:
    """Normalize a belief state to an order-independent, hashable key.

    The "belief state" for ko purposes is the *abstain set* — the claims the
    agent currently cannot assert. Two rounds are ko-equal iff their abstain
    sets are equal; intermediate reasoning text is deliberately ignored so a
    genuine flip is not masked by rewording.
    """
    return frozenset(abstain_state)


def detect_ko(revision_states: "list[set[str]]", *, max_rounds: int = KO_MAX_ROUNDS) -> KOAlert:
    """Detect a GO-ko recurrence in an iterative revision sequence.

    ``revision_states`` is the ordered list of abstain sets produced by each
    revise/reassert round (the ``abstain`` / ``claims_to_abstain`` output of
    ``okf.revision.revise`` at each step). A recurrence of a state within the
    trailing ``max_rounds`` window is a ko.

    Returns an alert whose ``recommendedVerdict`` is ``escalate`` when a ko is
    found. Callers MUST route a ko alert to the kernel's escalation path, not to
    abstention — see module docstring.
    """
    n = len(revision_states)
    if n < 2:
        return KOAlert(rounds=n)
    # Track the MOST RECENT round at which each belief state was seen, not the
    # first. The contract is "recurs within the trailing window": a state seen at
    # rounds 0 and 8 with max_rounds=4 must NOT ko on the 0->8 gap (8 > 4), but if
    # it recurs again at round 9 then 8->9 (gap 1 <= 4) IS a ko. Storing only the
    # first occurrence (seen.setdefault) would keep index 0 forever and both miss
    # 8->9 and wrongly flag 0->9 as a 9-gap non-ko vs the real 1-gap ko — so we
    # overwrite with the latest index each round.
    seen: dict[frozenset[str], int] = {}
    for i, state in enumerate(revision_states):
        key = _state_key(state)
        prev = seen.get(key)
        if prev is not None and (i - prev) <= max_rounds:
            return KOAlert(
                ko=True,
                cycle=(prev, i),
                rounds=n,
                recommendedVerdict="escalate",
                reason=(
                    f"ko: abstain state from round {prev} recurred at round {i} "
                    f"(gap {i - prev} <= {max_rounds}); irreducible without new information"
                ),
            )
        # update to the latest occurrence so the window is measured from the
        # most recent sighting, not the first.
        seen[key] = i
    return KOAlert(rounds=n, reason=f"no abstain-state recurrence across {n} rounds")


def is_ko(revision_states: "list[set[str]]", *, max_rounds: int = KO_MAX_ROUNDS) -> bool:
    """Convenience: True iff ``detect_ko`` raises the ko flag."""
    return detect_ko(revision_states, max_rounds=max_rounds).ko


__all__ = ["KO_MAX_ROUNDS", "KOAlert", "detect_ko", "is_ko"]
