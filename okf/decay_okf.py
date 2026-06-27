# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Belief dynamics for the OKF graph: decay, surprise-gating, competition suppression.

The OKF today is append-only: a belief's `authorConfidence` is static once written.
That is the gap vs. memory systems that model a belief *lifecycle* (Bayesian decay,
Ebbinghaus spacing, competition among mutually-exclusive claims). This module adds a
dynamics layer WITHOUT weakening source discipline:

  - Decay never deletes. A decayed belief is demoted via a provenance-carrying
    `supersededBy`/`decayedAt` edge; the reason (decay | contradiction | conscience
    veto | epistemic hygiene) is itself a record. Forgetting is a first-class,
    auditable act — not a silent score update.
  - Surprise-gating: only beliefs that SURPRISED the system (low predictive prob under
    current memory) get consolidated into the trusted semantic layer. Routine
    confirmations decay faster; contradictions get quarantined, not silently merged.
  - Competition suppression: among mutually-exclusive claims on one subject, the
    weakest-confidence ones are suppressed (decayed), not deleted — so a later
    contradiction can still resurrect them with new evidence.

Falsifiable property (the test): after N decay ticks, (a) total belief count is
non-decreasing (no silent deletion), (b) every decayed belief has a provenanced
reason, (c) no `consensus` belief is ever decayed by time alone (only by contradiction
with new evidence — source discipline outranks recency). Frontier-domain consensus
demotion under decisive evidence is governed separately by `okf.frontier_demotion`.

Honesty boundary: this is deterministic, offline, pure-stdlib dynamics over the graph.
It is NOT a learning rule and does not change weights; `level3Evidence: false` until a
real run clears the no-overclaim gate. See `cls_consolidation.py` for the weight side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from okf.schema import confidence_rank

# A belief's effective strength = base confidence rank * decay factor.
# Decay is exponential with a long half-life (Ebbinghaus): most beliefs are durable;
# only un-reinforced, low-surprise, contested ones fade.
DEFAULT_HALF_LIFE_DAYS = 365.0
SURPRISE_BOOST_MAX = 2.0          # a surprising, verified belief resists decay up to 2x
MIN_EFFECTIVE_STRENGTH = 0.05     # below this a belief is "suppressed" but NOT deleted

DECAY_REASONS = ("time", "contradiction", "conscience_veto", "epistemic_hygiene", "competition")


@dataclass(frozen=True)
class BeliefState:
    """The dynamic view of one OKF node, projected from its static frontmatter."""
    node_id: str
    author_confidence: str                 # the static, source-disciplined confidence
    written_at: float                      # epoch seconds
    last_reinforced_at: float              # epoch seconds (== written_at if never reinforced)
    surprise: float = 0.0                  # 0..1, how unexpected when first observed
    reinforcement_count: int = 0
    decayed_reason: str | None = None      # set when suppressed; None == live

    @property
    def base_rank(self) -> int:
        return confidence_rank(self.author_confidence)

    def effective_strength(self, now: float, *, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> float:
        """Exponential decay scaled by base confidence and surprise/reinforcement.

        consensus beliefs are decay-immune to time alone (returns base_rank) — only a
        contradiction record can suppress them, preserving source discipline. Frontier
        consensus demotion under decisive evidence is handled by frontier_demotion.
        """
        if self.author_confidence == "consensus":
            return float(self.base_rank)               # source discipline > recency
        age_days = max(0.0, (now - self.last_reinforced_at) / 86400.0)
        decay = math.pow(0.5, age_days / max(half_life_days, 1e-9))
        # surprise + reinforcement slow decay (a used, surprising belief is durable)
        resistance = 1.0 + min(SURPRISE_BOOST_MAX - 1.0, self.surprise + 0.1 * self.reinforcement_count)
        return float(self.base_rank) * decay * resistance


@dataclass
class DecayPlan:
    """The decision the dynamics layer emits. Default-deny: nothing is removed."""
    suppress: list[tuple[str, str]] = field(default_factory=list)   # (node_id, reason)
    reinforce: list[str] = field(default_factory=list)
    quarantine: list[tuple[str, str]] = field(default_factory=list)  # contradictions held for review
    deletions: int = 0   # ALWAYS 0 — forgetting is demotion, not destruction

    def to_dict(self) -> dict:
        return {
            "schema": "sophia.okf_decay_plan.v1",
            "suppress": self.suppress,
            "reinforce": self.reinforce,
            "quarantine": self.quarantine,
            "deletions": self.deletions,
            "noSilentDeletion": True,
            "candidateOnly": True,
            "level3Evidence": False,
        }


def plan_decay(
    beliefs: list[BeliefState],
    *,
    now: float | None = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    contradiction_groups: dict[str, list[str]] | None = None,
) -> DecayPlan:
    """Decide which beliefs to suppress/reinforce/quarantine.

    contradiction_groups: subject -> [node_ids] of mutually-exclusive claims.
    Within a group, the weakest-effective members are suppressed for `competition`,
    UNLESS the group contains a `consensus` claim (then the non-consensus ones lose).
    Ties are quarantined, not auto-decided (epistemic humility).
    """
    # ``now is None`` means "use the wall clock". Do NOT use ``now or ...``: a caller that
    # passes an explicit epoch of 0.0 (e.g. a GENESIS_EPOCH "arrival time unknown" marker)
    # is falsy, and ``or`` would silently swap it for the real clock — giving every belief a
    # ~20000-day age and time-decaying the whole corpus on an UNMEASURED timestamp. That is
    # exactly the fabricated-suppression the projection's HONESTY CONTRACT forbids.
    now = datetime.now(timezone.utc).timestamp() if now is None else now
    groups = contradiction_groups or {}
    plan = DecayPlan()

    for b in beliefs:
        s = b.effective_strength(now, half_life_days=half_life_days)
        if b.decayed_reason:
            continue                                   # already suppressed — leave the audit trail
        if s >= MIN_EFFECTIVE_STRENGTH:
            if b.surprise > 0.5 and b.reinforcement_count >= 1:
                plan.reinforce.append(b.node_id)       # surprising + used -> consolidate
            continue
        if b.author_confidence == "consensus":
            continue                                   # never time-decay consensus
        # Distinguish a genuine TIME decay (strength fell below the floor as age accrued)
        # from a STRUCTURAL low-base-confidence belief already below the floor at age 0
        # (none_extant / anachronism_risk -> base_rank 0). Calling the latter "time" would
        # be dishonest: no time elapsed, its recorded provenance is simply at/below the
        # floor. That is source discipline, not recency.
        undecayed = b.effective_strength(b.last_reinforced_at, half_life_days=half_life_days)
        reason = "time" if undecayed >= MIN_EFFECTIVE_STRENGTH else "epistemic_hygiene:low_base_confidence"
        plan.suppress.append((b.node_id, reason))

    # Competition: within a contradiction group, suppress the weak tail.
    for subject, members in groups.items():
        member_states = {b.node_id: b for b in beliefs if b.node_id in set(members)}
        if len(member_states) < 2:
            continue
        scored = sorted(
            member_states.values(),
            key=lambda b: (b.effective_strength(now, half_life_days=half_life_days), -b.base_rank),
        )
        # Hold the strongest; suppress the rest unless consensus is present (then it wins outright).
        if any(b.author_confidence == "consensus" for b in scored):
            losers = [b for b in scored if b.author_confidence != "consensus"]
            winner = "consensus"
        else:
            # tie at the top? quarantine instead of guessing (no silent auto-resolution)
            top_two = scored[-2:]
            if top_two[0].effective_strength(now) == top_two[1].effective_strength(now):
                for b in scored:
                    plan.quarantine.append((b.node_id, f"contradiction:{subject}:tied"))
                continue
            losers, winner = scored[:-1], scored[-1].author_confidence
        for b in losers:
            if not any(n == b.node_id for n, _ in plan.suppress):
                plan.suppress.append((b.node_id, f"competition:{subject}:lost_to_{winner}"))

    return plan


__all__ = [
    "BeliefState", "DecayPlan", "plan_decay",
    "DEFAULT_HALF_LIFE_DAYS", "SURPRISE_BOOST_MAX", "MIN_EFFECTIVE_STRENGTH", "DECAY_REASONS",
]
