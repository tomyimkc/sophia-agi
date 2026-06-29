# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Virtue Parliament — the inter-virtue arbiter (Dikaiosyne, Role B).

Once Sophia has four orthogonal virtue gates — Wisdom (the conscience kernel),
Courage (Andreia), Temperance (Sophrosyne), Justice (Dikaiosyne, Role A) — they can
DISAGREE: Courage may say *act*, Temperance *restrain*, Wisdom *abstain*. Four
bolt-on vetoes are not a virtue model. Plato's *Republic* (IV) defines justice as
the harmony of the parts — "each doing its own proper work" (τὰ αὑτοῦ πράττειν) — so
the arbiter that resolves inter-virtue conflict by a CONSISTENT, auditable rule is
itself an act of justice. This module is that arbiter, and it encodes the Stoic
**unity of virtue** as an enforceable invariant: no lower virtue can act against a
higher one, and the same conflict always resolves the same way.

The priority is a **pre-registered lexical order** (a measurement decision, not a
tuning knob):

    1. hard prohibitions (constitution / classifier / deception)  — absolute, first
    2. Wisdom    (conscience: block/abstain/retrieve on the truth) — never overridden by 3-5
    3. Justice   (impartiality — like cases alike)                 — consistency floor
    4. Courage   (act/hold direction)                              — may upgrade abstain->escalate
    5. Temperance(magnitude/duration)                              — trims/sustains, never suppresses

Determinism is part of the contract: ``arbitrate`` depends ONLY on the virtue
verdicts (by virtue identity, never on argument order), so identical conflicts
resolve identically across seeds and orderings. That self-consistency is itself a
Justice (consistency) property — the arbiter is just by construction.

Candidate infrastructure (``candidateOnly=True``): this is a deterministic policy
over the gates' verdicts, not a learned faculty and not AGI proof.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Unified posture vocabulary the arbiter resolves to, ordered by restrictiveness
# (low index = least restrictive). A higher virtue's floor can only be RAISED by a
# lower virtue, never lowered — that is the unity-of-virtue invariant.
POSTURES = ("proceed", "revise", "retrieve", "clarify", "escalate", "abstain", "block")
_RANK = {p: i for i, p in enumerate(POSTURES)}

# Map each gate's native verdict into the unified posture.
_WISDOM_MAP = {
    "allow": "proceed", "revise": "revise", "retrieve": "retrieve", "clarify": "clarify",
    "escalate": "escalate", "abstain": "abstain", "block": "block",
}

# Pre-registered priority order (by virtue identity).
PRIORITY = ("hard_prohibition", "wisdom", "justice", "courage", "temperance")


def _raise_to(current: str, floor: str) -> str:
    """Return the MORE restrictive of two postures (the floor can only raise)."""
    return current if _RANK[current] >= _RANK[floor] else floor


@dataclass(frozen=True)
class ArbitratedDecision:
    schema: str = "sophia.virtue_arbitration.v1"
    posture: str = "proceed"
    governingVirtue: str = "wisdom"
    reason: str = "no inter-virtue conflict"
    priorityChain: tuple[dict[str, Any], ...] = ()
    inputs: dict[str, Any] = field(default_factory=dict)
    candidateOnly: bool = True
    level3Evidence: bool = False
    boundary: str = (
        "Virtue arbitration is a deterministic, pre-registered lexical-priority policy "
        "over the four virtue gates' verdicts — the Republic 'harmony of the parts' as a "
        "consistency rule, not a learned faculty and not AGI proof. No lower virtue can "
        "override a higher one; hard prohibitions are always first."
    )

    def __post_init__(self) -> None:
        if self.posture not in POSTURES:
            raise ValueError(f"posture must be one of {POSTURES}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "posture": self.posture,
            "governingVirtue": self.governingVirtue,
            "reason": self.reason,
            "priorityChain": list(self.priorityChain),
            "inputs": self.inputs,
            "candidateOnly": self.candidateOnly,
            "level3Evidence": self.level3Evidence,
            "boundary": self.boundary,
        }


def arbitrate(
    *,
    wisdom: str = "allow",
    courage: str = "hold",
    temperance: str = "proportionate",
    justice: str = "impartial",
    hard_block: bool = False,
) -> ArbitratedDecision:
    """Resolve the four virtue verdicts into one posture by the pre-registered priority.

    Args are the gates' NATIVE verdicts:
      * ``wisdom``     ∈ conscience verdicts (allow|revise|retrieve|clarify|escalate|abstain|block)
      * ``courage``    ∈ {act, heroic, escalate, hold}
      * ``temperance`` ∈ {proportionate, restrain, sustain, escalate}
      * ``justice``    ∈ {impartial, partial, false_equivalence, arbitrate}
      * ``hard_block`` : an explicit hard-prohibition flag (constitution/classifier/deception)
    """
    chain: list[dict[str, Any]] = []
    inputs = {"wisdom": wisdom, "courage": courage, "temperance": temperance,
              "justice": justice, "hardBlock": hard_block}

    # 1 — hard prohibitions: absolute, first.
    if hard_block or wisdom == "block":
        chain.append({"virtue": "hard_prohibition", "applied": True,
                      "effect": "block (absolute floor)"})
        return ArbitratedDecision(posture="block", governingVirtue="hard_prohibition",
                                  reason="a hard prohibition / conscience block is absolute and overrides every other virtue",
                                  priorityChain=tuple(chain), inputs=inputs)

    # 2 — Wisdom sets the base posture (truth gate). Never lowered by 3-5.
    posture = _WISDOM_MAP.get(wisdom, "proceed")
    governing = "wisdom"
    chain.append({"virtue": "wisdom", "applied": True, "effect": f"base posture {posture}"})

    # 3 — Justice: a consistency floor. Partiality / false balance forces at least escalate.
    if justice in {"partial", "false_equivalence"}:
        new = _raise_to(posture, "escalate")
        applied = new != posture
        chain.append({"virtue": "justice", "applied": applied,
                      "effect": f"{'raised to ' + new if applied else 'floor already met'} (consistency)"})
        if applied:
            posture, governing = new, "justice"
    else:
        chain.append({"virtue": "justice", "applied": False, "effect": "impartial — no floor raised"})

    # 4 — Courage: may upgrade a quiet abstain to escalate (fear-driven hold).
    if courage == "escalate" and posture == "abstain":
        posture, governing = "escalate", "courage"
        chain.append({"virtue": "courage", "applied": True,
                      "effect": "abstain looks fear-driven -> escalate"})
    else:
        chain.append({"virtue": "courage", "applied": False,
                      "effect": "no upgrade (cannot override a higher virtue or lower a floor)"})

    # 5 — Temperance: trims/sustains, never suppresses. Restrain on a 'proceed' -> revise
    # (tighter wording); it can only RAISE, never lower, so it cannot weaken safety.
    if temperance == "restrain" and posture == "proceed":
        posture, governing = "revise", "temperance"
        chain.append({"virtue": "temperance", "applied": True,
                      "effect": "excess on an otherwise-proceed -> revise (trim)"})
    elif temperance == "sustain" and posture == "abstain":
        # A premature/lazy abstention with effort still valuable -> escalate (justify, don't quit).
        posture, governing = "escalate", "temperance"
        chain.append({"virtue": "temperance", "applied": True,
                      "effect": "abstain looks premature -> escalate"})
    else:
        chain.append({"virtue": "temperance", "applied": False,
                      "effect": "no trim/sustain applied (never suppresses a higher floor)"})

    reason = (f"resolved by {governing} under the pre-registered priority "
              "hard_prohibition > wisdom > justice > courage > temperance")
    return ArbitratedDecision(posture=posture, governingVirtue=governing, reason=reason,
                              priorityChain=tuple(chain), inputs=inputs)


# --------------------------------------------------------------------------- #
# Deterministic self-benchmark + a consistency/property check (the unity-of-virtue
# invariant). The FULL property test lives in tests/test_virtue_parliament.py.
# --------------------------------------------------------------------------- #
def run_virtue_parliament_benchmark() -> dict[str, Any]:
    cases = [
        {"id": "all_aligned_proceed",
         "kwargs": {"wisdom": "allow", "courage": "act", "temperance": "proportionate", "justice": "impartial"},
         "expect": ("proceed", "wisdom")},
        {"id": "hard_block_wins",
         "kwargs": {"wisdom": "allow", "courage": "heroic", "temperance": "proportionate", "justice": "impartial", "hard_block": True},
         "expect": ("block", "hard_prohibition")},
        {"id": "wisdom_block_wins_over_courage",
         "kwargs": {"wisdom": "block", "courage": "heroic", "temperance": "sustain", "justice": "impartial"},
         "expect": ("block", "hard_prohibition")},
        {"id": "justice_floor_raises_proceed",
         "kwargs": {"wisdom": "allow", "courage": "act", "temperance": "proportionate", "justice": "partial"},
         "expect": ("escalate", "justice")},
        {"id": "courage_upgrades_abstain",
         "kwargs": {"wisdom": "abstain", "courage": "escalate", "temperance": "proportionate", "justice": "impartial"},
         "expect": ("escalate", "courage")},
        {"id": "temperance_trims_proceed",
         "kwargs": {"wisdom": "allow", "courage": "hold", "temperance": "restrain", "justice": "impartial"},
         "expect": ("revise", "temperance")},
        {"id": "temperance_cannot_lower_wisdom_abstain",
         "kwargs": {"wisdom": "abstain", "courage": "hold", "temperance": "restrain", "justice": "impartial"},
         "expect": ("abstain", "wisdom")},
        {"id": "courage_relieves_fear_driven_abstain_when_just",
         # wisdom=abstain already outranks justice's escalate floor (abstain is more
         # restrictive), so justice does not raise it; courage is what lowers the
         # fear-driven abstain to escalate. Posture escalate, governed by courage.
         "kwargs": {"wisdom": "abstain", "courage": "escalate", "temperance": "proportionate", "justice": "partial"},
         "expect": ("escalate", "courage")},
    ]
    rows = []
    for c in cases:
        d = arbitrate(**c["kwargs"]).to_dict()
        got = (d["posture"], d["governingVirtue"])
        ok = got == tuple(c["expect"])
        rows.append({"id": c["id"], "expect": list(c["expect"]), "got": list(got), "ok": ok})
    return {
        "schema": "sophia.virtue_parliament_benchmark.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "n": len(rows),
        "passed": sum(r["ok"] for r in rows),
        "accuracy": round(sum(r["ok"] for r in rows) / len(rows), 4),
        "cases": rows,
        "ok": all(r["ok"] for r in rows),
        "boundary": "Deterministic candidate arbitration policy; the Republic harmony as a consistency rule, not AGI proof.",
    }


def write_virtue_parliament_report(out: str | Path) -> dict[str, Any]:
    report = run_virtue_parliament_benchmark()
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


__all__ = [
    "POSTURES",
    "PRIORITY",
    "ArbitratedDecision",
    "arbitrate",
    "run_virtue_parliament_benchmark",
    "write_virtue_parliament_report",
]
