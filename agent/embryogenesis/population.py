# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Embryo specifications for the crucible arena.

Each embryo is an isolated configuration (tradition seed, verifier subset, council
mix) — not a weight update. Reproduction emits new specs only; LoRA distillation
requires a separate human-gated path.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EmbryoSpec:
    """One forked mini-Sophia configuration."""

    embryo_id: str
    tradition_seed: str
    verifier_subset: tuple[str, ...] = ("provenance_faithful",)
    council_mix: tuple[str, ...] = ("epistemic_humility",)
    governed_rsi: bool = False
    selfextend: bool = True
    generation: int = 0
    parent_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "embryoId": self.embryo_id,
            "traditionSeed": self.tradition_seed,
            "verifierSubset": list(self.verifier_subset),
            "councilMix": list(self.council_mix),
            "governedRsi": self.governed_rsi,
            "selfextend": self.selfextend,
            "generation": self.generation,
            "parentId": self.parent_id,
        }


TRADITION_SEEDS = ("confucian", "daoist", "buddhism", "christianity", "islam")
COUNCIL_SEATS = (
    "epistemic_humility",
    "historical_critical",
    "theological_voice",
    "source_discipline",
)
VERIFIER_SUBSETS = (
    ("provenance_faithful",),
    ("provenance_faithful", "grounding"),
    ("provenance_faithful", "arithmetic_sound"),
)


def seed_population(size: int = 8, *, generation: int = 0, seed: int = 0) -> list[EmbryoSpec]:
    """Create an initial population (8–32) from tradition × council combinations.

    ``seed`` is recorded for audit; initial combos are deterministic (itertools).
    """
    _ = seed  # reserved for future stochastic seeding; combos stay deterministic
    size = max(1, min(32, size))
    combos = list(
        itertools.islice(
            itertools.product(TRADITION_SEEDS, COUNCIL_SEATS, VERIFIER_SUBSETS),
            size,
        )
    )
    out: list[EmbryoSpec] = []
    for i, (trad, seat, verifiers) in enumerate(combos):
        out.append(
            EmbryoSpec(
                embryo_id=f"embryo_g{generation}_{i:02d}",
                tradition_seed=trad,
                verifier_subset=verifiers,
                council_mix=(seat, "source_discipline"),
                generation=generation,
            )
        )
    return out


def mutate(parent: EmbryoSpec, child_index: int, *, generation: int, seed: int = 0) -> EmbryoSpec:
    """Emit one child spec by swapping tradition, council seat, or verifier subset."""
    _ = seed
    trad_idx = (TRADITION_SEEDS.index(parent.tradition_seed) + child_index + 1) % len(TRADITION_SEEDS)
    seat_idx = (COUNCIL_SEATS.index(parent.council_mix[0]) + child_index + 1) % len(COUNCIL_SEATS)
    ver_idx = (VERIFIER_SUBSETS.index(parent.verifier_subset) + child_index + 1) % len(VERIFIER_SUBSETS)
    return EmbryoSpec(
        embryo_id=f"embryo_g{generation}_{child_index:02d}",
        tradition_seed=TRADITION_SEEDS[trad_idx],
        verifier_subset=VERIFIER_SUBSETS[ver_idx],
        council_mix=(COUNCIL_SEATS[seat_idx], "source_discipline"),
        governed_rsi=parent.governed_rsi,
        selfextend=parent.selfextend,
        generation=generation,
        parent_id=parent.embryo_id,
    )
