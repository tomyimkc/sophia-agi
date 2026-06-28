# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""RL dataset + contamination-free split for the concept-discipline RLVR task.

Mirrors ``provenance_bench.rl_dataset`` for concept edges: the split is by the
``(subject, object)`` concept-PAIR entity, not by row, so a concept pair seen at
train time is never evaluated. Seed-locked and sealed-hash. The cases come from
``provenance_bench.ontology_rl_reward.ontology_rl_cases`` (generated parametrically
from the concept lexicon, so train-vs-heldout gaming stays detectable).

The reward (``ontology_rl_reward``) is the runtime treatment; the ``expected``
label (distinct / admit) is derived from the closed-world lexicon. Honest scope:
training on the treatment is legitimate (the reward defines the task); honest
evaluation needs the pair-disjoint split below so "generalized" means "on concept
pairs never seen during training". See docs/11-Platform/Ontology-Claim-Boundary.md.
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from provenance_bench.ontology_rl_reward import ontology_rl_cases


def case_to_row(case: dict) -> dict:
    """A TRL-ready row: ``prompt`` + the columns the reward fn routes on
    (``expected`` / ``answerable``). ``remove_unused_columns`` must stay False."""
    return {
        "prompt": case["prompt"],
        "expected": case["expected"],
        "answerable": bool(case.get("answerable", True)),
        "case_id": case["id"],
    }


def entity_key(case: dict) -> tuple[str, str]:
    """The (subject, object) concept pair that must NOT cross the train/eval boundary."""
    a = str(case.get("subject", "")).lower()
    b = str(case.get("object", "")).lower()
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def split_cases(cases: list[dict], *, eval_frac: float = 0.3, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Pair-disjoint train/eval split (cases sharing a concept pair move together)."""
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict]] = {}
    for c in cases:
        groups.setdefault(entity_key(c), []).append(c)
    keys = sorted(groups)
    rng.shuffle(keys)
    n_eval = max(1, min(len(keys) - 1, int(round(len(keys) * eval_frac)))) if len(keys) > 1 else 0
    eval_keys = set(keys[:n_eval])
    train: list[dict] = []
    eval_: list[dict] = []
    for k in keys:
        (eval_ if k in eval_keys else train).extend(groups[k])
    return train, eval_


def sealed_hash(cases: list[dict]) -> str:
    payload = sorted((c["id"], c["expected"]) for c in cases)
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]


def entity_intersection(train: list[dict], eval_: list[dict]) -> list[tuple[str, str]]:
    """Concept pairs shared across the split — must be empty (contamination guard)."""
    t = {entity_key(c) for c in train}
    e = {entity_key(c) for c in eval_}
    return sorted(t & e)


def build_ontology_rl_dataset(*, eval_frac: float = 0.3, seed: int = 0) -> dict[str, Any]:
    """Build train/eval rows + cases for the concept-discipline RLVR task."""
    cases = ontology_rl_cases()
    train, eval_ = split_cases(cases, eval_frac=eval_frac, seed=seed)
    return {
        "train_rows": [case_to_row(c) for c in train],
        "eval_rows": [case_to_row(c) for c in eval_],
        "train_cases": train,
        "eval_cases": eval_,
        "train_sealed": sealed_hash(train),
        "eval_sealed": sealed_hash(eval_),
        "entity_intersection": entity_intersection(train, eval_),
    }


__all__ = ["case_to_row", "entity_key", "split_cases", "sealed_hash",
           "entity_intersection", "build_ontology_rl_dataset"]
