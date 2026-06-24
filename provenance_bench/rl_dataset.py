# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""RL dataset + contamination-free split for the RLVR experiment.

Reuses ``provenance_bench.dataset.build_cases()`` (the external-ground-truth
case set) and the gate-record derivation. The split is by
``(work, author)`` ENTITY-PAIR, not by row, so a work seen at train time is
never evaluated — and gate records are built SEPARATELY per partition so an
eval case's reward verifier is never derived from a train case. Seed-locked
and sealed-hash, mirroring ``run_improvement_loop.py``'s contamination guard.

Non-circularity note (carried from ``dataset.py``): the gate record is the
runtime *treatment*; the true/false LABEL comes from the external citation.
For RL, training on the treatment is legitimate (the reward defines the task);
evaluating honestly requires the entity-disjoint split below so "generalized"
means "on works never seen during training", never "memorized a rule".
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any

from provenance_bench.dataset import Case, _alt_titles, _author_marker, build_cases


def case_to_row(case: Case) -> dict:
    """A TRL-ready row: ``prompt`` + the columns the reward fn routes on.

    ``remove_unused_columns`` must stay False (GRPOConfig default) so these
    columns survive to the reward call via ``**kwargs``.
    """
    return {
        "prompt": case.prompt,
        "label": case.label,
        "gold_author": case.gold_author,
        "claimed_author": case.claimed_author or "",
        "case_id": case.id,
    }


def entity_key(case: Case) -> tuple[str, str]:
    """The (work, author) pair that must NOT cross the train/eval boundary."""
    author = (case.claimed_author or case.gold_author or "").lower()
    return (case.work.lower(), author)


def split_cases(
    cases: list[Case],
    *,
    eval_frac: float = 0.3,
    seed: int = 0,
) -> tuple[list[Case], list[Case]]:
    """Entity-disjoint train/eval split.

    Cases sharing a ``(work, author)`` pair move together (so both labels of
    the same work land on one side), and no pair is shared across the split.
    """
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[Case]] = {}
    for c in cases:
        groups.setdefault(entity_key(c), []).append(c)
    keys = sorted(groups)
    rng.shuffle(keys)
    n_eval = max(1, min(len(keys) - 1, int(round(len(keys) * eval_frac))))
    eval_keys = set(keys[:n_eval])
    train: list[Case] = []
    eval_: list[Case] = []
    for k in keys:
        (eval_ if k in eval_keys else train).extend(groups[k])
    return train, eval_


def gate_records_for(cases: list[Case]) -> dict:
    """Gate records scoped to a partition (mirrors ``dataset.build_gate_records``
    but derived from the given case list, not the whole file).

    Only FALSE cases contribute ``doNotAttributeTo`` rules; building per
    partition keeps an eval case's reward verifier from being derived from a
    train case.
    """
    records: dict[str, dict] = {}
    for c in cases:
        if c.label != "false" or not c.claimed_author:
            continue
        rid = re.sub(r"[^a-z0-9]+", "_", c.work.lower()).strip("_")
        rec = records.setdefault(
            rid,
            {
                "canonicalTitleEn": c.work,
                "altTitlesEn": _alt_titles(c.work),
                "doNotAttributeTo": [],
            },
        )
        marker = _author_marker(c.claimed_author)
        if marker and marker not in rec["doNotAttributeTo"]:
            rec["doNotAttributeTo"].append(marker)
    return records


def sealed_hash(cases: list[Case]) -> str:
    """Stable hash of a partition's case ids/labels — proves the split didn't drift."""
    payload = sorted((c.id, c.label) for c in cases)
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]


def entity_intersection(train: list[Case], eval_: list[Case]) -> list[tuple[str, str]]:
    """Entity pairs shared across the split — must be empty (contamination guard)."""
    t = {entity_key(c) for c in train}
    e = {entity_key(c) for c in eval_}
    return sorted(t & e)


def build_rl_dataset(
    *,
    eval_frac: float = 0.3,
    seed: int = 0,
    data_dir: Any = None,
) -> dict[str, Any]:
    """Build train/eval rows + cases + per-partition gate records."""
    cases = build_cases(data_dir)
    train, eval_ = split_cases(cases, eval_frac=eval_frac, seed=seed)
    return {
        "train_rows": [case_to_row(c) for c in train],
        "eval_rows": [case_to_row(c) for c in eval_],
        "train_cases": train,
        "eval_cases": eval_,
        "train_gate_records": gate_records_for(train),
        "eval_gate_records": gate_records_for(eval_),
        "train_sealed": sealed_hash(train),
        "eval_sealed": sealed_hash(eval_),
        "entity_intersection": entity_intersection(train, eval_),
    }
