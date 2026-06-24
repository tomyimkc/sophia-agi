# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Math RLVR dataset + contamination-free split (the math analogue of rl_dataset).

Loads ``data/math_problems.json`` and splits train/eval FAMILY-disjoint. Modern
packs (``tools/gen_math_pack.py``) carry an explicit ``split`` field so the held-out
families are FIXED and identical across seeds (comparable per-seed numbers); older
packs fall back to a seeded random family holdout. Either way an eval problem is a
*type* unseen at train time — "generalized" means "a new kind of problem", never
"memorized this instance".

The reward (``math_reward``) is the sympy verifier ``math_equivalent(gold)`` — a
deterministic, judge-free signal, exactly the RLVR setup that works for code/math.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parent / "data" / "math_problems.json"


def load_problems(path: Path | None = None) -> list[dict]:
    """Load the math problem rows ({id, family, prompt, gold})."""
    p = path or DATA
    return json.loads(p.read_text(encoding="utf-8"))["problems"]


def problem_to_row(prob: dict) -> dict:
    """A TRL-ready row: ``prompt`` + the ``gold`` column the reward routes on.

    ``remove_unused_columns=False`` must stay set (as in run_rlvr) so ``gold`` /
    ``family`` survive to the reward call via ``**kwargs``.
    """
    return {
        "prompt": prob["prompt"],
        "gold": prob["gold"],
        "family": prob["family"],
        "problem_id": prob["id"],
    }


def family_key(prob: dict) -> str:
    """The partition key that must NOT cross the train/eval boundary."""
    return str(prob["family"])


def split_problems(
    problems: list[dict],
    *,
    eval_frac: float = 0.34,
    seed: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Train/eval split, family-disjoint.

    Preferred: an explicit, FIXED ``split`` field on each problem ("train"/"eval")
    — held-out families are then identical across seeds, so seeds are comparable
    and the held-out N is whatever the pack author chose. Falls back to a seeded
    random family holdout for older packs that carry no ``split`` field.
    """
    if any(p.get("split") for p in problems):
        train = [p for p in problems if p.get("split") != "eval"]
        eval_ = [p for p in problems if p.get("split") == "eval"]
        return train, eval_
    families = sorted({family_key(p) for p in problems})
    rng = random.Random(seed)
    rng.shuffle(families)
    n_eval = max(1, round(len(families) * eval_frac))
    eval_fams = set(families[:n_eval])
    train = [p for p in problems if family_key(p) not in eval_fams]
    eval_ = [p for p in problems if family_key(p) in eval_fams]
    return train, eval_


def sealed_hash(problems: list[dict]) -> str:
    """Order-independent content hash of a split (seed-lock / tamper check)."""
    ids = sorted(p["id"] for p in problems)
    return hashlib.sha256("|".join(ids).encode("utf-8")).hexdigest()[:16]


def family_intersection(train: list[dict], eval_: list[dict]) -> list[str]:
    """Families present in BOTH splits — must be empty (contamination guard)."""
    return sorted({family_key(p) for p in train} & {family_key(p) for p in eval_})


def build_math_rl_dataset(
    *,
    eval_frac: float = 0.34,
    seed: int = 0,
    path: Path | None = None,
) -> dict[str, Any]:
    """Build train/eval rows + problems + seal hashes for the math RLVR run."""
    problems = load_problems(path)
    train, eval_ = split_problems(problems, eval_frac=eval_frac, seed=seed)
    return {
        "train_rows": [problem_to_row(p) for p in train],
        "eval_rows": [problem_to_row(p) for p in eval_],
        "train_problems": train,
        "eval_problems": eval_,
        "train_sealed": sealed_hash(train),
        "eval_sealed": sealed_hash(eval_),
        "family_intersection": family_intersection(train, eval_),
    }
