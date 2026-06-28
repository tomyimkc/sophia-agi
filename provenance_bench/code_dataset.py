# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Code RLVR dataset + contamination-free split (the code analogue of math_dataset).

Loads ``data/code_tasks.json`` and splits train/eval FAMILY-disjoint. Modern packs
(``tools/gen_code_pack.py``) carry an explicit ``split`` field so the held-out
families are FIXED and identical across seeds (comparable per-seed numbers); either
way an eval task is a *type* unseen at train time — "generalized" means "a new kind
of problem", never "memorized this instance".

The reward (``code_reward``) is the tests-pass verifier ``check_answer`` — a
deterministic, judge-free signal (the interpreter decides), exactly the RLVR setup
that works for code/math.

Row shape carries the HIDDEN ``test`` column (the reward oracle). The pack's
``solution`` field is generator-only (used to self-verify the pack) and is NEVER
placed in an RLVR row — RLVR trains against the verifier, not gold solutions.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parent / "data" / "code_tasks.json"


def load_tasks(path: Path | None = None) -> list[dict]:
    """Load the code task rows ({id, family, split, entry_point, prompt, test, solution})."""
    p = path or DATA
    return json.loads(p.read_text(encoding="utf-8"))["tasks"]


def task_to_row(task: dict) -> dict:
    """A TRL-ready row: ``prompt`` + the hidden ``test`` column the reward routes on.

    The reference ``solution`` is deliberately OMITTED — RLVR trains against the
    verifier, never against a gold solution. ``remove_unused_columns=False`` must
    stay set (as in run_rlvr) so ``test``/``family`` survive to the reward call via
    ``**kwargs``.
    """
    return {
        "prompt": task["prompt"],
        "test": task["test"],
        "family": task["family"],
        "task_id": task["id"],
    }


def family_key(task: dict) -> str:
    """The partition key that must NOT cross the train/eval boundary."""
    return str(task["family"])


def split_tasks(
    tasks: list[dict],
    *,
    eval_frac: float = 0.34,
    seed: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Train/eval split, family-disjoint.

    Preferred: an explicit, FIXED ``split`` field on each task ("train"/"eval") —
    held-out families are then identical across seeds, so seeds are comparable and
    the held-out N is whatever the pack author chose. Falls back to a seeded random
    family holdout for packs that carry no ``split`` field.
    """
    if any(t.get("split") for t in tasks):
        train = [t for t in tasks if t.get("split") != "eval"]
        eval_ = [t for t in tasks if t.get("split") == "eval"]
        return train, eval_
    families = sorted({family_key(t) for t in tasks})
    rng = random.Random(seed)
    rng.shuffle(families)
    n_eval = max(1, round(len(families) * eval_frac))
    eval_fams = set(families[:n_eval])
    train = [t for t in tasks if family_key(t) not in eval_fams]
    eval_ = [t for t in tasks if family_key(t) in eval_fams]
    return train, eval_


def sealed_hash(tasks: list[dict]) -> str:
    """Order-independent content hash of a split (seed-lock / tamper check)."""
    ids = sorted(t["id"] for t in tasks)
    return hashlib.sha256("|".join(ids).encode("utf-8")).hexdigest()[:16]


def family_intersection(train: list[dict], eval_: list[dict]) -> list[str]:
    """Families present in BOTH splits — must be empty (contamination guard)."""
    return sorted({family_key(t) for t in train} & {family_key(t) for t in eval_})


def build_code_rl_dataset(
    *,
    eval_frac: float = 0.34,
    seed: int = 0,
    path: Path | None = None,
) -> dict[str, Any]:
    """Build train/eval rows + tasks + seal hashes for the code RLVR run."""
    tasks = load_tasks(path)
    train, eval_ = split_tasks(tasks, eval_frac=eval_frac, seed=seed)
    return {
        "train_rows": [task_to_row(t) for t in train],
        "eval_rows": [task_to_row(t) for t in eval_],
        "train_tasks": train,
        "eval_tasks": eval_,
        "train_sealed": sealed_hash(train),
        "eval_sealed": sealed_hash(eval_),
        "family_intersection": family_intersection(train, eval_),
    }
