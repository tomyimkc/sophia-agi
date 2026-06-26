# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Contamination-free RL split for the visual-trap suite.

Mirrors ``provenance_bench/math_dataset``'s family split: rows are grouped by a
``family`` (here the trap category) and whole families go to either train or
eval, never both — so an RL run can never be evaluated on a task type it trained
on. ``family_intersection`` is asserted empty by ``visual_reward.offline_invariants``.

Each row is ``{"prompt", "trap", "family"}``; the prompt is the trap's question
and the ``trap`` column rides through TRL to ``visual_reward.make_grpo_reward``.
"""

from __future__ import annotations

import random

from multimodal_bench import runner


def trap_family(trap: dict) -> str:
    """Group key for the disjoint split (the task category)."""
    return trap.get("category", "unknown")


def build_visual_rl_dataset(*, eval_frac: float = 0.34, seed: int = 0, include_synth: bool = True) -> dict:
    """Split traps into train/eval by family, with no family in both.

    ``eval_frac`` is the fraction of *families* (not rows) held out for eval.
    """
    traps = runner.load_all_traps() if include_synth else runner.load_traps()
    families = sorted({trap_family(t) for t in traps})
    rng = random.Random(seed)
    shuffled = list(families)
    rng.shuffle(shuffled)
    n_eval = max(1, min(len(families) - 1, round(eval_frac * len(families))))
    eval_families = set(shuffled[:n_eval])
    train_families = set(shuffled[n_eval:])

    def row(t: dict) -> dict:
        return {"prompt": t["question"], "trap": t, "family": trap_family(t)}

    train_rows = [row(t) for t in traps if trap_family(t) in train_families]
    eval_rows = [row(t) for t in traps if trap_family(t) in eval_families]
    return {
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "train_families": sorted(train_families),
        "eval_families": sorted(eval_families),
        "family_intersection": sorted(train_families & eval_families),
    }
