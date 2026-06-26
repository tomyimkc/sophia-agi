# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Held-out test banks: deterministic train/test split + content hashing (stdlib).

A capability number is only credible on a HELD-OUT split with a pinned content
hash, so a result is reproducible and can't be silently re-tuned on its own test
set. Banks stay category-level and contain requests only — never hazardous content.
"""
from __future__ import annotations

import hashlib
import random


def bank_hash(prompts: "tuple[str, ...] | list[str]") -> str:
    """Order-independent content hash of a bank (first 16 hex of sha256)."""
    h = hashlib.sha256()
    for p in sorted(prompts):
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def split_bank(prompts: "tuple[str, ...] | list[str]", *, test_frac: float = 0.5,
               seed: int = 0) -> "tuple[tuple[str, ...], tuple[str, ...]]":
    """Deterministic (train, test) split. Same (prompts, frac, seed) -> same split.
    The test split is what capability scores are reported on."""
    items = list(prompts)
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0,1), got {test_frac}")
    rng = random.Random(seed)
    order = list(range(len(items)))
    rng.shuffle(order)
    n_test = max(1, round(len(items) * test_frac))
    test_idx = set(order[:n_test])
    test = tuple(items[i] for i in range(len(items)) if i in test_idx)
    train = tuple(items[i] for i in range(len(items)) if i not in test_idx)
    return train, test
