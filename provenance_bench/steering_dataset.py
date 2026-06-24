# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Contamination-free extract/measure split of the IPIP item bank (Spec B).

Mirrors provenance_bench/rl_dataset.py's entity-disjoint split + sealed_hash, but
the unit is an IPIP item id (a vector is never measured on the items it was fit
on). Stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import random

from agent.personality_measure import load_bank


def _sealed(items: list) -> str:
    payload = sorted(it["id"] for it in items)
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]


def build_steering_split(*, eval_frac: float = 0.3, seed: int = 0) -> dict:
    items = load_bank()["items"]
    # split per OCEAN domain so each side keeps both poles where possible
    by_dim: dict = {}
    for it in items:
        by_dim.setdefault(it["domain"], []).append(it)
    rng = random.Random(seed)
    extract, measure = [], []
    for dim in sorted(by_dim):
        group = sorted(by_dim[dim], key=lambda it: it["id"])
        rng.shuffle(group)
        n_meas = max(1, min(len(group) - 1, int(round(len(group) * eval_frac))))
        measure.extend(group[:n_meas])
        extract.extend(group[n_meas:])
    ex_ids = {it["id"] for it in extract}
    me_ids = {it["id"] for it in measure}
    return {
        "extract_items": extract,
        "measure_items": measure,
        "extract_sealed": _sealed(extract),
        "measure_sealed": _sealed(measure),
        "item_intersection": sorted(ex_ids & me_ids),
    }
