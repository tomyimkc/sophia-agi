# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Seeded request workload for the serving simulator (pure stdlib)."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class Request:
    rid: int
    arrival: int          # integer iteration tick of arrival
    prompt_len: int       # tokens to prefill
    output_len: int       # decode tokens to generate


def poisson_workload(n: int, *, rate: float = 0.5, seed: int = 0,
                     prompt_lo: int = 8, prompt_hi: int = 64,
                     out_lo: int = 8, out_hi: int = 64) -> "list[Request]":
    """`n` requests with exponential inter-arrival gaps (mean 1/rate, discretized)
    and uniform prompt/output lengths. Deterministic for a fixed seed."""
    rng = random.Random(seed)
    reqs = []
    t = 0.0
    for i in range(n):
        gap = -math.log(1.0 - rng.random()) / rate  # exponential inter-arrival
        t += gap
        reqs.append(Request(
            rid=i,
            arrival=int(t),
            prompt_len=rng.randint(prompt_lo, prompt_hi),
            output_len=rng.randint(out_lo, out_hi),
        ))
    return reqs
