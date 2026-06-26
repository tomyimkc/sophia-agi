# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Job model + synthetic trace generation for the cluster simulator.

A Job is a request for `gpus` accelerators for `duration_s` seconds, submitted at
`submit_s`. `colocate` jobs are collective-communication heavy (training): their GPUs
should land on as few nodes / NVLink islands as possible, or they pay a network tax.
`klass` optionally pins a device class (e.g. a kernel that only runs on H100).

Traces are generated from a seeded PRNG so every report is reproducible — no wall-clock
or unseeded randomness leaks into the numbers (mirrors the repo's no-overclaim discipline).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum


class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    gpus: int
    duration_s: float
    submit_s: float
    priority: int = 0               # higher = scheduled first
    colocate: bool = True           # collective-heavy → wants topological locality
    klass: str | None = None        # required device class, or None for any

    # ----- runtime state (mutated by the simulator) ------------------------
    state: JobState = JobState.PENDING
    start_s: float | None = None
    end_s: float | None = None
    devices: list[str] = field(default_factory=list)
    # span: how many distinct nodes / islands this placement used (set at schedule time)
    node_span: int = 0
    island_span: int = 0
    # fault/recovery accounting
    restarts: int = 0
    wasted_s: float = 0.0           # GPU-seconds lost to failures before a checkpoint
    committed_s: float = 0.0        # nominal work durably checkpointed (carries across restarts)

    @property
    def wait_s(self) -> float | None:
        if self.start_s is None:
            return None
        return self.start_s - self.submit_s

    @property
    def turnaround_s(self) -> float | None:
        if self.end_s is None:
            return None
        return self.end_s - self.submit_s


def synthetic_trace(
    *,
    n_jobs: int,
    seed: int,
    horizon_s: float = 3600.0,
    gpu_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    size_weights: tuple[float, ...] = (0.30, 0.15, 0.20, 0.15, 0.12, 0.08),
    min_dur_s: float = 120.0,
    max_dur_s: float = 1800.0,
    colocate_frac: float = 0.8,
) -> list[Job]:
    """Generate a reproducible Poisson-ish arrival trace of training/eval jobs.

    Arrivals are spread uniformly across [0, horizon_s) then sorted; GPU counts are
    drawn from `gpu_sizes` with `size_weights` (heavy tail toward big training jobs);
    durations log-uniform in [min_dur_s, max_dur_s]. `colocate_frac` of jobs are
    collective-heavy (want locality); the rest are embarrassingly parallel evals.
    """
    rng = random.Random(seed)
    jobs: list[Job] = []
    for i in range(n_jobs):
        submit = rng.uniform(0.0, horizon_s)
        gpus = rng.choices(gpu_sizes, weights=size_weights, k=1)[0]
        # log-uniform duration
        lo, hi = (min_dur_s, max_dur_s)
        dur = lo * (hi / lo) ** rng.random()
        colocate = rng.random() < colocate_frac
        jobs.append(
            Job(
                id=f"j{i:04d}",
                gpus=gpus,
                duration_s=round(dur, 1),
                submit_s=round(submit, 1),
                priority=rng.choice([0, 0, 0, 1, 2]),  # most normal, a few high-pri
                colocate=colocate,
            )
        )
    jobs.sort(key=lambda j: j.submit_s)
    return jobs
