# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Static vs continuous batching simulator + serving metrics (pure stdlib).

Iteration-stepped model (1 tick = 1 engine iteration). A request needs
`prefill_iters = ceil(prompt_len / prefill_chunk)` prefill ticks, then `output_len`
decode ticks (one token/tick). Up to `max_batch` requests advance per tick.

- STATIC (request-level) batching: admit a batch of arrived requests, run it until
  EVERY request in it finishes (head-of-line blocking), then admit the next batch.
- CONTINUOUS (iteration-level) batching (Orca): each tick, free finished slots and
  admit waiting requests immediately, so short requests don't wait behind long ones.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from serving.sim.workload import Request


@dataclass
class _Live:
    req: Request
    prefill_iters: int
    progress: int = 0          # ticks advanced
    first_token: int | None = None
    finish: int | None = None

    @property
    def total(self) -> int:
        return self.prefill_iters + self.req.output_len

    @property
    def done(self) -> bool:
        return self.progress >= self.total


@dataclass
class SimResult:
    policy: str
    n: int
    makespan: int
    output_tokens: int
    throughput: float          # output tokens / makespan
    ttft_mean: float
    ttft_p99: float
    tpot_mean: float
    goodput: float             # fraction meeting both SLOs

    def to_dict(self) -> dict:
        return {
            "policy": self.policy, "n": self.n, "makespan": self.makespan,
            "outputTokens": self.output_tokens, "throughput": round(self.throughput, 6),
            "ttftMean": round(self.ttft_mean, 6), "ttftP99": round(self.ttft_p99, 6),
            "tpotMean": round(self.tpot_mean, 6), "goodput": round(self.goodput, 6),
        }


def _advance(live: _Live, t: int) -> None:
    live.progress += 1
    if live.first_token is None and live.progress == live.prefill_iters + 1:
        live.first_token = t
    if live.progress >= live.total and live.finish is None:
        live.finish = t


def _percentile(xs: "list[float]", q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def _metrics(policy: str, finished: "list[_Live]", first_arrival: int,
             ttft_slo: int, tpot_slo: float) -> SimResult:
    ttfts, tpots, good = [], [], 0
    out_tokens = 0
    last = first_arrival
    for lv in finished:
        out_tokens += lv.req.output_len
        last = max(last, lv.finish or 0)
        ttft = (lv.first_token or lv.finish or 0) - lv.req.arrival
        ttfts.append(ttft)
        if lv.req.output_len > 1 and lv.finish is not None and lv.first_token is not None:
            tpot = (lv.finish - lv.first_token) / (lv.req.output_len - 1)
        else:
            tpot = 0.0
        tpots.append(tpot)
        if ttft <= ttft_slo and tpot <= tpot_slo:
            good += 1
    makespan = max(1, last - first_arrival)
    n = len(finished)
    return SimResult(
        policy=policy, n=n, makespan=makespan, output_tokens=out_tokens,
        throughput=out_tokens / makespan,
        ttft_mean=sum(ttfts) / n if n else 0.0,
        ttft_p99=_percentile(ttfts, 0.99),
        tpot_mean=sum(tpots) / n if n else 0.0,
        goodput=good / n if n else 0.0,
    )


def _make_live(reqs: "list[Request]", prefill_chunk: int) -> "dict[int, _Live]":
    return {r.rid: _Live(r, max(1, math.ceil(r.prompt_len / prefill_chunk))) for r in reqs}


def simulate(reqs: "list[Request]", *, policy: str, max_batch: int = 4,
             prefill_chunk: int = 16, ttft_slo: int = 30, tpot_slo: float = 2.0,
             max_ticks: int = 1_000_000) -> SimResult:
    """Run one policy ('static' | 'continuous') over `reqs`; return serving metrics."""
    if policy not in ("static", "continuous"):
        raise ValueError(f"unknown policy {policy!r}")
    pending = sorted(reqs, key=lambda r: (r.arrival, r.rid))
    live = _make_live(pending, prefill_chunk)
    queue = [live[r.rid] for r in pending]
    first_arrival = pending[0].arrival if pending else 0
    finished: "list[_Live]" = []
    active: "list[_Live]" = []
    t = first_arrival
    qi = 0  # index into queue of next not-yet-admitted request

    while len(finished) < len(reqs) and t < max_ticks:
        if policy == "static":
            if not active:
                # Admit a fresh batch of up to max_batch ARRIVED requests.
                if qi < len(queue) and queue[qi].req.arrival > t:
                    t = queue[qi].req.arrival
                while qi < len(queue) and len(active) < max_batch and queue[qi].req.arrival <= t:
                    active.append(queue[qi]); qi += 1
        else:  # continuous: top up free slots every tick
            while qi < len(queue) and len(active) < max_batch and queue[qi].req.arrival <= t:
                active.append(queue[qi]); qi += 1
            if not active and qi < len(queue):
                t = queue[qi].req.arrival
                continue
        if not active:
            break
        for lv in active:
            _advance(lv, t)
        just = [lv for lv in active if lv.done]
        finished.extend(just)
        active = [lv for lv in active if not lv.done]
        t += 1

    return _metrics(policy, finished, first_arrival, ttft_slo, tpot_slo)


def compare(reqs: "list[Request]", **kw) -> dict:
    """Run both policies; return per-policy metrics + the continuous-over-static gains."""
    static = simulate(reqs, policy="static", **kw)
    cont = simulate(reqs, policy="continuous", **kw)
    return {
        "static": static.to_dict(),
        "continuous": cont.to_dict(),
        "throughputSpeedup": round(cont.throughput / static.throughput, 6)
        if static.throughput else 0.0,
        "ttftMeanReduction": round(1.0 - (cont.ttft_mean / static.ttft_mean), 6)
        if static.ttft_mean else 0.0,
    }
