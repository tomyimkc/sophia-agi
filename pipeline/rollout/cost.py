# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Prefix-cache cost model — quantifies the savings the append-only harness buys.

A prefix-caching provider (DeepSeek, and the OpenAI/Anthropic cache APIs) bills a
*cache hit* on the already-seen prefix at a small fraction of the fresh input rate.
So the cost of a turn is::

    cost = cache_rate * cached_prefix_tokens
         + 1.0        * fresh_input_tokens          (the new tail this turn)
         + 1.0        * completion_tokens

The append-only ``Session`` maximizes ``cached_prefix_tokens`` (the whole prior
history is a cache hit every turn). A *naive* harness that reorders/rebuilds context
each turn — or lets a planner and executor share one session so each disturbs the
other's prefix — gets zero cache hits and pays the full input rate on the entire
growing context every turn. This module makes that difference a number.

``CACHE_RATE`` defaults to 0.1 (DeepSeek's published cache-hit input price is ~1/10
of the miss price). It is a parameter, not a claim — pass the rate your provider
actually bills.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CACHE_RATE = 0.1  # cache-hit input tokens billed at ~10% of fresh (DeepSeek-like)


@dataclass
class CostMeter:
    """Accumulates cached-harness cost vs. a naive (no-cache) baseline."""

    cache_rate: float = CACHE_RATE
    cached_input: float = 0.0      # billed input under prefix caching
    naive_input: float = 0.0       # billed input with no caching
    completion: float = 0.0
    turns: int = 0
    _events: list[dict] = field(default_factory=list)

    def record_turn(self, *, cached_prefix_tokens: int, fresh_input_tokens: int,
                    completion_tokens: int) -> None:
        """One model turn. ``cached_prefix_tokens`` is the prior prefix (a cache hit
        under the append-only harness); ``fresh_input_tokens`` is the new tail."""
        total_input = cached_prefix_tokens + fresh_input_tokens
        cached_cost = self.cache_rate * cached_prefix_tokens + fresh_input_tokens
        naive_cost = total_input  # no cache: pay full rate on the whole context
        self.cached_input += cached_cost
        self.naive_input += naive_cost
        self.completion += completion_tokens
        self.turns += 1
        self._events.append({
            "turn": self.turns,
            "cachedPrefix": cached_prefix_tokens,
            "freshInput": fresh_input_tokens,
            "cachedInputCost": round(cached_cost, 3),
            "naiveInputCost": round(naive_cost, 3),
        })

    @property
    def cached_total(self) -> float:
        return self.cached_input + self.completion

    @property
    def naive_total(self) -> float:
        return self.naive_input + self.completion

    def savings_ratio(self) -> float:
        """naive_total / cached_total — how much cheaper the cached harness is.

        ``>= 1`` always (caching never costs more); grows with conversation depth.
        Completions are billed identically both ways, so the input-only ratio is
        higher; this whole-rollout ratio is the honest, conservative number.
        """
        denom = self.cached_total or 1.0
        return self.naive_total / denom

    def summary(self) -> dict:
        return {
            "turns": self.turns,
            "cachedTotal": round(self.cached_total, 3),
            "naiveTotal": round(self.naive_total, 3),
            "savingsRatio": round(self.savings_ratio(), 3),
            "inputOnlySavingsRatio": round((self.naive_input or 0.0) / (self.cached_input or 1.0), 3),
            "cacheRate": self.cache_rate,
        }
