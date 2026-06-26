# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Search strategies for the autonomous experiment runner.

Each strategy is a stateful policy with the same tiny contract:

    initial() -> config
    propose_next(history) -> config | None      # None means "converged, stop"

``history`` is the list of completed trials (``{config, score, result, trial}``) for THIS
strategy; ``score`` is held-out loss (lower is better; ``inf`` if the run diverged). The
policies are real closed-loop search — they read measured results and decide the next
config — not pre-baked grids. All deterministic and dependency-free.
"""
from __future__ import annotations

from typing import Any


class LearningRateSearch:
    """Adaptive hill-climb on learning rate in log space.

    Evaluates the current centre and its ×factor / ÷factor neighbours; recentres on the
    winner; shrinks the factor when the centre is already best; stops when the factor is
    small or the trial budget is hit. A diverged run (score=inf) naturally pushes the search
    toward smaller learning rates — the same instinct a human has when training blows up.
    """

    def __init__(self, base: "dict[str, Any]", *, lr0: float = 0.05, factor: float = 2.0,
                 min_factor: float = 1.25):
        self.base = dict(base)
        self.center = lr0
        self.factor = factor
        self.min_factor = min_factor
        self.evaluated: dict[float, float] = {}
        self.pending: list[float] = [lr0]

    def _cfg(self, lr: float) -> "dict[str, Any]":
        c = dict(self.base)
        c["lr"] = round(lr, 6)
        return c

    def initial(self) -> "dict[str, Any]":
        return self._cfg(self.pending.pop(0))

    def propose_next(self, history) -> "dict[str, Any] | None":
        last = history[-1]
        self.evaluated[last["config"]["lr"]] = last["score"]
        if self.pending:
            return self._cfg(self.pending.pop(0))
        # batch done: recentre on the best lr seen so far
        best_lr = min(self.evaluated, key=lambda k: self.evaluated[k])
        if best_lr == self.center:
            self.factor = self.factor ** 0.5            # tighten around a stable optimum
            if self.factor <= self.min_factor:
                return None
        self.center = best_lr
        for lr in (self.center * self.factor, self.center / self.factor):
            lr = round(lr, 6)
            if lr not in self.evaluated:
                self.pending.append(lr)
        if not self.pending:
            return None
        return self._cfg(self.pending.pop(0))


class MixtureSearch:
    """Ternary search on the mixing weight ``wA`` in [0,1] for a target distribution.

    The 配比 question, automated: held-out loss vs mixing ratio is ~unimodal, so a ternary
    search converges on the optimum in a handful of real runs instead of a full grid.
    """

    def __init__(self, base: "dict[str, Any]", *, iters: int = 4):
        self.base = dict(base)
        self.lo, self.hi = 0.0, 1.0
        self.iters_left = iters
        self.pending: list[float] = []
        self._last_pair: tuple[float, float] | None = None

    def _cfg(self, wA: float) -> "dict[str, Any]":
        c = dict(self.base)
        c["mix"] = round(wA, 4)
        return c

    def _new_pair(self) -> None:
        m1 = self.lo + (self.hi - self.lo) / 3
        m2 = self.hi - (self.hi - self.lo) / 3
        self._last_pair = (round(m1, 4), round(m2, 4))
        self.pending = [self._last_pair[0], self._last_pair[1]]

    def initial(self) -> "dict[str, Any]":
        self._new_pair()
        return self._cfg(self.pending.pop(0))

    def propose_next(self, history) -> "dict[str, Any] | None":
        if self.pending:
            return self._cfg(self.pending.pop(0))
        # both midpoints evaluated: narrow the bracket toward the better one
        m1, m2 = self._last_pair
        s1 = next(h["score"] for h in reversed(history) if h["config"]["mix"] == m1)
        s2 = next(h["score"] for h in reversed(history) if h["config"]["mix"] == m2)
        if s1 < s2:
            self.hi = m2
        else:
            self.lo = m1
        self.iters_left -= 1
        if self.iters_left <= 0:
            return None
        self._new_pair()
        return self._cfg(self.pending.pop(0))


class ComputeAllocation:
    """Compute-optimal allocation search: at ~fixed compute, how to split params vs tokens.

    A Chinchilla-flavoured question at toy scale. Compute proxy ≈ params(hidden) × D. We
    sweep allocations along an iso-compute curve (small-model/much-data ... big-model/
    little-data) and report which split minimizes held-out loss — the autonomous version of
    "对 scaling 行为建立认知并合理规划".
    """

    def __init__(self, base: "dict[str, Any]", *, hiddens=(4, 8, 16, 32, 64),
                 compute_proxy: int = 1600 * 808):
        self.base = dict(base)
        self.compute = compute_proxy
        self.hiddens = list(hiddens)
        self._i = 0

    def _params(self, hidden: int) -> int:
        V = self.base.get("vocab", 8)
        c = self.base.get("context", 2)
        in_dim = c * V
        return in_dim * hidden + hidden + hidden * V + V

    def _cfg(self, hidden: int) -> "dict[str, Any]":
        c = dict(self.base)
        c["hidden"] = hidden
        # hold compute ≈ params × D fixed -> D = compute / params
        c["D"] = max(100, int(self.compute / max(1, self._params(hidden))))
        return c

    def initial(self) -> "dict[str, Any]":
        self._i = 1
        return self._cfg(self.hiddens[0])

    def propose_next(self, history) -> "dict[str, Any] | None":
        if self._i >= len(self.hiddens):
            return None
        cfg = self._cfg(self.hiddens[self._i])
        self._i += 1
        return cfg


__all__ = ["LearningRateSearch", "MixtureSearch", "ComputeAllocation"]
