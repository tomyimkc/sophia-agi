# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Competence self-model: where am I reliable, and route by it.

A primitive metacognition layer. Per domain it tracks (correct, total) outcomes ->
a reliability estimate, and routes a query to ANSWER only where reliability clears a
threshold, otherwise ABSTAIN/ESCALATE. Routing-by-competence should beat flat
always-answer once reliability varies across domains.
"""

from __future__ import annotations


class CompetenceMap:
    def __init__(self, *, threshold: float = 0.7, prior: float = 0.5, prior_weight: int = 2):
        self.threshold = threshold
        self.prior = prior
        self.prior_weight = prior_weight
        self._stats: dict = {}  # domain -> [correct, total]

    def update(self, domain: str, correct: bool) -> None:
        s = self._stats.setdefault(domain, [0, 0])
        s[0] += int(bool(correct))
        s[1] += 1

    def reliability(self, domain: str) -> float:
        c, t = self._stats.get(domain, [0, 0])
        # Beta-smoothed so a thin record is not over-trusted
        return round((c + self.prior * self.prior_weight) / (t + self.prior_weight), 4)

    def route(self, domain: str) -> str:
        """'answer' where the domain is reliably handled, else 'abstain'."""
        return "answer" if self.reliability(domain) >= self.threshold else "abstain"

    def map(self) -> "dict[str, float]":
        return {d: self.reliability(d) for d in self._stats}
