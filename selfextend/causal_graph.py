"""A causal world model — beyond provenance ('who said what') to 'what causes what'.

Sophia's belief graph tracks where a claim came from; reasoning needs to predict the
effect of an *intervention*. This is a small linear structural causal model with the
do-operator. The point it proves: under confounding, the observational association
between X and Y (what correlational models see) differs from the causal effect of
do(X) — and this model returns the causal effect, where a correlational baseline is
biased. Deterministic, dependency-free.
"""

from __future__ import annotations


class CausalGraph:
    def __init__(self):
        self.parents: dict = {}    # node -> [(parent, weight)]
        self.children: dict = {}   # node -> [(child, weight)]
        self.nodes: set = set()

    def add_edge(self, cause: str, effect: str, weight: float = 1.0) -> "CausalGraph":
        self.nodes.update((cause, effect))
        self.parents.setdefault(effect, []).append((cause, weight))
        self.children.setdefault(cause, []).append((effect, weight))
        self.parents.setdefault(cause, self.parents.get(cause, []))
        return self

    def _topo(self) -> list:
        seen, order = set(), []

        def visit(n):
            if n in seen:
                return
            seen.add(n)
            for p, _ in self.parents.get(n, []):
                visit(p)
            order.append(n)

        for n in self.nodes:
            visit(n)
        return order

    def _reduced_form(self) -> dict:
        """Each node as a linear combination of independent unit-variance exogenous
        noises (one per node). Enables closed-form covariance."""
        rf: dict = {}
        for n in self._topo():
            coef: dict = {n: 1.0}  # own exogenous noise
            for p, w in self.parents.get(n, []):
                for e, c in rf.get(p, {}).items():
                    coef[e] = coef.get(e, 0.0) + w * c
            rf[n] = coef
        return rf

    def _cov(self, a: str, b: str, rf: dict) -> float:
        ca, cb = rf[a], rf[b]
        return sum(v * cb.get(e, 0.0) for e, v in ca.items())

    def observational_coef(self, x: str, y: str) -> float:
        """Naive univariate regression coef of y on x (what a correlational model
        learns) — confounded by backdoor paths."""
        rf = self._reduced_form()
        vx = self._cov(x, x, rf)
        return round(self._cov(x, y, rf) / vx, 4) if vx else 0.0

    def causal_effect(self, x: str, y: str) -> float:
        """Effect of do(x) on y: sum over DIRECTED x->...->y paths of weight products
        (intervention cuts x's parents, so backdoor paths vanish)."""
        memo: dict = {}

        def rec(n):
            if n == y:
                return 1.0
            if n in memo:
                return memo[n]
            memo[n] = sum(w * rec(c) for c, w in self.children.get(n, []))
            return memo[n]

        return round(sum(w * rec(c) for c, w in self.children.get(x, [])), 4)

    def confounded(self, x: str, y: str, *, tol: float = 1e-6) -> bool:
        """True when correlation misleads: observational association != causal effect."""
        return abs(self.observational_coef(x, y) - self.causal_effect(x, y)) > tol
