# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""A minimal, correct, in-process Datalog evaluator (pure stdlib).

Purpose
-------
Sophia's strategic plan (Verifiable-Sophia, commit cb887e5) calls for the
machine-checkable parts of the fail-closed abstention rule to live in a logic
engine rather than in scattered Python conditionals — so that "the gate
abstains here" is a *derivable theorem*, not a regex side-effect. The repo is
airgap-first (its data pipeline documents stdlib-only core paths as a value),
so this module adds **no dependency**: it is a self-contained least-fixed-point
Datalog with stratified negation-as-failure, small enough to audit line by line.

This is the symbolic half of a neuro-symbolic split: the *neural/lexical* side
(Python, in :mod:`agent.datalog_provenance`) extracts ground facts from free
text; this engine derives the verdict from those facts via Horn clauses.

Semantics
---------
- **Facts** and **rule heads** are ground or partially-ground atoms:
  ``("forbidden", ("Dao De Jing", "Confucius"))``. Terms are arbitrary hashable
  Python objects (strings here); constants only — no function symbols, so the
  Herbrand universe is finite and the fixpoint is guaranteed to terminate.
- **Rules** are definite Horn clauses ``head :- body`` where the body is a
  conjunction of positive atoms and *negated* atoms (``not atom``). Variables
  are the set-difference of symbols appearing in the body vs. the constants;
  we use the convention that any token wrapped as ``Var("X")`` is a variable.
- **Negation-as-failure** is *stratified*: a rule may only negate a predicate
  from an **earlier** stratum (one fully computed before the negating stratum
  begins). This is the standard, decidable Datalog negation strategy and matches
  the closed-world reading the gate needs ("abstain if NOT derivably covered").
  Non-stratifiable programs raise (we never want a flaky fixpoint).

The evaluator is deliberately tiny and slow-is-fine: provenance gates run on
~handfuls of facts per answer, never the millions where Datalog shines. The
point is *auditability*, not throughput.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

Term = object  # hashable constant
Pred = str


@dataclass(frozen=True)
class Var:
    """A logic variable. Distinguished from constants by type, not by name."""

    name: str

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"?{self.name}"


@dataclass(frozen=True)
class Atom:
    """A positive literal: ``pred(args...)``. Args are constants or Vars."""

    pred: Pred
    args: tuple

    @classmethod
    def of(cls, pred: Pred, *args: Term) -> "Atom":
        return cls(pred, tuple(args))

    def neg(self) -> "NegAtom":
        return NegAtom(self.pred, self.args)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        inner = ", ".join(repr(a) for a in self.args)
        return f"{self.pred}({inner})"


@dataclass(frozen=True)
class NegAtom:
    """A negated literal under negation-as-failure: ``not pred(args...)``."""

    pred: Pred
    args: tuple

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        inner = ", ".join(repr(a) for a in self.args)
        return f"not {self.pred}({inner})"


def A(pred: Pred, *args: Term) -> Atom:
    """Shorthand constructor for a positive atom (matches Prolog surface feel)."""
    return Atom.of(pred, *args)


@dataclass
class Rule:
    """A definite Horn clause with possible negated body literals.

    ``head :- body[0], body[1], ..., body[n]``. An empty body means the head is
    a fact (though facts are normally added directly via :meth:`Program.fact`).
    """

    head: Atom
    body: tuple  # tuple of Atom | NegAtom

    @classmethod
    def clause(cls, head: Atom, *body: Atom | NegAtom) -> "Rule":
        return cls(head, tuple(body))


@dataclass
class Program:
    """A Datalog program: a set of facts + rules, evaluated to a fixpoint.

    Build it incrementally, then call :meth:`solve` to materialise every
    derivable fact (the least fixed point, stratified over negation).
    """

    facts: set[Atom] = field(default_factory=set)
    rules: list[Rule] = field(default_factory=list)

    def fact(self, atom: Atom) -> "Program":
        self.facts.add(atom)
        return self

    def facts_from(self, atoms: Iterable[Atom]) -> "Program":
        self.facts.update(atoms)
        return self

    def rule(self, rule: Rule) -> "Program":
        self.rules.append(rule)
        return self

    # --- stratification ----------------------------------------------------
    def _stratify(self) -> list[set[Pred]]:
        """Group predicates into strata so every negated literal reads an
        earlier stratum. Raises :class:`ValueError` if the program is not
        stratifiable (a negation cycle) — we refuse to evaluate those."""
        preds: set[Pred] = {a.pred for a in self.facts}
        preds |= {r.head.pred for r in self.rules}
        for r in self.rules:
            for lit in r.body:
                preds.add(lit.pred)
        # edges: head depends-on body. Positive edges don't constrain strata;
        # a NEGATIVE edge head -> body requires stratum(head) > stratum(body).
        neg_deps: dict[Pred, set[Pred]] = {p: set() for p in preds}
        for r in self.rules:
            for lit in r.body:
                if isinstance(lit, NegAtom):
                    neg_deps[r.head.pred].add(lit.pred)
        # assign strata by longest-path over the neg-dependency DAG
        stratum: dict[Pred, int] = {}

        def s(p: Pred, stack: tuple) -> int:
            if p in stratum:
                return stratum[p]
            if p in stack:
                cyc = " -> ".join(stack[stack.index(p):] + (p,))
                raise ValueError(f"program is not stratifiable (negation cycle): {cyc}")
            d = neg_deps[p]
            level = 0 if not d else 1 + max(s(q, stack + (p,)) for q in d)
            stratum[p] = level
            return level

        for p in preds:
            s(p, ())
        n = (max(stratum.values()) + 1) if stratum else 1
        layers: list[set[Pred]] = [set() for _ in range(n)]
        for p, i in stratum.items():
            layers[i].add(p)
        return layers

    # --- evaluation --------------------------------------------------------
    def solve(self) -> set[Atom]:
        """Return the least fixed point: every derivable ground atom.

        Stratified: predicates in stratum 0 are computed to fixpoint before any
        rule whose head is in stratum 1 (which may negate stratum-0 preds) fires.
        Negation-as-failure then reads a fully-materialised relation.

        Uses a predicate-indexed store (``dict[Pred, set[Atom]]``) so a body
        literal only iterates the facts of ITS predicate, not the whole known-set.
        This is the standard Datalog indexing move — pure correctness-preserving
        (identical answers, far fewer iterations); without it the provenance audit
        takes ~25 min because ~85 static ``forbidden`` facts get rescanned per
        literal per text. Semantics are unchanged.
        """
        layers = self._stratify()
        idx: dict[Pred, set[Atom]] = {}
        for a in self.facts:
            idx.setdefault(a.pred, set()).add(a)
        for layer in layers:
            if not layer:
                continue
            layer_rules = [r for r in self.rules if r.head.pred in layer]
            if not layer_rules:
                continue
            # Repeat to fixpoint. Match against a SNAPSHOT of the index each pass
            # (we must not mutate a structure while iterating it); merge newly
            # derived atoms afterwards. Naive evaluation is fine here.
            changed = True
            while changed:
                changed = False
                snapshot = {p: set(s) for p, s in idx.items()}
                derived: set[Atom] = set()
                for rule in layer_rules:
                    for binding in _match_body(rule.body, snapshot):
                        derived.add(_substitute(rule.head, binding))
                new = {a for a in derived if a not in idx.get(a.pred, ())}
                if new:
                    for a in new:
                        idx.setdefault(a.pred, set()).add(a)
                    changed = True
        return {a for s in idx.values() for a in s}

    def query(self, atom: Atom) -> set[tuple]:
        """Return all arg-tuples for which ``atom.pred`` is derivable (constants
        in ``atom.args`` act as filters; Vars are the query's unknowns)."""
        sol = self.solve()
        out: set[tuple] = set()
        for fact in sol:
            if fact.pred != atom.pred:
                continue
            if len(fact.args) != len(atom.args):
                continue
            ok = True
            for fa, qa in zip(fact.args, atom.args):
                if isinstance(qa, Var):
                    continue
                if fa != qa:
                    ok = False
                    break
            if ok:
                out.add(fact.args)
        return out


# --- unification / body matching -------------------------------------------
def _match_body(body: tuple, idx: dict[Pred, set[Atom]]) -> Iterable[dict]:
    """Yield every variable binding under which every literal in ``body`` holds.

    ``idx`` is a predicate-indexed store: a positive literal only consults
    ``idx[lit.pred]``, never the full fact set. Negated literals succeed under
    negation-as-failure when NO grounding of their args is in their slice.
    """
    yield from _match_seq(body, 0, {}, idx)


def _match_seq(body: tuple, i: int, binding: dict, idx: dict[Pred, set[Atom]]) -> Iterable[dict]:
    if i >= len(body):
        yield dict(binding)
        return
    lit = body[i]
    if isinstance(lit, NegAtom):
        if _naf_holds(lit, binding, idx):
            yield from _match_seq(body, i + 1, binding, idx)
        return
    # positive atom: unify against each compatible known fact OF THIS PREDICATE
    for fact in idx.get(lit.pred, ()):
        if len(fact.args) != len(lit.args):
            continue
        b2 = _unify(lit.args, fact.args, binding)
        if b2 is not None:
            yield from _match_seq(body, i + 1, b2, idx)


def _unify(pattern: tuple, ground: tuple, binding: dict) -> dict | None:
    """Unify a (possibly variable-bearing) pattern against a ground tuple,
    extending ``binding``. Return None on conflict."""
    b = dict(binding)
    for p, g in zip(pattern, ground):
        if isinstance(p, Var):
            if p.name in b:
                if b[p.name] != g:
                    return None
            else:
                b[p.name] = g
        elif p != g:
            return None
    return b


def _naf_holds(lit: NegAtom, binding: dict, idx: dict[Pred, set[Atom]]) -> bool:
    """Negation-as-failure: true iff NO known fact (in this literal's predicate
    slice) matches under the current binding (or any extension of it for still-
    free variables)."""
    for fact in idx.get(lit.pred, ()):
        if len(fact.args) != len(lit.args):
            continue
        if _unify(lit.args, fact.args, binding) is not None:
            return False
    return True


def _substitute(atom: Atom, binding: dict) -> Atom:
    """Ground an atom's variables from ``binding``; leave constants unchanged.
    A head variable not bound by the body is a malformed rule (unsafe) — we drop
    it by returning the atom as-is, which then won't match any ground query."""
    args = tuple(binding.get(a.name, a) if isinstance(a, Var) else a for a in atom.args)
    return Atom(atom.pred, args)
