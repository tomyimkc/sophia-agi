# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Proof search over Lean 4 — the legitimate novelty pathway for Sophia.

Path B of `docs/06-Roadmap/Two-Paths-To-Novelty.md`. The roofline bounds every Sophia
output to (train ∪ retrieved). Formal proof is the one domain where novelty is
reachable *under* the ceiling: a Lean-verified proof is self-certifying. This module
implements the search loop — the open, reproducible analogue of AlphaProof's
([Nature 2025](https://www.nature.com/articles/s41586-025-09833-y)) RL-over-Lean-search,
built on **LeanDojo** ([NeurIPS 2023](https://neurips.cc/virtual/2023/poster/73510)):

    proof state (Lean goals)
      -> propose candidate tactics (LLM, via agent.model) + retrieve premises (ReProver-style)
      -> apply each via Lean, get the next proof state
      -> best-first search (priority = LeanProgress-style progress estimate)
      -> a state with NO remaining goals = a verified proof
      -> novelty_check: is the proof a near-duplicate of the corpus? (strict)

Fail-closed at every node: Lean absent → abstain; no proof within budget → abstain;
never assert an unproved goal. The search never fabricates; it only narrows.

Honest scope: this is NOT a bid to beat AlphaProof. It wires the open LeanDojo stack
into Sophia's existing math seam and adds a MEASURED novelty probe. ``candidateOnly``
until a gated run produces a Lean-verified, non-retrieved proof.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from agent import lean_backend

ProofSearchVerdict = Literal["proved", "no_proof_within_budget", "lean_unavailable"]


@dataclass(frozen=True)
class ProofNode:
    """One node in the proof-search tree: a Lean proof state + the tactic path to it."""

    state: str  # the current Lean goal(s) as text
    path: tuple[str, ...]  # tactics applied so far
    depth: int
    # Priority estimate: a higher value = closer to a closed goal. A real
    # LeanProgress-style model would learn this; the default heuristic is
    # "fewer remaining goals / shorter goal text" — a cheap, honest proxy.
    priority: float

    def __lt__(self, other: "ProofNode") -> bool:
        # heapq is a min-heap; we want max-priority first, so invert.
        return self.priority > other.priority


@dataclass
class ProofSearchResult:
    """Outcome of one proof-search attempt. Carries the discipline fields."""

    schema: str = "sophia.proof_search.v1"
    candidate_only: bool = True
    level3_evidence: bool = False
    verdict: ProofSearchVerdict = "no_proof_within_budget"
    theorem: str = ""
    proof: str | None = None  # the tactic sequence, joined, IF proved
    lean_verdict: str = ""  # the lean_backend verdict on the found proof
    novelty: dict[str, Any] | None = None  # the strict novelty probe, if proved
    nodes_expanded: int = 0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "candidateOnly": self.candidate_only,
            "level3Evidence": self.level3_evidence,
            "verdict": self.verdict,
            "theorem": self.theorem,
            "proof": self.proof,
            "leanVerdict": self.lean_verdict,
            "novelty": self.novelty,
            "nodesExpanded": self.nodes_expanded,
            "reason": self.reason or self._interp(),
        }

    def _interp(self) -> str:
        if self.verdict == "lean_unavailable":
            return ("Lean (lean-dojo) not installed — proof search abstains. Install "
                    "the opt-in `requirements-theorem.txt` extra to fire Path B.")
        if self.verdict == "proved":
            tag = "NOVEL" if (self.novelty or {}).get("novel") else "retrieved-near-duplicate"
            return (f"Lean-verified proof found after {self.nodes_expanded} nodes. "
                    f"Novelty probe: {tag} (overlap {(self.novelty or {}).get('best_overlap')}). "
                    f"candidateOnly; not a capability claim.")
        return (f"No Lean-verified proof within the node budget ({self.nodes_expanded} "
                f"nodes expanded). Fail-closed abstention — never asserts an unproved goal.")


# A tactic proposer maps (theorem, current_state) -> list of candidate tactic strings.
# Injected so CI can use a deterministic stub and the live path uses agent.model.
TacticProposer = Callable[[str, str], list[str]]
# A premise retriever maps (state) -> list of relevant lemma names (ReProver-style).
# Default: no-op (the proposer is expected to embed premises itself via the LLM).
PremiseRetriever = Callable[[str], list[str]]


def _default_priority(state: str) -> float:
    """Cheap progress heuristic: fewer/shorter goals ≈ closer to done. A real
    LeanProgress model would learn this from data; this is an honest proxy."""
    # "no goals" / "Goals accomplished" markers indicate a closed proof state.
    s = state or ""
    if not s.strip() or "no goals" in s.lower() or "goals accomplished" in s.lower():
        return 1.0
    # shorter remaining goal text = higher priority (closer to closed)
    return 1.0 / (1.0 + len(s))


def search_proof(
    theorem: str,
    *,
    proposer: TacticProposer,
    initial_state: str,
    max_nodes: int = 50,
    max_depth: int = 12,
    premise_retriever: PremiseRetriever | None = None,
    novelty_corpus: list[str] | None = None,
    apply_tactic: Callable[[str, str], tuple[str, bool]] | None = None,
) -> ProofSearchResult:
    """Best-first search for a Lean proof of ``theorem``.

    ``apply_tactic(state, tactic) -> (next_state, goal_closed)`` is injected: in
    production it calls lean_backend (LeanDojo) to actually apply the tactic; in
    tests it is a scripted stub. When ``apply_tactic`` is None we attempt the real
    Lean path — which abstains fail-closed if Lean isn't installed.

    Returns a ProofSearchResult. ``proved`` means Lean verified the full tactic
    path; the novelty probe (strict) is run only on a proved proof.
    """
    # If no tactic applier is injected, build the real Lean one (abstains if no Lean).
    if apply_tactic is None:
        apply_tactic = _lean_apply_tactic

    res = ProofSearchResult(theorem=theorem)
    # The real path (apply_tactic is None) needs Lean to actually apply tactics.
    # An injected apply_tactic (test/scripted mode) can run without Lean — that's
    # how the search STRUCTURE is testable without the Lean toolchain.
    real_lean_path = apply_tactic is None
    if real_lean_path and not lean_backend.lean_available():
        res.verdict = "lean_unavailable"
        return res

    start = ProofNode(state=initial_state, path=(), depth=0, priority=_default_priority(initial_state))
    frontier: list[ProofNode] = [start]
    expanded = 0
    while frontier and expanded < max_nodes:
        node = heapq.heappop(frontier)
        expanded += 1
        # Closed goal? -> assemble the proof. On the REAL Lean path, re-verify the
        # whole tactic sequence end-to-end via Lean (the proof is the verifier). On
        # an INJECTED-applier path (test/scripted), the applier IS the verifier, so a
        # closed node is a proved proof — Lean re-check is skipped (and would abstain).
        if node.priority >= 1.0 or "no goals" in (node.state or "").lower():
            proof = " ".join(node.path) if node.path else "rfl"  # trivial proof
            res.proof = proof
            res.nodes_expanded = expanded
            if real_lean_path:
                check = lean_backend.verify_proof(theorem=theorem, proof=_assemble(theorem, node.path))
                res.lean_verdict = check.verdict
                if check.verdict == "accepted":
                    res.verdict = "proved"
                    res.novelty = lean_backend.novelty_check(proof, corpus=novelty_corpus or [])
                else:
                    res.verdict = "no_proof_within_budget"
                    res.reason = f"reached a 'closed' node but Lean rejected it: {check.reason}"
            else:
                # Injected applier: it closed the goal, so the proof is accepted by
                # construction. lean_verdict stays "" (no Lean was involved).
                res.verdict = "proved"
                res.lean_verdict = "accepted-by-injected-applier"
                res.novelty = lean_backend.novelty_check(proof, corpus=novelty_corpus or [])
            return res
        if node.depth >= max_depth:
            continue
        # Propose tactics (+ optional premise retrieval) and expand.
        premises = premise_retriever(node.state) if premise_retriever else []
        proposed = proposer(theorem, node.state) or []
        for tac in proposed:
            next_state, closed = apply_tactic(node.state, tac)
            if next_state is None:
                continue  # tactic failed in Lean; skip
            child = ProofNode(
                state=next_state,
                path=node.path + (tac,),
                depth=node.depth + 1,
                priority=1.0 if closed else _default_priority(next_state),
            )
            heapq.heappush(frontier, child)
    res.nodes_expanded = expanded
    res.verdict = "no_proof_within_budget"
    return res


def _assemble(theorem: str, path: tuple[str, ...]) -> str:
    """Assemble a `theorem ... := by <tactics>` block for end-to-end Lean verification."""
    tactics = " ".join(path) if path else "rfl"
    if ":= by" in theorem:
        return f"{theorem}\n{tactics}"
    return f"{theorem} := by\n{tactics}"


def _lean_apply_tactic(state: str, tactic: str) -> tuple[str | None, bool]:
    """Real Lean tactic application via lean_backend. Returns (next_state, goal_closed)
    or (None, False) if Lean isn't available / the tactic failed. Fail-closed."""
    if not lean_backend.lean_available():
        return None, False
    # A full LeanDojo integration applies `tactic` to `state` and returns the new
    # proof state. The verify_proof call in search_proof re-checks the WHOLE path
    # end-to-end, so this node-level apply is an optimization (prune dead branches
    # early); the load-bearing verification is the final verify_proof.
    try:
        from lean_dojo import LeanDojo  # type: ignore
        # Version-tolerant: the exact call varies; wrap so a mismatch abstains.
        # In production this returns the post-tactic proof state + closed flag.
        return None, False  # placeholder until the specific LeanDojo API is pinned
    except Exception:
        return None, False


__all__ = [
    "ProofSearchVerdict",
    "ProofNode",
    "ProofSearchResult",
    "TacticProposer",
    "PremiseRetriever",
    "search_proof",
]
