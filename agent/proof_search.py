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
    lean_session: "LeanProofSession | None" = None,
) -> ProofSearchResult:
    """Best-first search for a Lean proof of ``theorem``.

    ``apply_tactic(state, tactic) -> (next_state, goal_closed)`` is injected: in
    production it calls lean_backend (LeanDojo) to actually apply the tactic; in
    tests it is a scripted stub. When ``apply_tactic`` is None we attempt the real
    Lean path — which abstains fail-closed if Lean isn't installed.

    For the real Lean path, pass a ``lean_session`` (a `LeanProofSession` opened on
    the theorem's source); the search uses `lean_session.apply` as the tactic
    applier so LeanDojo's stateful `proof_state` object is threaded across calls.

    Returns a ProofSearchResult. ``proved`` means Lean verified the full tactic
    path; the novelty probe (strict) is run only on a proved proof.
    """
    res = ProofSearchResult(theorem=theorem)
    # If a real Lean session is supplied, use it as the tactic applier (stateful).
    if lean_session is not None:
        apply_tactic = lean_session.apply
        real_lean_path = True
    elif apply_tactic is None:
        # The stateless real path needs Lean — but cannot thread proof_state, so it
        # abstains. Callers wanting the real path must pass a lean_session.
        res.verdict = "lean_unavailable" if not lean_backend.lean_available() else "lean_unavailable"
        res.reason = ("real Lean path requires a lean_session (LeanDojo is stateful; "
                      "pass LeanProofSession). An injected apply_tactic runs the test path.")
        return res
    else:
        real_lean_path = False  # injected applier (test/scripted) — applier is the verifier

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
            proof = "\n".join(node.path) if node.path else "rfl"  # trivial proof
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
    """Assemble a `theorem ... := by <tactics>` block for end-to-end Lean verification.

    Tactics are newline-separated: a Lean 4 ``by`` block runs each tactic on its own
    line (space-joining multi-tactic sequences produces invalid syntax for anything
    beyond a single tactic, which would make Lean reject otherwise-correct proofs and
    corrupt the novelty-probe text). An empty path falls back to ``rfl``.
    """
    tactics = "\n".join(path) if path else "rfl"
    if ":= by" in theorem:
        return f"{theorem}\n{tactics}"
    return f"{theorem} := by\n{tactics}"


def _lean_apply_tactic(state: str, tactic: str) -> tuple[str | None, bool]:
    """Stateless fallback — deprecated in favor of LeanProofSession. Kept so the
    old call signature still works; the stateful path is required because LeanDojo
    carries a `proof_state` OBJECT across tactic applications, not a state string."""
    if not lean_backend.lean_available():
        return None, False
    # A stateless apply cannot carry the proof_state object LeanDojo needs, so we
    # refuse rather than fake it. Callers wanting the real Lean path must use
    # LeanProofSession (below) which threads the proof_state object through the search.
    return None, False


class LeanProofSession:
    """A stateful Lean tactic-application session. LeanDojo's API is STATEFUL: each
    `run_tac(proof_state, tactic)` returns a NEW proof_state object that must be
    threaded to the next call. The best-first search in `search_proof` works over
    state STRINGS (for priority heuristics + logging), so this session adapts the
    string-keyed search to LeanDojo's object-keyed API via a proof_state cache.

    Usage: the search calls `session.apply(state_str, tactic)`; the session looks up
    the proof_state object keyed by state_str, applies the tactic via LeanDojo, caches
    the new proof_state under its rendered string, and returns (next_state_str, closed).
    """

    def __init__(self, repo_url: str = "https://github.com/leanprover-community/mathlib4"):
        self._dojo = None
        self._states: dict[str, object] = {}  # state_str -> LeanDojo proof_state object
        self._repo_url = repo_url
        self._open = False

    def open(self, initial_state_str: str, theorem_source: str) -> bool:
        """Open a LeanDojo session on `theorem_source` and record its initial proof
        state under `initial_state_str`. Returns False (fail-closed) if Lean is absent
        or the session can't open.

        lean-dojo 4.x API: `Dojo(entry, ...)` where entry is a `Theorem` or a
        `(LeanGitRepo, Path, line)` tuple; `LeanGitRepo(url, commit)` (NOT the
        pre-4.x `LeanRepo` name). The session is stateful; `apply` threads the
        proof_state object it returns. Wrapped defensively so any version mismatch
        abstains (returns False) rather than crashes.
        """
        if not lean_backend.lean_available():
            return False
        try:
            # lean-dojo 4.x renamed LeanRepo -> LeanGitRepo. Import both names and use
            # whichever exists, so a version skew (4.x vs older) abstains instead of crashing.
            import lean_dojo as _ldj  # type: ignore
            Dojo = _ldj.Dojo
            LeanGitRepo = getattr(_ldj, "LeanGitRepo", None) or getattr(_ldj, "LeanRepo", None)
            if LeanGitRepo is None:
                self._open = False
                return False
            repo = LeanGitRepo(self._repo_url, "master")
            # Dojo's entry can be a Theorem (repo, file_path, full_name). theorem_source
            # here is the caller's theorem string; a full integration constructs a
            # Theorem against a traced repo. We attempt the open defensively.
            self._dojo = Dojo(repo)
            ps = self._dojo.run_tac(theorem_source) if hasattr(self._dojo, "run_tac") else None
            if ps is not None:
                self._states[initial_state_str] = ps
                self._open = True
            return self._open
        except Exception:
            self._open = False
            return False

    def apply(self, state_str: str, tactic: str) -> tuple[str | None, bool]:
        """Apply `tactic` to the proof_state cached under `state_str`. Returns
        (next_state_str, goal_closed) or (None, False) on any failure (fail-closed)."""
        if self._dojo is None:
            return None, False
        ps = self._states.get(state_str)
        if ps is None:
            return None, False  # unknown state string -> can't apply
        try:
            result = self._dojo.run_tac(ps, tactic)  # type: ignore[attr-defined]
        except Exception:
            return None, False  # tactic failed in Lean (or API mismatch) -> prune
        # LeanDojo result types: TacticSuccess (carries new state) / TacticFailure / etc.
        # A closed goal is signaled by an empty-goals state. Be defensive across versions.
        new_ps = getattr(result, "ps", getattr(result, "proof_state", None))
        if new_ps is None:
            # tactic produced no next state (failure or no-op) -> prune
            return None, False
        new_str = self._render(new_ps)
        self._states[new_str] = new_ps
        closed = self._is_closed(new_ps)
        return new_str, closed

    @staticmethod
    def _render(proof_state: object) -> str:
        """Render a LeanDojo proof state to a string for the search's priority/log."""
        for attr in ("pp", "goals", "state", "content"):
            v = getattr(proof_state, attr, None)
            if isinstance(v, str) and v.strip():
                return v
            if isinstance(v, (list, tuple)) and v:
                return "\n".join(str(g) for g in v)
        return str(proof_state)

    @staticmethod
    def _is_closed(proof_state: object) -> bool:
        """A proof state with no remaining goals = closed. Defensive across versions."""
        goals = getattr(proof_state, "goals", None)
        if goals is not None:
            return len(goals) == 0
        # Fallback: the rendered string indicates completion.
        return "no goals" in LeanProofSession._render(proof_state).lower()

    def close(self) -> None:
        """Release the LeanDojo session."""
        try:
            if self._dojo is not None and hasattr(self._dojo, "close"):
                self._dojo.close()
        except Exception:
            pass
        self._dojo = None
        self._open = False


__all__ = [
    "ProofSearchVerdict",
    "ProofNode",
    "ProofSearchResult",
    "TacticProposer",
    "PremiseRetriever",
    "search_proof",
]
