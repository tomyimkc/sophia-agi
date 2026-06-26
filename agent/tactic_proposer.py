# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""LLM tactic proposer for Lean proof search (Path B).

Maps Sophia's model adapter onto the `TacticProposer` contract in
`agent.proof_search`: given (theorem, current_proof_state), propose a list of
candidate Lean 4 tactic strings. Default is a deterministic stub (CI-safe, no
model call) so the search STRUCTURE is testable; the live path wires
`agent.model.default_client`.

This is the "tactic generator" half of the ReProver/AlphaProof recipe — the other
halves are premise retrieval (ReProver-style) and Lean verification
(``lean_backend``). The proposer NEVER decides correctness — it only suggests;
Lean is the verifier.
"""

from __future__ import annotations

import re
from typing import Callable

from agent.proof_search import TacticProposer

# A model client duck-types: client.generate(system, user) -> ModelResult with .text
ModelGenerate = Callable[[str, str], str]

_SYSTEM = (
    "You are a Lean 4 theorem prover. Given the current proof state, propose up to 5 "
    "candidate next tactics — ONE PER LINE, no commentary. Each line is a single Lean 4 "
    "tactic (e.g. 'induction n', 'simp', 'apply Nat.add_comm'). Prefer standard library "
    "tactics. Do NOT explain; output only the tactic lines."
)


def _extract_tactics(text: str, *, max_n: int = 5) -> list[str]:
    """Parse model output into clean tactic lines. Filters code fences + prose.

    A real tactic line is a single Lean 4 tactic (possibly with arguments): e.g.
    ``induction n``, ``rw [Nat.add_comm]``, ``exact h``. This filters out fence
    markers, language tags, prose, numbering, and comments."""
    out: list[str] = []
    in_fence = False
    for raw in (text or "").splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        line = line.strip("`").strip()
        if not line or line.startswith(("#", "--", "//")):
            continue
        # drop leading bullets/numbering: "1. ", "- ", "1) ", "* "
        line = re.sub(r"^\s*(?:\d+[.)]?|[-*])\s*", "", line)
        # drop trailing comments
        line = line.split("--")[0].strip()
        if not line:
            continue
        # a bare fence-language tag like 'lean'/'python' is not a tactic
        if line.lower() in {"lean", "python", "text", "raw"}:
            continue
        # skip prose lead-ins
        low = line.lower()
        if low.startswith(("here", "sure", "candidate", "explanation", "tactic", "i ", "these ")):
            continue
        out.append(line)
        if len(out) >= max_n:
            break
    return out


def make_llm_proposer(generate: ModelGenerate, *, max_n: int = 5) -> TacticProposer:
    """Build a TacticProposer backed by an injected model `generate(system, user) -> text`.

    The generator is injected so tests pass a deterministic stub and the live path
    passes `lambda s,u: default_client().generate(s,u).text`."""

    def propose(theorem: str, state: str) -> list[str]:
        user = f"## Theorem\n{theorem}\n\n## Current proof state\n{state or '(initial)'}\n\nPropose up to {max_n} next tactics, one per line:"
        text = generate(_SYSTEM, user)
        return _extract_tactics(text, max_n=max_n)

    return propose


# A deterministic stub proposer for CI: returns a small fixed set of common opening
# tactics. NEVER claims to prove anything — just gives the search something to branch on.
# Includes `trivial`/`decide` so the bundled `trivial_true` rehearsal theorem is
# provable by the stub (exercising the search's `proved` path in the smoke test).
_STUB_TACTICS = ["intro", "induction n", "simp", "rw", "apply", "exact", "rfl", "trivial", "decide"]


def stub_proposer(theorem: str, state: str) -> list[str]:
    """Deterministic CI-safe proposer: a fixed small set of common Lean tactics.

    Used by tests and the no-model dry-run path. It will NOT prove nontrivial
    theorems — that's the point (the search should abstain within budget, proving
    the fail-closed path). A real proof needs the LLM proposer."""
    return list(_STUB_TACTICS)


def default_proposer(*, model_spec: str | None = None) -> TacticProposer:
    """Build the default proposer: LLM-backed if a model is available, else the stub.

    Mirrors agent.model's fail-closed pattern: if the resolved provider kind is
    'mock' (the CI/offline default) or unavailable, fall back to the deterministic
    stub rather than emit non-tactic prose parsed as fake tactics."""
    try:
        from agent.model import default_client, resolve_config

        cfg = resolve_config(model_spec)
        if cfg.kind == "mock":
            return stub_proposer  # deterministic in CI / no model available
        client = default_client(model_spec)
        return make_llm_proposer(lambda s, u: client.generate(s, u).text)
    except Exception:
        return stub_proposer  # no model available -> CI-safe stub


__all__ = ["make_llm_proposer", "stub_proposer", "default_proposer"]
