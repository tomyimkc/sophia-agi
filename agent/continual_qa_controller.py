"""Control-flow layer for CPQA: route a natural-language question to a fact id.

This is the "LLM as control flow" half of the design. The knowledge store
(``GraphBackedSystem``) only answers about a specific fact id; *deciding which fact a
question is about* is the controller's job — and, per limitation #1 of the failure
ledger, that decision needs an interpretive prior the store does not hold. Isolating it
here lets us measure the **control-flow error**: the gap between an oracle that always
routes correctly (pure substrate) and a real router.

Controllers share one contract: ``route(question, vocab, gold=None) -> id | None``,
where ``vocab`` maps every candidate fact id to its searchable text and ``None`` means
"abstain — I can't tell which fact this is about."

- ``OracleController``  — returns the gold target; upper bound (substrate-only error).
- ``LexicalController`` — deterministic token-overlap routing; offline, CI-able. The
  gap it opens vs the oracle is a *lower bound* on control-flow error (it has no NL
  understanding, only lexical overlap).
- ``LLMController``     — real LLM routing via ``agent.llm.complete``; gated behind an
  API key, never run in CI. Measures the true control-flow error.
"""

from __future__ import annotations

import re

# Generic words that carry no routing signal.
STOP = {
    "the", "a", "an", "of", "to", "in", "on", "is", "are", "was", "were", "did", "do",
    "does", "who", "what", "when", "where", "why", "how", "still", "know", "we", "it",
    "and", "or", "for", "by", "with", "about", "this", "that", "after", "claim", "view",
    "fact", "never", "taught", "corpus",
}


def _tokens(s: str) -> "set[str]":
    return {t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if t not in STOP}


class OracleController:
    """Always routes to the correct fact — measures the knowledge substrate alone."""

    name = "oracle"

    def route(self, question: str, vocab, *, gold=None):  # noqa: ARG002
        return gold


class LexicalController:
    """Deterministic token-overlap router. No NL understanding — a floor on routing."""

    name = "lexical"

    def __init__(self, min_overlap: int = 1) -> None:
        self.min_overlap = min_overlap

    def route(self, question: str, vocab, *, gold=None):  # noqa: ARG002
        q = _tokens(question)
        best, best_score = None, 0
        for cid, text in vocab.items():
            score = len(q & _tokens(text))
            if score > best_score:
                best, best_score = cid, score
        return best if best_score >= self.min_overlap else None


_SYSTEM = (
    "You route a question to exactly one knowledge-base entry id, or to NONE if no entry "
    "fits. Reply with only the id (or the word NONE). Do not explain."
)


class LLMController:
    """Real LLM routing. Requires an API key (see agent/llm.py); not run in CI."""

    name = "llm"

    def __init__(self, complete=None) -> None:
        self._complete = complete

    def route(self, question: str, vocab, *, gold=None):  # noqa: ARG002
        complete = self._complete
        if complete is None:
            from agent.llm import complete as complete  # noqa: PLC0415
        catalog = "\n".join(f"- {cid}: {text}" for cid, text in sorted(vocab.items()))
        user = f"Entries:\n{catalog}\n\nQuestion: {question}\n\nWhich entry id?"
        out = (complete(_SYSTEM, user) or "").strip().split()[0:1]
        token = out[0] if out else ""
        return token if token in vocab else None


__all__ = ["OracleController", "LexicalController", "LLMController"]
