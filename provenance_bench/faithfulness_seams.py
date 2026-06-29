# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Live seams for the retrieve-then-reason rollout — adapters onto the repo's
real retrieval + verification machinery.

``faithfulness_rollout.rollout`` takes injectable ``retrieve`` / ``generate`` /
``extract_claims`` / ``verify_claim`` seams. The offline path uses deterministic
mocks; this module wires the *live* seams onto what already ships:

  retrieve      -> agent.ai_search.search (hybrid dense+sparse over the committed
                   RAG index; provenance carried forward from OKF frontmatter). This
                   adapter runs OFFLINE on CPU — no model needed.
  verify_claim  -> an entailment_fn in the agent.source_verifier idiom
                   ((claim_text, source_text) -> "entails"|"contradicts"|"irrelevant").
                   The entailment_fn itself is the one LLM seam; a deterministic mock
                   is provided for the conformance check.
  extract_claims-> a deterministic attribution-claim extractor (the default); the
                   richer LLM decomposition is the live upgrade.

``generate`` (the policy under training) is the only seam that intrinsically needs
the model, so it stays injected — there is no live default here. ``conformance_check``
proves every adapter conforms to the rollout interface end-to-end (offline), so the
only ungated pieces are the policy and a real entailment LLM.
"""

from __future__ import annotations

import re
from typing import Any, Callable

# authorConfidence used when a retrieved chunk carries no OKF provenance. Unknown
# provenance is the WEAKEST rank (okf.schema CONFIDENCE_RANK "none_extant" == 0), so
# a claim resting on it can never be laundered into a confident assertion.
_UNKNOWN_CONFIDENCE = "none_extant"


def make_ai_search_retrieve(*, top_k: int = 8, client: Any | None = None) -> Callable[[str], list]:
    """Live ``retrieve(query) -> list[Chunk]`` backed by ``agent.ai_search.search``.

    Each ``SourceChunk`` becomes ``{chunk_id, text, author_confidence}`` — the shape
    the rollout + reward expect, with provenance carried forward (falling back to the
    weakest rank when a chunk has no OKF frontmatter)."""
    from agent import ai_search

    def retrieve(query: str) -> list:
        result = ai_search.search(query, top_k=top_k, client=client)
        out = []
        for c in result.chunks:
            chunk_id = c.page_id or f"{c.path}::{c.title}"
            out.append({
                "chunk_id": chunk_id,
                "text": c.excerpt,
                "author_confidence": c.author_confidence or _UNKNOWN_CONFIDENCE,
            })
        return out

    return retrieve


def make_entailment_verify(
    entailment_fn: Callable[[str, str], str],
    *,
    scope: str = "cited",
) -> Callable[[dict, list], str]:
    """Live ``verify_claim(claim, context) -> verdict`` from an entailment fn.

    ``entailment_fn(claim_text, source_text) -> "entails"|"contradicts"|"irrelevant"``
    is the ``agent.source_verifier`` contract. A claim is ``contradicted`` if ANY
    in-scope chunk contradicts it (fail-closed), ``supported`` if any entails it, else
    ``unsupported``. ``scope="cited"`` checks only the chunks the claim cites (falling
    back to all context if it cites none); ``scope="all"`` checks every context chunk."""
    def verify_claim(claim: dict, context: list) -> str:
        cited = set(claim.get("support_chunk_ids") or [])
        if scope == "cited" and cited:
            texts = [c["text"] for c in context if c["chunk_id"] in cited]
        else:
            texts = [c["text"] for c in context]
        verdicts = [entailment_fn(claim.get("text", ""), t) for t in texts]
        if any(v == "contradicts" for v in verdicts):
            return "contradicted"
        if any(v == "entails" for v in verdicts):
            return "supported"
        return "unsupported"

    return verify_claim


# Active voice: "<Author> wrote/authored ...". Passive: "written/authored by <Author>".
_NAME = r"[A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+){0,3}"
_ATTRIB_ACTIVE = re.compile(rf"\b({_NAME})\s+(?:wrote|authored|composed|penned|created)\b")
_ATTRIB_PASSIVE = re.compile(rf"\b(?:written|authored|composed|penned|created)\s+by\s+({_NAME})")


def heuristic_extract_claims(answer: str, context: list) -> list:
    """Deterministic default ``extract_claims`` — surfaces attribution claims (both
    "<Author> wrote ..." and "... written by <Author>") and grounds each in the context
    chunks that mention that author (so the citation is anchored in retrieval, not
    invented). The richer LLM decomposition (multi-claim, non-attribution) is the live
    upgrade; this keeps the offline path honest and dependency-free. Dedupes by author."""
    claims, seen = [], set()
    for rx in (_ATTRIB_ACTIVE, _ATTRIB_PASSIVE):
        for m in rx.finditer(answer or ""):
            author = m.group(1).strip().rstrip(".,;:'\"")
            low = author.lower()
            if low in seen:
                continue
            seen.add(low)
            support = [c["chunk_id"] for c in context if low in (c.get("text") or "").lower()]
            claims.append({
                "text": m.group(0).strip(),
                "key": f"author={low}",
                "kind": "knowledge",
                "support_chunk_ids": support,
                "asserted_confidence": None,  # unknown -> weakest rank; no laundering
            })
    return claims


# --------------------------------------------------------------------------- #
# Conformance check — proves the live adapters conform to the rollout interface
# end-to-end, offline (no policy model, no entailment LLM, no GPU).
# --------------------------------------------------------------------------- #

def lexical_entailment(claim_text: str, source_text: str) -> str:
    """Deterministic placeholder for the entailment LLM: an attribution-aware lexical
    heuristic ((author in source) -> entails). Handles BOTH the active ("<Author> wrote
    ...") and passive ("written by <Author>") attribution forms that
    heuristic_extract_claims emits, so the offline path is honest for either. Lets the
    rollout reward be computed on-box without a second model, so the GRPO loop can run; a
    real entailment LLM is the live upgrade (Open in the ledger). Returns
    "entails"|"contradicts"|"irrelevant"."""
    ct, st = claim_text.lower(), source_text.lower()
    passive = re.search(r"(?:written|authored|composed|penned|created)\s+by\s+(.+)$", ct)
    if passive:                       # "written by <author>" -> author follows "by"
        author = passive.group(1)
    else:                             # "<author> wrote/authored/..." -> author precedes verb
        author = re.split(r"\s+(?:wrote|authored|composed|penned|created)\b", ct, maxsplit=1)[0]
    author = author.strip().rstrip(".,;:'\"")
    if author and author in st:
        return "entails"
    return "irrelevant"


# Back-compat alias used by conformance_check.
_mock_entailment = lexical_entailment


_ENTAILMENT_SYS = (
    "You are a strict entailment checker for a retrieval-grounded reasoning system. "
    "Given a CLAIM and a SOURCE, decide whether the SOURCE supports the CLAIM. Reply with "
    "EXACTLY one lowercase word and nothing else: 'entails' (the source supports the claim), "
    "'contradicts' (the source refutes it), or 'irrelevant' (the source neither supports nor "
    "refutes it). Judge only from the source text; do not use outside knowledge."
)


def make_llm_entailment(complete: Callable[[str, str], str]) -> Callable[[str, str], str]:
    """Wrap a ``complete(system, user) -> str`` LLM callable (e.g. from
    ``agent.deepseek_llm.make_complete`` / ``agent.llmhub_llm.make_complete``) into the
    ``entailment_fn(claim_text, source_text) -> "entails"|"contradicts"|"irrelevant"``
    contract that ``make_entailment_verify`` consumes. This is the REAL semantic
    verifier behind the verify seam (the ``lexical_entailment`` placeholder's upgrade).
    Fails closed to ``"irrelevant"`` on any unparseable reply (the verdict then drives
    an unsupported -> non-positive reward, never a false "supported")."""
    def entailment_fn(claim_text: str, source_text: str) -> str:
        user = f"CLAIM: {claim_text}\nSOURCE: {source_text}\nOne word:"
        try:
            out = (complete(_ENTAILMENT_SYS, user) or "").strip().lower()
        except Exception:  # noqa: BLE001 — network/parse failure must not crash the rollout
            return "irrelevant"
        if "contradict" in out:
            return "contradicts"
        if "entail" in out:
            return "entails"
        return "irrelevant"

    return entailment_fn


def entailment_from_provider(
    provider: str,
    *,
    model: str | None = None,
    api_key_file: str | None = None,
) -> Callable[[str, str], str]:
    """Build a live entailment_fn from a provider name. ``provider`` is ``"deepseek"``
    or ``"llmhub"`` (the repo's two OpenAI-compatible clients). Keys come from the
    client's env var / ``api_key_file`` (default ``private/secrets/<provider>_api_key``,
    which is gitignored) — never hardcoded."""
    key_file = api_key_file or f"private/secrets/{provider}_api_key"
    if provider == "deepseek":
        from agent import deepseek_llm
        complete = deepseek_llm.make_complete(
            model=model or deepseek_llm.DEFAULT_MODEL, api_key_file=key_file, max_tokens=5)
    elif provider == "llmhub":
        from agent import llmhub_llm
        complete = llmhub_llm.make_complete(
            model=model or "gemini-2.5-flash-lite", api_key_file=key_file, max_tokens=5)
    else:
        raise ValueError(f"unknown entailment provider {provider!r} (use 'deepseek' or 'llmhub')")
    return make_llm_entailment(complete)


def conformance_check() -> tuple[bool, dict]:
    """Assert every live adapter conforms to the rollout interface. The retrieve
    adapter is exercised against the REAL committed RAG index (offline); the rollout
    is driven with a stub policy + mock entailment so no model is needed."""
    from provenance_bench import faithfulness_rollout as fr
    from provenance_bench.retrieval_faithfulness import (
        REWARD_MAX,
        REWARD_MIN,
        reward_for_trajectory,
    )

    live_retrieve = make_ai_search_retrieve(top_k=5)
    live_verify = make_entailment_verify(_mock_entailment)

    # 1. retrieve adapter returns well-formed chunks against the real index.
    chunks = live_retrieve("Who wrote the Dao De Jing?")
    retrieve_shape_ok = all(
        isinstance(c, dict) and {"chunk_id", "text", "author_confidence"} <= set(c)
        and isinstance(c["chunk_id"], str)
        for c in chunks
    )

    # 2. extract adapter returns rollout-shaped claims.
    sample_claims = heuristic_extract_claims(
        "The Dao De Jing was written by Laozi according to tradition.", chunks)
    extract_shape_ok = all(
        {"text", "key", "kind", "support_chunk_ids"} <= set(cl) for cl in sample_claims
    ) if sample_claims else True

    # 3. verify adapter returns a valid verdict.
    verdict = live_verify({"text": "Laozi wrote it", "support_chunk_ids": []},
                          [{"chunk_id": "x", "text": "Laozi wrote the Dao De Jing."}])
    verify_ok = verdict in ("supported", "unsupported", "contradicted")

    # 4. full rollout through the LIVE retrieve + verify + extract seams (stub policy).
    #    Falls back to the mock retriever only if the index is empty in this env, so the
    #    interface end-to-end is always checked.
    retrieve_for_e2e = live_retrieve if chunks else fr._mock_retrieve
    case = {"prompt": "Who wrote the Project Phoenix Charter?",
            "should_retrieve": True, "answerable": True}
    traj = fr.rollout(
        case,
        retrieve=retrieve_for_e2e,
        generate=lambda q, ctx: "The Project Phoenix Charter was written by the founding committee.",
        extract_claims=heuristic_extract_claims,
        verify_claim=live_verify,
    )
    r, _ = reward_for_trajectory(traj)
    e2e_ok = (REWARD_MIN <= r <= REWARD_MAX) and isinstance(traj.get("claims"), list)

    checks = {
        "retrieveShape": retrieve_shape_ok,
        "extractShape": extract_shape_ok,
        "verifyVerdict": verify_ok,
        "rolloutEndToEnd": e2e_ok,
    }
    detail = {
        "checks": checks,
        "liveIndexChunks": len(chunks),
        "usedLiveRetrieveForE2E": bool(chunks),
        "sampleVerdict": verdict,
        "e2eReward": round(r, 4),
    }
    return all(checks.values()), detail


__all__ = [
    "make_ai_search_retrieve",
    "make_entailment_verify",
    "heuristic_extract_claims",
    "lexical_entailment",
    "make_llm_entailment",
    "entailment_from_provider",
    "conformance_check",
]
