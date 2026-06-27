# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Live external source retrieval for the independent verification channel.

Productionizes the "independent truth-references" in ``agent.source_verifier``: instead
of caller-curated text, this fetches real external references at verify-time. The first
backend is Wikipedia (REST summary API) — keyless, genuinely external (independent of any
grounding source in the repo), and authoritative for the attribution/unknown-author claims
the source-contamination defense targets.

Why Wikipedia for the contamination defense:
  - It is EXTERNAL to the repo's corpus and to any (possibly contaminated) grounding source
    a caller passes to ``grounded_answer_policy`` — independence is the load-bearing property.
  - Its attribution pages reliably state consensus ("authorship debated / unknown") rather
    than asserting a specific name, so a contaminated answer asserting a fabricated author
    CONTRADICTS the live reference.
  - Keyless (no API key) and low-latency, so the verifier can run in CI/eval without secrets.

Honest scope: Wikipedia is not ground truth — it can be wrong, lag, or be the target of
vandalism. For high-stakes claims, add a second independent backend (e.g. a curated
reference DB). This backend makes the verifier architecture work end-to-end on live data;
it does not make it infallible.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Callable

__all__ = ["wikipedia_summary", "make_wikipedia_verifier"]

UA = "sophia-agi-research/1.0 (verification channel; contact via repo)"


def wikipedia_summary(topic: str, *, timeout: float = 15.0) -> "str | None":
    """Fetch the Wikipedia REST summary extract for ``topic``. Returns None on any failure
    (network error, missing page, non-200) so callers can fail-closed (treat None as
    'no independent reference available' -> the verifier abstains rather than trusts)."""
    slug = urllib.parse.quote(re.sub(r"\s+", "_", topic.strip()))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status != 200:
                return None
            d = json.loads(r.read())
    except Exception:  # noqa: BLE001 — any failure -> None (fail-closed upstream)
        return None
    extract = d.get("extract", "")
    return extract if extract else None


def make_wikipedia_verifier(
    topic_resolver: "Callable[[str, str], str | None]",
    entailment_fn: "Callable[[str, str], str]",
    *,
    min_refs: int = 1,
    block_on_hold: bool = False,
) -> "Callable[[str, answer], bool]":
    """Build a ``(question, answer) -> bool`` verifier backed by LIVE Wikipedia.

    Args:
        topic_resolver: ``(question, answer) -> topic_string | None``. Maps a Q/A pair to the
            Wikipedia page title to fetch as the independent reference (e.g.
            "Who wrote the Voynich Manuscript?" -> "Voynich manuscript"). Returns None when no
            clear topic can be resolved -> the verifier abstains (fail-closed) rather than
            trusting the answer. This is the one caller-supplied piece (a production deploy
            would use an entity linker / the retriever; a test can hardcode it).
        entailment_fn: ``(claim, source) -> "entails"|"contradicts"|"irrelevant"`` (typically
            an LLM call). Threaded into ``source_verifier``.
        min_refs: minimum number of non-None references required to run the check; if fewer
            are fetched (network failure / no page), the verifier abstains (fail-closed).
        block_on_hold: how to treat a ``held`` verdict (claim unverified but not contradicted).
            Default False (lenient): held is ACCEPTED — the contamination defense's goal is to
            catch CONTRADICTIONS (an injected fabrication), and a single live source can rarely
            fully corroborate an answer (``fact_check_gate`` needs >=2 distinct domains for
            ``accepted``). Requiring full corroboration would over-block clean answers.
            Set True (strict) when you have >=2 independent-domain backends and want full
            corroboration before trusting.

    Returns:
        A verifier. True iff the answer is not contradicted by live Wikipedia (accepted OR
        held, when block_on_hold=False); False if contradicted, or if no reference could be
        fetched / no topic resolved (fail-closed — the gate abstains rather than trusts).

    Independence guarantee: the references come from Wikipedia, external to any grounding
    source the caller passed to ``grounded_answer_policy``. The verifier never sees that
    grounding source.
    """
    from agent.source_verifier import make_independent_verifier  # noqa: PLC0415

    def verify(question: str, answer: str) -> bool:
        if not answer or not answer.strip():
            return True  # nothing to verify
        topic = topic_resolver(question, answer)
        if not topic:
            return False  # no topic resolved -> fail closed (do not trust the answer)
        refs = []
        for t in ([topic] if isinstance(topic, str) else topic):
            s = wikipedia_summary(t)
            if s:
                refs.append(s)
        if len(refs) < min_refs:
            return False  # could not fetch independent reference -> fail closed
        # block_on_hold -> accept_on_hold (the contamination defense catches contradictions;
        # held = unverified-but-not-contradicted is acceptable unless full corroboration is set)
        return make_independent_verifier(refs, entailment_fn, accept_on_hold=not block_on_hold)(question, answer)

    return verify
