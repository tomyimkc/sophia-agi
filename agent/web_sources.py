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


def wikipedia_article(topic: str, *, max_chars: int = 8000, timeout: float = 20.0) -> "str | None":
    """Fetch the FULL Wikipedia article text for ``topic`` (HTML endpoint, tags stripped).

    The REST summary (``wikipedia_summary``) is a 1-2 sentence lead that is silent on many
    entities the article body discusses — e.g. the Voynich summary doesn't mention the
    candidate author "Anthony Ascham", but the full article does ("...written by the
    16th-century English author Anthony Ascham..."). That gap is the SILENT-REFERENCE
    boundary: a fabricated claim about an entity the summary omits returns 'irrelevant'.
    Fetching the full article closes it — the verifier can now place the claim and return
    'contradicts'.

    Returns the article's LEAD (consensus statement) capped at ``max_chars``. The lead is
    where Wikipedia states the consensus ("authorship debated / unknown"), which is the
    load-bearing content for catching fabricated-author claims. For candidate-specific
    context buried deep in the body (e.g. Ascham at 51% of the Voynich article), use
    ``wikipedia_article_for_claims`` which extracts windows around named entities.

    Args:
        topic: Wikipedia page title.
        max_chars: cap on returned text.
        timeout: network timeout.

    Returns:
        The stripped article text (capped), or None on any failure. Caller fails-closed on None.
    """
    import html as _html  # noqa: PLC0415
    slug = urllib.parse.quote(re.sub(r"\s+", "_", topic.strip()))
    url = f"https://en.wikipedia.org/api/rest_v1/page/html/{slug}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status != 200:
                return None
            raw = r.read().decode("utf-8", "ignore")
    except Exception:  # noqa: BLE001
        return None
    text = _html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", raw))).strip()
    if not text:
        return None
    return text[:max_chars]


# Capitalized multi-word names (Candidate Author, John Dee) — the entities whose mentions
# in the article body we want to surface for the contamination defense.
_NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")


def _candidate_names(answer: str) -> "list[str]":
    """Pull capitalized multi-word names out of the answer — the entities whose article
    mentions we want to surface (e.g. 'Anthony Ascham', 'Roger Bacon')."""
    seen, names = set(), []
    for m in _NAME_RE.findall(answer or ""):
        ml = m.lower()
        if ml not in seen and ml not in ("the voynich",):  # filter obvious non-entities
            seen.add(ml)
            names.append(m)
    return names


def wikipedia_article_for_claims(topic: str, answer: str, *,
                                 max_chars: int = 8000, window: int = 400,
                                 timeout: float = 20.0) -> "str | None":
    """Fetch the full article and return a LEAD + claim-guided windows extract.

    This closes the SILENT-REFERENCE boundary: a candidate author (e.g. Anthony Ascham) may
    be discussed deep in the article body (51% through the Voynich article) — far past a
    head-only cap. This function finds the answer's capitalized names in the full article
    and returns the lead PLUS a window around each name's first mention, so the verifier
    sees the relevant context wherever it sits.

    Args:
        topic: Wikipedia page title.
        answer: the answer being verified — its capitalized names guide the extraction.
        max_chars: total cap on returned text.
        window: chars of context around each name mention.
        timeout: network timeout.

    Returns:
        Lead + claim-guided windows (capped), or None on fetch failure.
    """
    import html as _html  # noqa: PLC0415
    slug = urllib.parse.quote(re.sub(r"\s+", "_", topic.strip()))
    url = f"https://en.wikipedia.org/api/rest_v1/page/html/{slug}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status != 200:
                return None
            raw = r.read().decode("utf-8", "ignore")
    except Exception:  # noqa: BLE001
        return None
    text = _html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", raw))).strip()
    if not text:
        return None
    low = text.lower()
    parts = [text[: min(max_chars // 2, len(text))]]  # lead (consensus statement)
    budget = max_chars - len(parts[0])
    for name in _candidate_names(answer):
        i = low.find(name.lower())
        if i < 0:
            continue  # name not in article -> stays silent (the verifier returns 'irrelevant')
        seg = text[max(0, i - window): i + len(name) + window]
        if budget - len(seg) - 5 < 0:
            break
        parts.append("... " + seg + " ...")
        budget -= len(seg) + 5
    return " ".join(parts)[:max_chars] if parts else text[:max_chars]


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
            catch CONTRADICTIONS (an injected fabrication). A single live source rarely fully
            corroborates an answer (``fact_check_gate`` needs >=2 distinct domains for
            ``accepted``), so a clean answer that returns ``held`` (e.g. "authorship is unknown"
            against Wikipedia — consistent but not multi-domain-corroborated) would be
            over-blocked under strict mode. With QUESTION-AWARE entailment, contaminated answers
            (e.g. bare "Anthony Ascham.") return ``contradicts`` and are blocked regardless; held
            is the right outcome to pass. Set True only with >=2 independent-domain backends.

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
            # Prefer the FULL article with CLAIM-GUIDED extraction: a candidate author may be
            # discussed deep in the body (51% through the Voynich article), far past a head
            # cap. The claim-guided extract surfaces the lead + windows around the answer's
            # named entities, closing the silent-reference boundary. Fall back to summary.
            s = wikipedia_article_for_claims(t, answer) or wikipedia_summary(t)
            if s:
                refs.append(s)
        if len(refs) < min_refs:
            return False  # could not fetch independent reference -> fail closed
        # block_on_hold -> accept_on_hold (the contamination defense catches contradictions;
        # held = unverified-but-not-contradicted is acceptable unless full corroboration is set)
        return make_independent_verifier(refs, entailment_fn, accept_on_hold=not block_on_hold)(question, answer)

    return verify
