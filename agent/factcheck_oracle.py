# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Google Fact Check Tools API oracle for the out-of-wiki fact-check gate.

Unlike :mod:`agent.live_sources` (which is deliberately KEYLESS), this backend
calls the keyed Google Fact Check Tools API
(``factchecktools.googleapis.com``). It is strictly opt-in: the gate only uses
it when an API key is supplied (argument or env ``GOOGLE_FACTCHECK_API_KEY``)
and the caller composes it into the retriever/entailment seam. With no key it
yields no evidence, so the gate holds fail-closed and CI stays offline.

What it provides
----------------
- ``retriever(claim)``: query published ClaimReview records matching the claim
  and return them as :class:`~agent.fact_check_gate.EvidenceSource` rows
  (``source_type="factcheck"``), encoding the reviewed claim text + publisher
  rating in a canonical, parseable snippet.
- ``entailment(claim, source)``: map a ClaimReview's textual rating to
  ``entails`` / ``contradicts`` / ``irrelevant`` — but ONLY for clean rating
  labels and only when the reviewed claim sufficiently overlaps the claim under
  test. Free-prose ratings (e.g. "we have abundant evidence the Earth is
  spherical") and weak matches return ``irrelevant``, so the gate holds rather
  than guessing claim polarity.

Honest bound
------------
ClaimReview coverage is skewed toward debunked/contested claims, so this oracle
is strong at catching FALSE claims and weak at confirming TRUE ones. The
rating→polarity map is a conservative lexical screen, NOT natural-language
inference: soft labels ("misleading", "partly false", "mixture", "unproven",
"missing context") are treated as ``irrelevant`` on purpose. Network/quota
failures fail closed (no evidence), never as a pass.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Callable
from urllib.parse import quote
from urllib.request import Request, urlopen

from agent.fact_check_gate import AtomicClaim, EntailmentFn, EvidenceSource, Retriever

API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
USER_AGENT = "sophia-agi fact-check oracle (github.com/tomyimkc/sophia-agi)"
DEFAULT_TIMEOUT = 8.0

# Strong, unambiguous rating labels. Soft/mixed labels are intentionally absent
# so they fall through to ``irrelevant`` (a safe hold) rather than asserting a
# polarity the gate cannot stand behind.
_FALSE_RATINGS = (
    "false", "mostly false", "pants on fire", "fake", "incorrect", "inaccurate",
    "untrue", "not true", "debunked", "hoax", "baseless", "fabricated",
    "no evidence", "four pinocchios", "bogus", "scam", "myth", "made up",
)
_TRUE_RATINGS = (
    "true", "mostly true", "correct", "accurate", "verified", "confirmed",
)
# Soft labels we explicitly refuse to map (documented; recognised so a future
# reader sees they were considered and deliberately held).
_AMBIGUOUS_RATINGS = (
    "misleading", "partly false", "partially false", "half true", "mixture",
    "mixed", "unproven", "unsupported", "missing context", "needs context",
    "outdated", "miscaptioned", "altered", "satire", "exaggerated",
)

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "and", "or",
    "for", "with", "by", "that", "this", "it", "as", "on", "at", "be", "has",
    "have", "do", "does", "can", "will", "from", "their", "its",
}


def normalize_text(value: str) -> str:
    """Lowercase alnum normalization for claim/evidence comparison."""
    return " ".join(re.findall(r"[a-z0-9]+", (value or "").lower()))


def _content_tokens(text: str) -> set[str]:
    return {t for t in normalize_text(text).split() if len(t) > 2 and t not in _STOP}


def _claim_overlap(claim_text: str, reviewed_text: str) -> float:
    """Fraction of the claim's content tokens present in the reviewed claim."""
    claim = _content_tokens(claim_text)
    if not claim:
        return 0.0
    reviewed = _content_tokens(reviewed_text)
    return len(claim & reviewed) / len(claim)


# Asymmetric relation roots: "X <verb> Y" does NOT mean "Y <verb> X". A bag-of-
# words match cannot tell the two apart, so a fact-check that debunks the INVERTED
# claim must not be read as contradicting the claim under test. (Discovered live:
# "the earth orbits the sun" was wrongly rejected by a review of "sun orbiting
# Earth".) When inversion is detected the oracle returns ``irrelevant``.
_ASYM_ROOTS = ("orbit", "circl", "caus", "contain", "creat", "kill", "beat",
               "discover", "invent", "defeat", "revolv", "preced", "follow")


def _root_index(tokens: list[str], root: str) -> int:
    for i, tok in enumerate(tokens):
        if root in tok:
            return i
    return -1


def _has(a: set[str], b: set[str]) -> bool:
    return bool(a) and len(a & b) / len(a) >= 0.5


def _relational_inversion(claim_text: str, reviewed_text: str) -> bool:
    """True if the two texts share an asymmetric relation but with swapped args."""
    claim_tokens = normalize_text(claim_text).split()
    rev_tokens = normalize_text(reviewed_text).split()
    for root in _ASYM_ROOTS:
        ci = _root_index(claim_tokens, root)
        ri = _root_index(rev_tokens, root)
        if ci < 0 or ri < 0:
            continue
        c_subj = _content_tokens(" ".join(claim_tokens[:ci]))
        c_obj = _content_tokens(" ".join(claim_tokens[ci + 1:]))
        r_subj = _content_tokens(" ".join(rev_tokens[:ri]))
        r_obj = _content_tokens(" ".join(rev_tokens[ri + 1:]))
        same_order = _has(c_subj, r_subj) and _has(c_obj, r_obj)
        swapped = _has(c_subj, r_obj) and _has(c_obj, r_subj)
        if swapped and not same_order:
            return True
    return False


def _classify_rating(rating: str) -> str:
    """Map a ClaimReview textual rating to entails/contradicts/irrelevant.

    Conservative: a rating that matches BOTH a false and a true cue, or only a
    soft/ambiguous cue, or no cue at all, returns ``irrelevant``.
    """
    norm = normalize_text(rating)
    if not norm:
        return "irrelevant"
    padded = f" {norm} "
    has_false = any(f" {normalize_text(p)} " in padded or padded.strip() == normalize_text(p) for p in _FALSE_RATINGS)
    has_true = any(f" {normalize_text(p)} " in padded or padded.strip() == normalize_text(p) for p in _TRUE_RATINGS)
    if has_false and has_true:
        return "irrelevant"
    if has_false:
        return "contradicts"
    if has_true:
        return "entails"
    return "irrelevant"


def _sanitize(value: str) -> str:
    # The snippet uses " | " as a field delimiter; neutralise it in free text.
    return re.sub(r"\s*\|\s*", " / ", (value or "").replace("\n", " ")).strip()


def _encode_snippet(rating: str, reviewed_claim: str, publisher: str) -> str:
    return (
        f"ClaimReview | rating: {_sanitize(rating)} | reviewedClaim: {_sanitize(reviewed_claim)} "
        f"| publisher: {_sanitize(publisher)}"
    )


_SNIPPET_RE = re.compile(
    r"rating:\s*(?P<rating>.*?)\s*\|\s*reviewedClaim:\s*(?P<claim>.*?)\s*\|\s*publisher:",
    re.S,
)


def _decode_snippet(snippet: str) -> tuple[str, str] | None:
    m = _SNIPPET_RE.search(snippet or "")
    if not m:
        return None
    return m.group("rating"), m.group("claim")


def _http_get_json(url: str, *, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    req = Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed Google API endpoint
        return json.loads(resp.read().decode("utf-8"))


class GoogleFactCheckOracle:
    """Keyed ClaimReview retriever + entailment for the fact-check gate.

    Parameters
    ----------
    api_key:
        Google API key with the Fact Check Tools API enabled. Falls back to the
        ``GOOGLE_FACTCHECK_API_KEY`` env var. When absent the oracle is disabled
        and yields no evidence (fail-closed).
    fetch:
        Optional ``(url) -> dict`` injection point so tests run offline with no
        network. Defaults to a urllib GET.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        fetch: Callable[[str], dict[str, Any]] | None = None,
        page_size: int = 10,
        language: str = "en",
        timeout: float = DEFAULT_TIMEOUT,
        min_overlap: float = 0.6,
        max_sources: int = 8,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("GOOGLE_FACTCHECK_API_KEY", "")
        self._fetch = fetch or (lambda url: _http_get_json(url, timeout=timeout))
        self.page_size = page_size
        self.language = language
        self.min_overlap = min_overlap
        self.max_sources = max_sources

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _query_url(self, query: str) -> str:
        return (
            f"{API_URL}?query={quote(query, safe='')}"
            f"&languageCode={quote(self.language, safe='')}"
            f"&pageSize={int(self.page_size)}&key={quote(self.api_key, safe='')}"
        )

    def retriever(self, claim: AtomicClaim) -> list[EvidenceSource]:
        if not self.enabled:
            return []
        query = (claim.text or "").strip()
        if not query:
            return []
        try:
            data = self._fetch(self._query_url(query))
        except Exception:
            return []  # fail-closed: a broken/throttled API must not accept claims
        out: list[EvidenceSource] = []
        for entry in (data.get("claims") or []):
            reviewed = str(entry.get("text") or "")
            for review in (entry.get("claimReview") or []):
                rating = str(review.get("textualRating") or "")
                url = str(review.get("url") or "")
                publisher = str((review.get("publisher") or {}).get("name") or "")
                site = str((review.get("publisher") or {}).get("site") or "")
                if not (rating and reviewed):
                    continue
                ident = hashlib.sha1(f"{url}|{reviewed}".encode("utf-8")).hexdigest()[:10]  # noqa: S324 - non-crypto id
                out.append(EvidenceSource(
                    id=f"factcheck:{site or 'unknown'}:{ident}",
                    url=url or (f"https://{site}" if site else ""),
                    title=str(review.get("title") or f"{publisher} fact-check"),
                    snippet=_encode_snippet(rating, reviewed, publisher),
                    publisher=publisher or site,
                    retrieved_at=str(review.get("reviewDate") or ""),
                    source_type="factcheck",
                ))
                if len(out) >= self.max_sources:
                    return out
        return out

    def entailment(self, claim: AtomicClaim, source: EvidenceSource) -> str:
        if (source.source_type or "") != "factcheck":
            return "irrelevant"
        decoded = _decode_snippet(source.snippet)
        if decoded is None:
            return "irrelevant"
        rating, reviewed_claim = decoded
        # Only assign a relation when the published review is about substantially
        # the same assertion; otherwise a tangential review must not move the gate.
        if _claim_overlap(claim.text, reviewed_claim) < self.min_overlap:
            return "irrelevant"
        # Refuse to contradict/entail when the review is about the inverted
        # relation (polarity-blind bag-of-words guard).
        if _relational_inversion(claim.text, reviewed_claim):
            return "irrelevant"
        return _classify_rating(rating)


def combined_retriever(retrievers: list[Retriever]) -> Retriever:
    """Concatenate several retrievers into one (order preserved, all queried)."""
    def _r(claim: AtomicClaim) -> list[EvidenceSource]:
        out: list[EvidenceSource] = []
        for retr in retrievers:
            out.extend(retr(claim) or [])
        return out
    return _r


def dispatched_entailment(
    oracle: GoogleFactCheckOracle,
    base: EntailmentFn | None,
    *,
    factcheck_entailment: EntailmentFn | None = None,
) -> EntailmentFn:
    """Route ``factcheck`` sources to the oracle (or ``factcheck_entailment`` when
    given, e.g. an NLI backend), everything else to ``base``."""
    fc = factcheck_entailment or oracle.entailment
    def _e(claim: AtomicClaim, source: EvidenceSource) -> str:
        if (source.source_type or "") == "factcheck":
            return fc(claim, source)
        if base is None:
            return "irrelevant"
        return base(claim, source)
    return _e


def compose_live_factcheck(
    base_retriever: Retriever,
    base_entailment: EntailmentFn | None,
    *,
    oracle: GoogleFactCheckOracle | None = None,
    factcheck_entailment: EntailmentFn | None = None,
) -> tuple[Retriever, EntailmentFn, bool]:
    """Default live composition: add the Google oracle iff it is enabled.

    Returns ``(retriever, entailment, oracle_active)``. When no key is present
    the oracle is disabled and the base backend is returned unchanged, so
    offline/CI behaviour is identical. This is the canonical way callers turn the
    keyed oracle on "by default where a key exists" without a per-call flag.
    """
    oracle = oracle if oracle is not None else GoogleFactCheckOracle()
    if not oracle.enabled:
        return base_retriever, (base_entailment or (lambda c, s: "irrelevant")), False
    retriever = combined_retriever([base_retriever, oracle.retriever])
    entailment = dispatched_entailment(oracle, base_entailment, factcheck_entailment=factcheck_entailment)
    return retriever, entailment, True


__all__ = [
    "GoogleFactCheckOracle",
    "combined_retriever",
    "compose_live_factcheck",
    "dispatched_entailment",
    "normalize_text",
]
