"""Atomic-claim decomposition + claim-type routing (FActScore / SAFE-style).

The fixed panel in :func:`agent.gate.check_response` runs every verifier over the
WHOLE answer. That gates a text if it contains *any* false arithmetic, *any*
forbidden attribution, etc. — but it cannot say *which* sentence failed, and it
cannot route a checkable predicate to the one verifier that can judge it.

This module decomposes an answer into atomic claims, classifies each claim by the
*kind* of checkable predicate it asserts, and runs only the MATCHING registered
verifier per claim. The result is per-claim attribution: the loop gates ANY
checkable predicate, not just whole-text scans, and reports exactly which claim
tripped which verifier.

Reuses the existing verifiers in :mod:`agent.verifiers` verbatim — it never
reimplements a check. ``'other'`` claims (no checkable predicate) pass: this is a
soundness gate, not a presence requirement. Pure stdlib, deterministic, offline.
"""

from __future__ import annotations

import re
from typing import Any

from agent import verifiers as _v

# --------------------------------------------------------------------------- #
# Atomic-claim decomposition.
# --------------------------------------------------------------------------- #

# Sentence boundary: terminal punctuation (Latin + CJK) or a newline. We keep the
# split conservative so we do not shatter abbreviations into fragments below the
# verifiers' own min-length guards.
_SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|\n+")

# Light clause split for compound sentences: coordinating connectors that join two
# independent assertions ("Plato wrote the Republic and 2 + 2 = 5"). Kept narrow
# (";", " and ", " but ", " however ", em/en dashes) so an appositive comma is NOT
# treated as a clause boundary — that would break author->title patterns the
# provenance verifier relies on.
_CLAUSE_SPLIT = re.compile(
    r"\s*;\s*|\s+\band\b\s+|\s+\bbut\b\s+|\s+\bhowever\b\s+|\s+\byet\b\s+|\s*—\s*|\s*–\s*"
)


def split_claims(text: str) -> list[str]:
    """Split an answer into atomic claim strings.

    Sentence split (terminal punctuation / newlines), then a light clause split on
    coordinating connectors — BUT only for sentences that are NOT authorship/legal
    assertions. Authorship and legal sentences are kept whole: clause-splitting on
    " and " would shatter multi-word titles ("Beyond Good and Evil" -> "Beyond
    Good") and sever a corrective carve-out ("it is a myth, and X wrote Y") from its
    assertion — both produce wrong verdicts. The provenance/legal verifiers do their
    own sentence-internal clause handling, so they need the full sentence. Clause
    splitting is reserved for arithmetic-style independent assertions. Whitespace-
    normalized; empty fragments dropped. Stdlib only, deterministic.
    """
    claims: list[str] = []
    for sentence in _SENT_SPLIT.split(text or ""):
        sentence = re.sub(r"\s+", " ", sentence).strip()
        if not sentence:
            continue
        # Keep authorship/legal sentences whole (carve-outs + multi-word titles).
        if _AUTHORSHIP_RE.search(sentence) or _has_legal(sentence):
            claims.append(sentence.strip(" \t\r\n,.!?。！？"))
            continue
        for clause in _CLAUSE_SPLIT.split(sentence):
            clause = re.sub(r"\s+", " ", clause).strip(" \t\r\n,.!?。！？")
            if clause:
                claims.append(clause)
    return claims


# --------------------------------------------------------------------------- #
# Claim-type classification.
# --------------------------------------------------------------------------- #

# Arithmetic: a binary equality "a OP b = c". Reuse the verifier's own pattern so
# classification and checking agree on what counts as arithmetic.
_ARITH_RE = _v._ARITH

# Authorship: an assertion of who produced a work.
_AUTHORSHIP_RE = re.compile(
    r"\b(?:wrote|writes?|writing|written|authored?|author of|penned|composed|attribut\w*|credited with)",
    re.IGNORECASE,
)

# Legal: a neutral citation / ordinance ref, a "X v. Y" case style, or a holding
# verb. We borrow the legal extractor for the citation half and add the prose cues.
# The holding cues are case-insensitive; the "X v. Y" case style stays
# case-sensitive (it relies on capitalized party names).
_LEGAL_HOLDING_RE = re.compile(r"\b(?:the court (?:held|found|ruled)|it was held)\b", re.IGNORECASE)
_LEGAL_CASE_RE = re.compile(r"[A-Z][a-z]+ v\.? [A-Z]")

# Citation: a bracketed source marker ([1] / [local 1] / [web 1] / [source 1]) or
# the literal word "source".
_CITATION_RE = re.compile(r"\[(?:local|web|source)?\s*\d+\]|\bsource\b", re.IGNORECASE)


def _has_legal(claim: str) -> bool:
    if _LEGAL_HOLDING_RE.search(claim) or _LEGAL_CASE_RE.search(claim):
        return True
    try:
        from agent.legal_citations import extract_citations

        return bool(extract_citations(claim))
    except Exception:  # noqa: BLE001 - legal extraction is best-effort for routing
        return False


def classify_claim(claim: str) -> str:
    """Return the claim's checkable type.

    One of ``'authorship'``, ``'citation'``, ``'arithmetic'``, ``'legal'``, or
    ``'other'`` (no machine-checkable predicate). Heuristic, keyword/regex based.

    Ordering matters: arithmetic is the most syntactically specific, then the legal
    cues (a citation/holding), then authorship, then a bare source marker. A claim
    that matches nothing is ``'other'`` and will pass routing (not checkable here).
    """
    if _ARITH_RE.search(claim):
        return "arithmetic"
    if _has_legal(claim):
        return "legal"
    if _AUTHORSHIP_RE.search(claim):
        return "authorship"
    if _CITATION_RE.search(claim):
        return "citation"
    return "other"


# --------------------------------------------------------------------------- #
# Route + check.
# --------------------------------------------------------------------------- #


def route_and_check(text: str, *, records: "dict | None" = None,
                    sources: "list[str] | None" = None,
                    legal_resolver=None) -> dict:
    """Split, classify, and run the MATCHING verifier per atomic claim.

    Routing (each runs the existing verifier from :mod:`agent.verifiers`):

    - ``arithmetic``  -> :func:`verifiers.arithmetic_sound`
    - ``authorship``  -> :func:`verifiers.provenance_faithful` (``records``)
    - ``legal``       -> :func:`verifiers.legal_citation_exists`
    - ``citation``    -> :func:`verifiers.citation_faithful` (``sources``) — skipped
      (claim passes) when no ``sources`` are supplied, since there is nothing to
      check faithfulness against.
    - ``other``       -> passes (no checkable predicate).

    Returns ``{passed, perClaim:[{claim,type,passed,reasons}], violations:[...]}``.
    ``passed`` is True iff every claim passed. ``violations`` aggregates each failed
    claim's reasons, prefixed with the claim type for attribution.
    """
    # Build verifiers once. arithmetic/provenance/legal are parameterless-ish and
    # cheap to construct; citation depends on sources and is built only if needed.
    arith = _v.arithmetic_sound()
    prov = _v.provenance_faithful(records)
    # Pass the operator's resolver so a real citation absent from the bundled static
    # register gets its second-chance lookup (mirrors agent.gate._legal_gate). Without
    # this the routed legal check is fail-closed and flags valid citations as forged.
    legal = _v.legal_citation_exists(resolver=legal_resolver)
    cite = _v.citation_faithful(sources) if sources else None

    per_claim: list[dict] = []
    violations: list[str] = []

    for claim in split_claims(text):
        ctype = classify_claim(claim)
        if ctype == "arithmetic":
            result = arith(claim, None, {})
        elif ctype == "authorship":
            result = prov(claim, None, {})
        elif ctype == "legal":
            result = legal(claim, None, {})
        elif ctype == "citation":
            if cite is None:
                result = {"passed": True, "reasons": [], "detail": {"skipped": "no sources"}}
            else:
                result = cite(claim, None, {})
        else:  # 'other' — not machine-checkable here
            result = {"passed": True, "reasons": [], "detail": {}}

        passed = bool(result["passed"])
        reasons = list(result.get("reasons", []))
        per_claim.append({"claim": claim, "type": ctype, "passed": passed, "reasons": reasons})
        if not passed:
            violations.extend(f"[{ctype}] {r}" for r in reasons)

        # Arithmetic is self-contained (a OP b = c), so also check it on a claim
        # whose PRIMARY type isn't arithmetic — e.g. an authorship sentence kept
        # whole that also bundles a false equality ("X wrote Y and 2 + 2 = 5").
        # No-op when the claim has no equality.
        if ctype != "arithmetic" and _ARITH_RE.search(claim):
            ares = arith(claim, None, {})
            if not ares["passed"]:
                areasons = list(ares.get("reasons", []))
                per_claim.append({"claim": claim, "type": "arithmetic", "passed": False, "reasons": areasons})
                violations.extend(f"[arithmetic] {r}" for r in areasons)

    return {
        "passed": not violations,
        "perClaim": per_claim,
        "violations": violations,
    }
