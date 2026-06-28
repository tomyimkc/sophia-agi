# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Attribution-swap verifier — catches a REAL work credited to the WRONG creator.

The citation-existence verifier confirms a cited study *exists*; it cannot catch a swap of a
REAL work's attribution (Hamlet -> Marlowe, the Mona Lisa -> Raphael, penicillin -> Pasteur).
That is an *attribution-faithfulness* question (the repo's Ayinde / legal-holding-faithful
failure mode). This module answers it with HIGH independence: resolve the work's entity in
Wikidata, read its true creator/author/discoverer, and flag a credited person who is NOT among
them.

Trustworthy, fail-OPEN-on-ignorance: a swap is asserted only when Wikidata returns an
authoritative creator AND the credited person is not among them (by surname). If the work
cannot be resolved or has no creator record, the verdict is ``unknown`` and the gate does not
claim a swap — it never *fabricates* a contradiction. Independence is HIGH (a structured
external record, no model judgment). Honest bound: entity disambiguation is imperfect (an
ambiguous title may resolve to the wrong entity -> ``unknown``), and a genuinely co-credited
person (e.g. Franklin for DNA) is correctly NOT a swap.
"""
from __future__ import annotations

import re
from typing import Any, Callable

__all__ = [
    "extract_attributions", "true_attributees", "verify_attribution",
    "make_attribution_corroborate_fn",
]

# Creator/author/discoverer verbs that credit a person with a work.
_ATTR_VERB = (r"(?:painted|wrote|authored|composed|created|designed|built|sculpted|directed|"
              r"discovered|invented|developed|formulated|proposed|devised|founded|programmed)")
_PERSON = r"[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,3}"
# "... by <Person>"  and  "<Person> <verb> ..."
_BY_RE = re.compile(rf"\bby\s+(?P<who>{_PERSON})")
_VERB_RE = re.compile(rf"\b(?P<who>{_PERSON})\s+{_ATTR_VERB}\b")
_ATTRIB_TO_RE = re.compile(rf"\battribut\w+\s+(?:the\s+)?.*?\bto\s+(?P<who>{_PERSON})", re.IGNORECASE)
_STOPNAMES = {"The", "A", "An", "This", "That", "It", "Based", "According", "Source", "However",
              "Although", "While", "Both", "He", "She", "They", "Dr", "Professor"}


def _name_matches(credited: str, true_label: str) -> bool:
    """True iff ``credited`` plausibly names the same person as ``true_label``.

    Robust to lowercase particles and partial names: substring either way ("Leonardo" vs
    "Leonardo da Vinci"), or a shared significant (>=4-char) name token. A wrong-but-famous
    swap (Raphael vs Leonardo da Vinci) shares neither and is flagged."""
    c = credited.lower().strip()
    t = true_label.lower().strip()
    if not c or not t:
        return False
    if c in t or t in c:
        return True
    ct = {w for w in re.split(r"[^a-z]+", c) if len(w) >= 4}
    tt = {w for w in re.split(r"[^a-z]+", t) if len(w) >= 4}
    return bool(ct & tt)


def extract_attributions(question: str, answer: str) -> "list[tuple[str, str]]":
    """Return ``[(work, credited_person)]`` claimed by the answer.

    The ``work`` is the question's subject (best-effort: the noun phrase after a leading
    who/what-developed stem, else the question minus the wh-word); the credited persons are the
    proper-name strings the answer attaches via "by X" / "X <verb>" / "attributed to X".
    """
    work = _question_subject(question)
    persons: "list[str]" = []
    for rx in (_BY_RE, _VERB_RE, _ATTRIB_TO_RE):
        for m in rx.finditer(answer or ""):
            who = re.sub(r"[.,;:'\"\s]+$", "", m.group("who").strip())
            if not who or who.split()[0] in _STOPNAMES:
                continue
            if who not in persons:
                persons.append(who)
    return [(work, p) for p in persons] if work else []


def _question_subject(question: str) -> str:
    q = (question or "").strip().rstrip("?").strip()
    # Drop a leading "who/what (first|originally) <verb> the" stem to expose the work.
    m = re.search(rf"\b(?:who|what)\b.*?\b{_ATTR_VERB}\s+(?:the\s+)?(?P<work>.+)$", q, re.IGNORECASE)
    if m:
        return m.group("work").strip()
    # Fallback: a quoted or Capitalized work phrase.
    m = re.search(r"[\"“](?P<w>[^\"”]+)[\"”]", q)
    if m:
        return m.group("w").strip()
    return q


def true_attributees(work: str, *, wikidata_lookup: "Callable[[str], list[str]]") -> "list[str]":
    """Authoritative creator/author/discoverer labels for ``work`` (via injected lookup)."""
    try:
        return list(wikidata_lookup(work) or [])
    except Exception:  # noqa: BLE001 — fail-closed: no record, not a guessed swap
        return []


def verify_attribution(work: str, credited: str, *,
                       wikidata_lookup: "Callable[[str], list[str]]") -> "dict[str, Any]":
    """``{"verdict": "swapped"|"correct"|"unknown", "true": [...], "credited": credited}``.

    ``swapped`` only when authoritative attributees exist AND ``credited``'s surname is not among
    them. ``unknown`` when no authoritative record (never a fabricated contradiction)."""
    true = true_attributees(work, wikidata_lookup=wikidata_lookup)
    if not true:
        return {"verdict": "unknown", "true": [], "credited": credited}
    if any(_name_matches(credited, t) for t in true):
        return {"verdict": "correct", "true": true, "credited": credited}
    return {"verdict": "swapped", "true": true, "credited": credited}


def make_attribution_corroborate_fn(
    *, wikidata_lookup: "Callable[[str], list[str]]",
) -> "Callable[[str, str], bool]":
    """Build a ``(question, answer) -> bool`` corroborate_fn. Returns False (REJECT -> swap
    caught) iff the answer credits a work to a person who is NOT its authoritative creator.
    Exposes ``.last_result``."""
    holder: "dict[str, Any]" = {}

    def verify(question: str, answer: str) -> bool:
        if not answer or not answer.strip():
            return True
        swaps = []
        for work, person in extract_attributions(question, answer):
            v = verify_attribution(work, person, wikidata_lookup=wikidata_lookup)
            if v["verdict"] == "swapped":
                swaps.append(v)
        holder.clear()
        holder.update({"swaps": swaps})
        return not swaps

    verify.last_result = holder  # type: ignore[attr-defined]
    return verify
