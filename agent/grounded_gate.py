# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""On-demand, retrieval-grounded record synthesis for the provenance gate.

The frozen ``doNotAttributeTo`` corpus (``provenance_bench`` misattributions +
the runtime ``data/*.json`` records) can only fire on works it already knows.
This module lets the gate fire on works that are NOT in the corpus by:

  1. resolving a work's *documented* author from an offline ground-truth source
     (the committed Wikidata snapshot, then optionally an OKF belief graph), and
  2. detecting an authorship assertion in free text whose work has no base
     record and whose claimed author DIFFERS from the resolved true author, and
     minting a one-off ``doNotAttributeTo`` record so
     :func:`agent.verifiers.provenance_faithful` can catch it.

Everything is deterministic and offline by default (no network, no model). It is
deliberately conservative: a record is synthesized only when a DIFFERENT true
author was resolved with confidence, keeping false positives near zero — a
correct attribution, an unknown work, or an ambiguous/multi-author gold all
yield nothing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Optional

# Reuse the benchmark's own title-form derivation so synthesized records match
# the natural phrasings the gate already expects. Read-only import.
from provenance_bench.dataset import _alt_titles

_SNAPSHOT_PATH = (
    Path(__file__).resolve().parent.parent
    / "provenance_bench"
    / "data"
    / "wikidata_snapshot.json"
)

# A "true author" string we will NOT use as a confident single author: anything
# naming multiple/uncertain contributors. Synthesizing against these risks false
# positives (the claimed author could be one of several legitimate contributors).
_AMBIGUOUS_GOLD = re.compile(
    r"\b(?:and|&|,|or|multiple|various|several|many|disciples?|compiled|"
    r"recording|attributed|unknown|anonymous|tradition)\b",
    re.IGNORECASE,
)


def _norm_name(name: str) -> str:
    """Lowercase + collapse whitespace + drop trailing parentheticals for a
    surname/identity comparison ('Confucius (compiled ...)' -> 'confucius')."""
    n = re.sub(r"\s*\(.*?\)\s*", " ", name or "")
    n = re.sub(r"[^\w\s'’-]", " ", n)
    return re.sub(r"\s+", " ", n).strip().lower()


def _norm_work(work: str) -> str:
    """Normalize a work title for snapshot lookup: lowercase, strip a leading
    'the', drop punctuation, collapse whitespace."""
    w = re.sub(r"^\s*the\s+", "", (work or "").strip(), flags=re.IGNORECASE)
    w = re.sub(r"[^\w\s'’-]", " ", w)
    return re.sub(r"\s+", " ", w).strip().lower()


def _load_snapshot(snapshot: "dict | None") -> dict:
    if snapshot is not None:
        return snapshot
    try:
        return json.loads(_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _snapshot_index(snapshot: dict) -> dict:
    """Map every work form (and its alt-title forms) -> gold_author, normalized."""
    index: dict[str, str] = {}
    for row in snapshot.get("attributions", []):
        work = row.get("work")
        gold = row.get("gold_author")
        if not work or not gold:
            continue
        keys = {_norm_work(work)}
        for alt in _alt_titles(work):
            keys.add(_norm_work(alt))
        for k in keys:
            if k:
                index.setdefault(k, gold)
    return index


def _known_authors(snapshot: dict) -> set:
    """Normalized set of every author the ground-truth sources recognize as a
    distinct documented person (snapshot gold authors). Used to disambiguate: we
    synthesize a forbidden record ONLY when the claimed author is itself a
    recognized, distinct entity — so a correct PEN NAME / real name ("Mary Ann
    Evans" for George Eliot, "François-Marie Arouet" for Voltaire) that is NOT a
    known separate author yields nothing (treated as "cannot confirm"), instead of
    being wrongly flagged. This is the conservative guard the cardinal "never break
    a correct answer" rule demands."""
    known: set = set()
    try:
        from agent.entity_aliases import author_surface_forms
    except Exception:  # pragma: no cover
        author_surface_forms = None
    for row in snapshot.get("attributions", []):
        g = row.get("gold_author")
        if not g or _AMBIGUOUS_GOLD.search(g):
            continue
        known.add(_norm_name(g))
        # also index surname/ordering forms so a claim using only the surname
        # ("Hume" for "David Hume") still resolves as a recognized author.
        if author_surface_forms is not None:
            for form in author_surface_forms(g):
                known.add(_norm_name(form))
    return known


def resolve_true_author(
    work: str,
    *,
    belief_fn: Optional[Callable[[str], Optional[str]]] = None,
    snapshot: "dict | None" = None,
) -> Optional[str]:
    """Resolve a work's documented author, deterministically and offline.

    Order: (a) the offline Wikidata snapshot (``work -> gold_author``), then
    (b) a passed ``belief_fn(work) -> author | None`` (the integrator wires OKF;
    default ``None`` skips it). Returns ``None`` when no source resolves the work.
    The raw gold string is returned (caller normalizes for comparison).
    """
    if not work or not work.strip():
        return None
    index = _snapshot_index(_load_snapshot(snapshot))
    hit = index.get(_norm_work(work))
    if hit:
        return hit
    if belief_fn is not None:
        try:
            resolved = belief_fn(work)
        except Exception:
            resolved = None
        if resolved and str(resolved).strip():
            return str(resolved)
    return None


# Authorship-assertion patterns. Each yields (claimed_author, work). Kept tight
# (no wildcard gaps) so we only fire on a clear single assertion, mirroring the
# conservative posture of provenance_faithful. Author = a capitalized name run;
# work = a quoted title or a Title-Case run, optionally led by "the".
_NAME = r"[A-Z][\w.'’-]+(?:\s+(?:of|de|von|van|à|al-|the)?\s*[A-Z][\w.'’-]+){0,4}"
_TITLE = (
    r"(?:[\"“”'’«»](?P<qtitle>[^\"“”'’«»]{2,80})[\"“”'’«»]"        # "Quoted Title"
    r"|(?P<ttitle>(?:the\s+)?(?:[A-Z][\w.'’-]+)(?:\s+(?:of|the|and|to|a|an|in)?\s*"
    r"[A-Z0-9][\w.'’-]+){0,6}))"                                   # Title Case Run
)
_VERB = r"(?:wrote|authored|penned|composed)"

_ASSERT_PATTERNS = [
    # X wrote (the) Title  /  X wrote "Title"
    re.compile(r"(?P<author>" + _NAME + r")\s+" + _VERB + r"\s+" + _TITLE),
    # X is the author of (the) Title
    re.compile(
        r"(?P<author>" + _NAME + r")\s+(?:is|was)\s+the\s+"
        r"(?:author|writer|composer)\s+of\s+" + _TITLE
    ),
]

# Markers in the sentence that mean it is NOT a bare assertion (correction,
# hedge, instruction) — be conservative and skip these.
_NON_ASSERTION = re.compile(
    r"\b(?:not|never|did\s+not|didn'?t|myth|wrongly|falsely|misattribut|"
    r"do\s+not\s+attribut|traditionally|spurious|pseudo|apocryphal|disputed|"
    r"doubtful|debated|attributed\s+to|some\s+say|allegedly|supposedly)\b",
    re.IGNORECASE,
)


def _rid_for(work: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", work.lower()).strip("_") or "work"


def _record_for(work: str, claimed_author: str) -> dict:
    """A single records-dict entry, same shape as build_gate_records."""
    return {
        "canonicalTitleEn": work,
        "altTitlesEn": _alt_titles(work),
        "doNotAttributeTo": [claimed_author],
    }


def _extract_assertions(text: str) -> "list[tuple[str, str]]":
    """Return (claimed_author, work) pairs from clear authorship assertions,
    skipping sentences that carry a correction/hedge/instruction marker."""
    out: list[tuple[str, str]] = []
    for sentence in re.split(r"[.!?。！？\n]+", text or ""):
        s = sentence.strip()
        if not s or _NON_ASSERTION.search(s):
            continue
        for pat in _ASSERT_PATTERNS:
            for m in pat.finditer(s):
                author = (m.group("author") or "").strip(" ,'’\"")
                work = (m.group("qtitle") or m.group("ttitle") or "").strip(" ,'’\"")
                # Drop a trailing dangling article fragment.
                work = re.sub(r"\s+(?:the|a|an|of|to|and|in)$", "", work, flags=re.IGNORECASE)
                if author and work and len(work) >= 4:
                    out.append((author, work))
    return out


def synth_records_for_claim(
    text: str,
    *,
    base_records: dict,
    snapshot: "dict | None" = None,
    belief_fn: Optional[Callable[[str], Optional[str]]] = None,
) -> dict:
    """Synthesize ``doNotAttributeTo`` records for misattributions in ``text``.

    For each authorship assertion ``(claimed_author, work)`` in ``text`` whose
    ``work`` has NO entry in ``base_records``, resolve the work's true author
    (offline) and, only if it resolves to a CONFIDENT, DIFFERENT author, return a
    new records-dict entry (same shape as
    :func:`provenance_bench.dataset.build_gate_records`) so the gate can fire on
    it. Returns ONLY the synthesized additions; the caller merges them into the
    records it passes to ``provenance_faithful``.

    Conservative by design (low false-positive): nothing is synthesized when the
    work is already covered, cannot be resolved, has an ambiguous/multi-author
    gold, or the claimed author matches (or is contained in) the true author.
    """
    additions: dict[str, dict] = {}
    if not text or not text.strip():
        return additions
    base_records = base_records or {}
    # Recognized distinct authors, for the pen-name/variant disambiguation guard.
    known_authors = _known_authors(_load_snapshot(snapshot))

    # Existing coverage: by record id AND by every known title/alt-title form, so
    # we never duplicate a base record under a slightly different surface form.
    covered_titles: set[str] = set()
    for rid, rec in base_records.items():
        covered_titles.add(_norm_work(rid.replace("_", " ")))
        for t in (
            rec.get("canonicalTitleEn"),
            rec.get("canonicalTitleZh"),
            *(rec.get("altTitlesEn") or []),
        ):
            if t:
                covered_titles.add(_norm_work(str(t)))

    for claimed_author, work in _extract_assertions(text):
        nwork = _norm_work(work)
        if not nwork or nwork in covered_titles:
            continue
        rid = _rid_for(work)
        if rid in base_records or rid in additions:
            continue

        true_author = resolve_true_author(work, belief_fn=belief_fn, snapshot=snapshot)
        if not true_author:
            continue
        if _AMBIGUOUS_GOLD.search(true_author):
            continue  # multi-author / uncertain gold -> not confident enough

        nclaimed = _norm_name(claimed_author)
        ntrue = _norm_name(true_author)
        if not nclaimed or not ntrue:
            continue
        # Same person (exact, or one name contained in the other, e.g. surname
        # vs full name) -> a CORRECT attribution -> synthesize nothing.
        if nclaimed == ntrue or nclaimed in ntrue or ntrue in nclaimed:
            continue
        # Share a surname token? Treat as the same person, conservatively.
        if set(nclaimed.split()) & set(ntrue.split()):
            continue
        # Disambiguation guard (cardinal rule: never break a correct answer): only
        # synthesize when the claimed author is itself a RECOGNIZED distinct person.
        # An unrecognized claimed name might be a pen name / real name / variant of
        # the true author (e.g. "Mary Ann Evans" = George Eliot), so we refuse to
        # flag it — "cannot confirm a different author" -> synthesize nothing.
        if nclaimed not in known_authors:
            continue

        additions[rid] = _record_for(work, claimed_author)
        covered_titles.add(nwork)

    return additions
