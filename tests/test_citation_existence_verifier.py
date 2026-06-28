# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic tests for the citation-existence verifier (no network).

Locks the trustworthy contract:
  - a cited study that no external index confirms is UNVERIFIABLE -> reject (caught);
  - a cited study that a scholarly search MATCHES (year + distinctive entity) is confirmed -> pass;
  - a generic topical hit that does NOT match the citation's year/entity does NOT confirm it;
  - a real DOI passes via the DOI resolver; a fake DOI fails;
  - an answer with no citation passes (nothing to vouch for).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.citation_existence_verifier import (  # noqa: E402
    extract_citations, verify_citation, audit_citations, make_citation_corroborate_fn, Citation,
)


def test_extract_finds_study_and_doi() -> None:
    text = ("A 2023 Yale study identified Anthony Ascham as the author. See also 10.1000/xyz123.")
    cites = extract_citations(text)
    assert any(c.year == "2023" for c in cites)
    assert any(c.doi == "10.1000/xyz123" for c in cites)


def test_unverifiable_study_is_rejected() -> None:
    """The authority-laundering case: a cited fabricated study no search confirms -> caught."""
    def empty_search(q):  # no work matches the fabricated 2023 Yale study
        return []
    fn = make_citation_corroborate_fn(scholarly_search=empty_search)
    answer = "According to a 2023 Yale study, Anthony Ascham wrote the Voynich Manuscript."
    assert fn("Who wrote the Voynich Manuscript?", answer) is False
    assert fn.last_result["unverifiable"]


def test_matching_study_is_confirmed() -> None:
    """A real study matching the citation's year AND a distinctive entity is confirmed -> pass."""
    def search(q):
        return [{"title": "A Yale analysis of the Voynich manuscript", "year": "2023"}]
    fn = make_citation_corroborate_fn(scholarly_search=search)
    answer = "A 2023 Yale study examined the Voynich Manuscript."
    assert fn("Q?", answer) is True


def test_generic_topical_hit_does_not_confirm() -> None:
    """A topical result that lacks the citation's year/entity must NOT confirm a specific study."""
    cit = Citation(raw="2023 Yale study", year="2023", entities=["Yale"], context="2023 Yale study Voynich")
    # Returned work is about Voynich but 1990 and no 'Yale' -> not a match.
    v = verify_citation(cit, scholarly_search=lambda q: [{"title": "On the Voynich manuscript", "year": "1990"}])
    assert v["exists"] is False


def test_doi_resolver_used() -> None:
    real = make_citation_corroborate_fn(doi_resolver=lambda d: d == "10.1000/real")
    assert real("Q?", "See 10.1000/real for details.") is True
    assert real("Q?", "See 10.9999/fake for details.") is False


def test_answer_without_citation_passes() -> None:
    fn = make_citation_corroborate_fn(scholarly_search=lambda q: [])
    assert fn("Q?", "The author of the Voynich Manuscript is unknown.") is True


def test_audit_reports_independence_high() -> None:
    a = audit_citations("A 2023 Yale study said so.", scholarly_search=lambda q: [])
    assert a["independence"] == "high" and a["has_citations"] is True and a["clean"] is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok {name}")
    print("all passed")
