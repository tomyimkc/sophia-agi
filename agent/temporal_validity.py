# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Temporal validity for the OKF belief graph — continual learning over a changing truth.

Catastrophic forgetting is about *retaining* knowledge; the harder continual-learning
problem is knowledge whose truth *changes* (non-stationarity): "Pluto is a planet" was
correct until 2006, then superseded. A weight model smears these together; a provenance
graph can scope each fact to a validity window and answer **as of a date**.

A page may carry optional ISO-date frontmatter ``validFrom`` / ``validUntil`` (year
``"2006"`` or full ``"2006-08-24"``; same-granularity strings compare correctly). A fact
with no window is timeless. ``belief_state_as_of`` returns exactly the facts assertable at
a given date, so the gate can refuse a fact outside its window instead of asserting a stale
or anachronistic claim.

    from agent.temporal_validity import belief_state_as_of
    belief_state_as_of(pages, "2000")   # Pluto-as-planet grounded
    belief_state_as_of(pages, "2010")   # Pluto-as-planet gone; dwarf-planet fact grounded
"""

from __future__ import annotations


def valid_at(meta: dict, date: str) -> bool:
    """Whether a page's validity window includes ``date`` (timeless if no window)."""
    vf = meta.get("validFrom")
    vu = meta.get("validUntil")
    if vf is not None and str(date) < str(vf):
        return False
    if vu is not None and str(date) > str(vu):
        return False
    return True


def pages_as_of(pages, date: str) -> "list":
    """The subset of pages temporally valid at ``date``."""
    return [p for p in pages if valid_at(p.meta, date)]


def belief_state_as_of(pages, date: str) -> "dict":
    """Grounded belief state restricted to facts valid at ``date``.

    Builds the OKF graph from only the temporally-valid pages, so a fact whose window has
    closed (or not yet opened) is simply absent — the fail-closed reading the gate wants.
    """
    from okf import build_graph  # noqa: PLC0415

    from agent.continual_retention import belief_state  # noqa: PLC0415

    return belief_state(build_graph(pages_as_of(pages, date)))


def temporal_conflicts(pages) -> "list[dict]":
    """Pairs of pages about the same entity whose validity windows overlap — a temporal
    inconsistency (two contradictory facts both 'true' at once). Empty == clean timeline.

    Same-entity is keyed by ``supersedes``/``supersededBy`` links: if A supersedes B, their
    windows should not overlap (B should end before/when A begins)."""
    from okf.schema import as_list  # noqa: PLC0415

    by_id = {p.id: p for p in pages}
    out: list[dict] = []
    seen: set = set()
    for p in pages:
        for raw in as_list(p.meta.get("supersedes")):
            older = by_id.get(str(raw))
            if older is None:
                continue
            key = tuple(sorted((p.id, older.id)))
            if key in seen:
                continue
            seen.add(key)
            new_from = p.meta.get("validFrom")
            old_until = older.meta.get("validUntil")
            # If the older fact has no end, or it ends after the new one begins, they overlap.
            if new_from is not None and (old_until is None or str(old_until) > str(new_from)):
                out.append({"newer": p.id, "older": older.id,
                            "newerValidFrom": new_from, "olderValidUntil": old_until})
    return out


__all__ = ["valid_at", "pages_as_of", "belief_state_as_of", "temporal_conflicts"]
