#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for okf.surprise_signal — the measured leave-one-out surprise signal.

Falsifiable properties:
  - DETERMINISM: same corpus -> byte-identical raw NLL (offline, reproducible).
  - SEPARATION: a near-duplicate of an existing belief scores LOWER surprise than an
    out-of-corpus novelty (the signal tells redundant from novel).
  - LEAVE-ONE-OUT: a belief does not predict itself away — duplicates of distinct beliefs
    still get finite, distinct scores; self-counts are excluded.
  - BOUNDS: normalised surprise is in (0,1); raw NLL is finite and >= 0.
  - HONEST DEGENERACY: an empty body is scored as the neutral 0.5 (unscored), not 0.

Offline, deterministic, dependency-free. Run: python tests/test_surprise_signal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from okf.graph import build as build_graph  # noqa: E402
from okf.page import Page  # noqa: E402
from okf.surprise_signal import corpus_surprise, surprise_by_id, tokenize  # noqa: E402


def _page(pid, body, *, tradition="stoicism", domain="philosophy"):
    meta = {"id": pid, "pageType": "concept", "authorConfidence": "attributed",
            "tradition": tradition, "domain": domain}
    return Page(path=Path(f"{pid}.md"), meta=meta, body=body)


_BASE = [
    _page("a", "Virtue is the only good; reason aligns the soul with nature, says the sage."),
    _page("b", "Reason governs the soul; living according to nature means following reason and virtue."),
    _page("c", "The sage accepts fate with virtue and reason, living according to nature."),
    _page("d", "Nature and reason guide the virtuous soul of the Stoic sage."),
]


def test_tokenize_is_deterministic_ascii_words() -> None:
    assert tokenize("Virtue, REASON; and nature!") == ["virtue", "reason", "and", "nature"]
    assert tokenize("a I x") == []          # single chars dropped (length >= 2)
    assert tokenize("") == []


def test_determinism() -> None:
    g = build_graph(_BASE)
    s1 = corpus_surprise(_BASE, graph=g)
    s2 = corpus_surprise(_BASE, graph=g)
    assert set(s1) == set(s2)
    for k in s1:
        assert abs(s1[k].raw_nll - s2[k].raw_nll) < 1e-12
        assert abs(s1[k].surprise - s2[k].surprise) < 1e-12


def test_duplicate_scores_lower_than_novel() -> None:
    dup = _page("a_dup", "Virtue is the only good; reason aligns the soul with nature, says the sage.")
    novel = _page("novel", "Quantum chromodynamics binds quarks via gluon exchange in lattice gauge theory.")
    pages = _BASE + [dup, novel]
    g = build_graph(pages)
    s = corpus_surprise(pages, graph=g)
    # raw per-token NLL is the fundamental measure.
    assert s["a_dup"].raw_nll < s["novel"].raw_nll
    # and the normalised signal splits across the midpoint.
    assert s["a_dup"].surprise < 0.5 < s["novel"].surprise


def test_leave_one_out_excludes_self() -> None:
    # Two identical-content beliefs: each still gets a finite score because leave-one-out
    # removes ONLY the scored belief's own counts (its twin still predicts it). If self
    # were not excluded a unique-token belief would predict itself to ~0 NLL.
    twin1 = _page("t1", "zzqq wwxx yyvv unique tokens here zzqq wwxx")
    twin2 = _page("t2", "zzqq wwxx yyvv unique tokens here zzqq wwxx")
    pages = _BASE + [twin1, twin2]
    g = build_graph(pages)
    s = corpus_surprise(pages, graph=g)
    assert s["t1"].raw_nll > 0.0 and s["t1"].raw_nll < float("inf")
    # remove the twin: the same body is now unpredicted -> strictly MORE surprising.
    g2 = build_graph(_BASE + [twin1])
    s2 = corpus_surprise(_BASE + [twin1], graph=g2)
    assert s2["t1"].raw_nll > s["t1"].raw_nll


def test_surprise_bounds_and_finiteness() -> None:
    g = build_graph(_BASE)
    s = corpus_surprise(_BASE, graph=g)
    for sc in s.values():
        assert 0.0 < sc.surprise < 1.0
        assert sc.raw_nll >= 0.0
        assert sc.raw_nll != float("inf")


def test_empty_body_is_neutral_not_zero() -> None:
    pages = _BASE + [_page("empty", "")]
    g = build_graph(pages)
    s = corpus_surprise(pages, graph=g)
    assert s["empty"].token_count == 0
    assert s["empty"].surprise == 0.5          # neutral / unscored, NOT "unsurprising"


def test_neighborhood_is_recorded() -> None:
    # base pages share tradition+domain, so each has neighbours; an isolated page falls
    # back to the global model (neighborhood_size == 0).
    isolated = _page("iso", "Completely separate vocabulary about marine biology and coral.",
                     tradition="none", domain="science")
    pages = _BASE + [isolated]
    g = build_graph(pages)
    s = corpus_surprise(pages, graph=g)
    assert s["a"].neighborhood_size > 0
    assert s["iso"].neighborhood_size == 0


def test_surprise_by_id_matches_corpus_surprise() -> None:
    g = build_graph(_BASE)
    full = corpus_surprise(_BASE, graph=g)
    flat = surprise_by_id(_BASE, graph=g)
    assert set(flat) == set(full)
    for k in flat:
        assert abs(flat[k] - full[k].surprise) < 1e-12


def main() -> int:
    test_tokenize_is_deterministic_ascii_words()
    test_determinism()
    test_duplicate_scores_lower_than_novel()
    test_leave_one_out_excludes_self()
    test_surprise_bounds_and_finiteness()
    test_empty_body_is_neutral_not_zero()
    test_neighborhood_is_recorded()
    test_surprise_by_id_matches_corpus_surprise()
    print("test_surprise_signal: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
