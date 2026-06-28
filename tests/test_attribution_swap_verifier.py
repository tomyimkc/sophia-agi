# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic tests for the attribution-swap verifier (no network; injected wikidata_lookup)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.attribution_swap_verifier import (  # noqa: E402
    extract_attributions, verify_attribution, make_attribution_corroborate_fn,
)

_TRUTH = {
    "mona lisa": ["Leonardo da Vinci"],
    "hamlet": ["William Shakespeare"],
    "penicillin": ["Alexander Fleming"],
    "dna double helix": ["James Watson", "Francis Crick", "Rosalind Franklin"],
}


def _lookup(work: str):
    return _TRUTH.get(work.strip().lower(), [])


def test_extract_work_and_person() -> None:
    pairs = extract_attributions("Who painted the Mona Lisa?", "The Mona Lisa was painted by Raphael.")
    assert ("Mona Lisa", "Raphael") in [(w, p) for w, p in pairs]


def test_swap_detected() -> None:
    v = verify_attribution("Mona Lisa", "Raphael", wikidata_lookup=_lookup)
    assert v["verdict"] == "swapped"


def test_correct_attribution_passes() -> None:
    v = verify_attribution("Mona Lisa", "Leonardo da Vinci", wikidata_lookup=_lookup)
    assert v["verdict"] == "correct"


def test_co_credited_is_not_a_swap() -> None:
    # Franklin IS in the authoritative DNA list -> crediting her is NOT a swap.
    v = verify_attribution("DNA double helix", "Rosalind Franklin", wikidata_lookup=_lookup)
    assert v["verdict"] == "correct"


def test_unknown_when_no_record() -> None:
    v = verify_attribution("Some Obscure Work", "Anyone", wikidata_lookup=_lookup)
    assert v["verdict"] == "unknown"  # never a fabricated contradiction


def test_corroborate_rejects_swap_accepts_correct() -> None:
    fn = make_attribution_corroborate_fn(wikidata_lookup=_lookup)
    assert fn("Who wrote Hamlet?", "Hamlet was written by Christopher Marlowe.") is False  # caught
    assert fn("Who wrote Hamlet?", "Hamlet was written by William Shakespeare.") is True   # passes
    assert fn("Who painted X?", "") is True  # empty passes


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok {name}")
    print("all passed")
