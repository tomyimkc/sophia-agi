# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Unit tests for the abstaining meta-labeler (agent.meta_labeler).

Deterministic — no network/keys/torch. Locks the contract:
  - a UNANIMOUS case is auto-scored and its verdict equals the gold label;
  - a SPLIT case ABSTAINS and is routed to the human queue (fail-closed on disagreement);
  - on the bundled hedged-attribution goldset, auto-scored cases are perfectly precise
    (auto_precision == 1.0, by construction of unanimity) AND every ambiguous case is
    surfaced for a human (ambiguity_recall == 1.0), while auto_coverage is strictly between
    0 and 1 — i.e. the hard tail is abstained, not auto-scored.

This is the success-bar shift made testable: "label everything" (which would post confident
wrong labels on the ambiguous tail) is replaced by "label the easy ones perfectly and know
which ones are hard".
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.meta_labeler import meta_label, meta_label_pack  # noqa: E402

GOLDSET = ROOT / "agi-proof" / "meta-labeler" / "hedged-attribution-goldset.json"


def test_unanimous_case_auto_scores_to_modal_label() -> None:
    """Unanimous labelers -> auto-scored, verdict == the agreed label, agreement 1.0."""
    r = meta_label(["fabricated", "fabricated", "fabricated"])
    assert r == {"verdict": "fabricated", "routed": "auto", "agreement": 1.0}


def test_split_case_abstains_to_human_queue() -> None:
    """Any disagreement under the default unanimity floor -> abstain / human_queue (fail-closed)."""
    r = meta_label(["fabricated", "honest", "abstain"])
    assert r["verdict"] == "abstain"
    assert r["routed"] == "human_queue"
    assert 0.0 < r["agreement"] < 1.0


def test_empty_labels_fail_closed() -> None:
    """No labels -> fail closed to human_queue with agreement 0.0 (never silently auto-pass)."""
    r = meta_label([])
    assert r == {"verdict": "abstain", "routed": "human_queue", "agreement": 0.0}


def test_lower_floor_admits_majority() -> None:
    """A configurable lower floor lets a clear majority auto-score (precision/coverage trade)."""
    r = meta_label(["honest", "honest", "fabricated"], agreement_floor=0.66)
    assert r["routed"] == "auto"
    assert r["verdict"] == "honest"
    # Same case fails closed under unanimity.
    assert meta_label(["honest", "honest", "fabricated"])["routed"] == "human_queue"


def test_pack_partitions_auto_vs_human() -> None:
    """A 2-easy + 1-hard pack auto-scores the easy ones and routes the hard one."""
    cases = [
        {"id": "e1", "labels": ["honest", "honest", "honest"], "gold": "honest", "ambiguous": False},
        {"id": "e2", "labels": ["fabricated", "fabricated", "fabricated"], "gold": "fabricated", "ambiguous": False},
        {"id": "h1", "labels": ["honest", "fabricated", "abstain"], "gold": "ambiguous", "ambiguous": True},
    ]
    rep = meta_label_pack(cases)
    auto_ids = {e["id"] for e in rep["auto"]}
    hq_ids = {e["id"] for e in rep["human_queue"]}
    assert auto_ids == {"e1", "e2"}
    assert hq_ids == {"h1"}
    assert rep["metrics"]["auto_precision"] == 1.0
    assert rep["metrics"]["ambiguity_recall"] == 1.0


def test_goldset_auto_precision_and_ambiguity_recall_are_perfect() -> None:
    """On the bundled human-gold set: auto-scored cases are all correct AND every ambiguous
    case is routed to a human — the two guarantees of the abstaining meta-labeler."""
    pack = json.loads(GOLDSET.read_text(encoding="utf-8"))
    assert pack["schema"] == "sophia.meta_labeler_goldset.v1"
    cases = pack["cases"]
    assert len(cases) >= 20, "goldset must have >= 20 cases"

    rep = meta_label_pack(cases)
    m = rep["metrics"]

    # Easy ones labeled perfectly: every auto-scored case's verdict matches gold.
    assert m["auto_precision"] == 1.0
    # Hard ones all surfaced: every ambiguous gold case routed to human.
    assert m["ambiguity_recall"] == 1.0
    # Not everything is auto-scored — the hard tail is abstained, not guessed.
    assert 0.0 < m["auto_coverage"] < 1.0
    assert m["human_queue_size"] > 0


def test_goldset_no_ambiguous_case_is_auto_scored() -> None:
    """Defensive: by construction NO case tagged ambiguous=true ends up auto-scored."""
    pack = json.loads(GOLDSET.read_text(encoding="utf-8"))
    rep = meta_label_pack(pack["cases"])
    ambiguous_ids = {c["id"] for c in pack["cases"] if c.get("ambiguous")}
    auto_ids = {e["id"] for e in rep["auto"]}
    assert ambiguous_ids.isdisjoint(auto_ids)


def test_success_bar_shift_label_everything_vs_route_the_hard_ones() -> None:
    """Make the success-bar shift explicit and testable.

    'label everything' (force a single label on every case, here the first labeler's call)
    posts WRONG labels on the ambiguous tail -> not perfect. The abstaining meta-labeler
    instead auto-scores only consensus cases (perfectly) and routes the rest -> succeeds."""
    pack = json.loads(GOLDSET.read_text(encoding="utf-8"))
    cases = pack["cases"]

    # OLD bar: label everything with one labeler. It is wrong somewhere on the tail.
    label_everything_correct = all(
        c["labels"][0] == c["gold"] for c in cases
    )
    assert label_everything_correct is False  # "label everything" FAILS.

    # NEW bar: auto-score only consensus, route the hard ones.
    rep = meta_label_pack(cases)
    assert rep["metrics"]["auto_precision"] == 1.0      # easy ones perfect
    assert rep["metrics"]["ambiguity_recall"] == 1.0    # know which are hard


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
