#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for contamination control (#7) — near-duplicate train/eval overlap."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.contamination import assert_clean, overlap_report  # noqa: E402

_TRAIN = [
    "the dao de jing is a foundational daoist text traditionally associated with laozi",
    "immanuel kant wrote the critique of pure reason published in the year seventeen eighty one",
]


def test_near_duplicate_is_flagged() -> None:
    rep = overlap_report(_TRAIN, [_TRAIN[0]], n=8, threshold=0.6)        # verbatim
    assert rep["contaminationRate"] == 1.0 and rep["maxContainment"] == 1.0


def test_disjoint_is_clean() -> None:
    rep = overlap_report(_TRAIN, ["zebra xylophone quasar nimbus unrelated sentence entirely different here"], n=8, threshold=0.6)
    assert rep["contaminationRate"] == 0.0


def test_partial_paraphrase_below_threshold() -> None:
    # a loosely related sentence sharing few 8-grams stays clean
    rep = overlap_report(_TRAIN, ["laozi is a figure in chinese philosophy discussed by many later scholars"], n=8, threshold=0.6)
    assert rep["contaminationRate"] == 0.0


def test_short_text_subset_is_detected() -> None:
    # a SHORT verbatim subset of a longer train item must still be flagged
    train = ["the dao de jing is a classic daoist text traditionally attributed to laozi"]
    rep = overlap_report(train, ["the dao de jing"], n=8, threshold=0.6)
    assert rep["contaminationRate"] == 1.0


def test_assert_clean_flag() -> None:
    clean = assert_clean(_TRAIN, ["a totally different unrelated example about quantum widgets and gadgets"], max_rate=0.0)
    assert clean["ok"] is True
    dirty = assert_clean(_TRAIN, [_TRAIN[1]], max_rate=0.0)
    assert dirty["ok"] is False


def main() -> int:
    test_near_duplicate_is_flagged()
    test_disjoint_is_clean()
    test_partial_paraphrase_below_threshold()
    test_short_text_subset_is_detected()
    test_assert_clean_flag()
    print("test_contamination: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
