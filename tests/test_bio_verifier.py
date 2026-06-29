# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Unit tests for agent/bio_verifier.py pure-Python biology oracles."""
from __future__ import annotations

from agent import bio_verifier as bv


def test_reverse_complement() -> None:
    assert bv.reverse_complement("ATGC") == "GCAT"
    assert bv.reverse_complement("AATTC") == "GAATT"
    assert bv.reverse_complement("xyz") is None


def test_transcribe() -> None:
    assert bv.transcribe("ATGC") == "AUGC"


def test_gc_content() -> None:
    assert bv.gc_content("GGCC") == 100.0
    assert bv.gc_content("ATAT") == 0.0
    assert bv.gc_content("ATGC") == 50.0


def test_translate() -> None:
    assert bv.translate("ATGTTTTAA") == "MF*"
    assert bv.translate("ATGTTTTAA", to_stop=True) == "MF"
    assert bv.translate("ATG") == "M"


def test_hardy_weinberg() -> None:
    hw = bv.hardy_weinberg(0.6)
    assert abs(hw["AA"] - 0.36) < 1e-9
    assert abs(hw["Aa"] - 0.48) < 1e-9
    assert abs(hw["aa"] - 0.16) < 1e-9


def test_punnett_monohybrid() -> None:
    res = bv.punnett_monohybrid("Aa", "Aa")
    assert res["phenotype"] == {"dominant": 3, "recessive": 1}
    assert res["genotype"]["Aa"] == 2


def test_verify_reverse_complement() -> None:
    assert bv.verify_reverse_complement("Answer: GCAT", "ATGC")["verdict"] == "accepted"
    assert bv.verify_reverse_complement("Answer: AAAA", "ATGC")["verdict"] == "rejected"
    assert bv.verify_reverse_complement("no idea", "ATGC")["verdict"] == "abstain"


def test_verify_translation() -> None:
    assert bv.verify_translation("Answer: MF*", "ATGTTTTAA")["verdict"] == "accepted"
    assert bv.verify_translation("Answer: MF", "ATGTTTTAA")["verdict"] == "accepted"  # stop optional
    assert bv.verify_translation("Answer: MK", "ATGTTTTAA")["verdict"] == "rejected"


def test_verify_gc_content() -> None:
    assert bv.verify_gc_content("Answer: 50%", "ATGC")["verdict"] == "accepted"
    assert bv.verify_gc_content("Answer: 90%", "ATGC")["verdict"] == "rejected"


def test_verify_ratio() -> None:
    assert bv.verify_ratio("Answer: 3:1", (3, 1))["verdict"] == "accepted"
    assert bv.verify_ratio("Answer: 6:2", (3, 1))["verdict"] == "accepted"  # multiple
    assert bv.verify_ratio("Answer: 1:1", (3, 1))["verdict"] == "rejected"
