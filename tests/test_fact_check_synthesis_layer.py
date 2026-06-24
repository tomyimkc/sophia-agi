#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the optional synthesized-verifier slot in the fact-check gate."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.fact_check_gate import AtomicClaim, LayerResult, fact_check_claim  # noqa: E402


def test_admitted_synthesized_verifier_can_accept_unknown_type() -> None:
    claim = AtomicClaim("The SKU code ABC-2468 has an even numeric suffix", "open_empirical")

    def verifier(c: AtomicClaim):
        suffix = int(c.text.rsplit("-", 1)[-1].split()[0])
        if suffix % 2 == 0:
            return LayerResult("synthesized_verifier", "accepted", "held-out admitted even-suffix verifier", confidence=0.93,
                               details={"metaVerified": True, "heldoutN": 40})
        return LayerResult("synthesized_verifier", "rejected", "suffix is not even", confidence=0.93)

    dec = fact_check_claim(claim, synthesized_verifier=verifier)
    assert dec.verdict == "accepted"
    assert dec.layers[1].layer == "synthesized_verifier"


def test_synthesized_verifier_errors_hold_fail_closed() -> None:
    claim = AtomicClaim("The SKU code ABC-2468 has an even numeric suffix", "open_empirical")

    def bad(_claim):
        raise RuntimeError("boom")

    dec = fact_check_claim(claim, synthesized_verifier=bad)
    assert dec.verdict == "held"
    assert any(l.layer == "synthesized_verifier" and l.verdict == "held" for l in dec.layers)


def main() -> int:
    test_admitted_synthesized_verifier_can_accept_unknown_type()
    test_synthesized_verifier_errors_hold_fail_closed()
    print("test_fact_check_synthesis_layer: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
