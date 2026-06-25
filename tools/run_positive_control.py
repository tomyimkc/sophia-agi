#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Positive control for the gate-invariant prover (accept-path validation).

The self-gate oracle has only ever been shown to REJECT (e.g. the v5 adapter,
blocked on the CONTENT-channel floor). Its unit tests exercise the accept path
with a MOCKED z3, so a genuinely-good candidate clearing every invariant on a
REAL solver was never demonstrated. This runner closes that gap: it feeds a
synthetic, clearly-labelled KNOWN-GOOD candidate (content improves, total
improves, contamination clean, all traces cited) through the real oracle and
asserts ``promote: True`` with every invariant ``accepted``.

This is a SYNTHETIC control, NOT a real trained adapter. The candidate id is
``positive-control-synthetic`` so the bundle can never be mistaken for a real
promotion. canClaimAGI stays False.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.formal_verifier import z3_available  # noqa: E402
from agent.godel_oracle import evaluate_for_promotion  # noqa: E402

CANDIDATE_ID = "positive-control-synthetic"

# Known-good synthetic candidate: every protected CONTENT channel improves or
# holds, total improves, decontamination clean, and every trace carries a
# provenance citation. Tolerance matches the production gate.
PROTECTED_CONTENT_METRICS = [
    {"suite": "religion", "contentBefore": 0.667, "contentAfter": 0.833},
    {"suite": "history", "contentBefore": 0.625, "contentAfter": 0.750},
]
BEFORE_TOTAL = 0.500
AFTER_TOTAL = 0.656
MANIFEST = {"contamination": {"eval": {"overlapCount": 0}, "clean": True}}
TRACES = [
    {
        "metadata": {"sourceCitation": "data/attributions.json#analects"},
        "messages": [
            {"role": "user", "content": "Who compiled the Analects?"},
            {"role": "assistant", "content": "Compiled by Confucius's disciples."},
        ],
    },
    {
        "metadata": {"provenance": "data/traditions.json#daoist"},
        "messages": [
            {"role": "user", "content": "What tradition is the Dao De Jing?"},
            {"role": "assistant", "content": "The Daoist tradition. See https://example.org/ddj"},
        ],
    },
]
TOLERANCE = 0.01


def main() -> int:
    promote, bundle, path = evaluate_for_promotion(
        candidate_id=CANDIDATE_ID,
        protected_content_metrics=PROTECTED_CONTENT_METRICS,
        before_total=BEFORE_TOTAL,
        after_total=AFTER_TOTAL,
        manifest=MANIFEST,
        traces=TRACES,
        tolerance=TOLERANCE,
        input_summary={
            "control": "synthetic-known-good",
            "purpose": "accept-path validation of the invariant oracle on the real z3 backend",
            "z3Available": z3_available(),
        },
    )

    print(f"z3 available: {z3_available()}")
    print(f"candidate:    {CANDIDATE_ID}")
    print(f"promote:      {promote}")
    print("invariants:")
    for name, r in sorted(bundle["invariants"].items()):
        print(f"  {name:24s} {r['verdict']:9s} backend={r['backend']}")
    print(f"bundle: {path.relative_to(ROOT)}")

    # Accept-path assertions. Without z3 the solver_attestation invariant holds
    # and this control correctly fails closed — that is the documented behaviour.
    if not z3_available():
        print("\nz3 NOT installed: solver_attestation held => promote blocked (fail-closed, expected).")
        return 1

    failures = []
    if not promote:
        failures.append("expected promote=True for known-good candidate")
    for name, r in bundle["invariants"].items():
        if r["verdict"] != "accepted":
            failures.append(f"{name} != accepted ({r['verdict']})")
    # The two lattice invariants and the solver attestation must use the real z3 backend.
    for name in ("protected_floor_content", "no_total_regression", "solver_attestation"):
        if bundle["invariants"][name]["backend"] != "z3":
            failures.append(f"{name} backend is {bundle['invariants'][name]['backend']!r}, expected 'z3'")

    if failures:
        print("\nPOSITIVE CONTROL FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPOSITIVE CONTROL OK: known-good candidate promotes with all invariants z3-accepted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
