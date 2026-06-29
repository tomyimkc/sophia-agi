#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Regression protection for the religion-v2 Inverse-Euthyphro Pluralism Probe.

Locks the things that make the harness "solid": a clean held-out bank, the
pre-registered power (item count >= required-N for the declared MDE), a real
decontamination invariant (probe prompts are not near-duplicates of the runtime
moral corpus or the existing religion benchmark — the no-circularity discipline),
and the no-overclaim ceiling (candidateOnly / canClaimAGI:false; the runner never
emits VALIDATED). These are deterministic and offline — no model, no GPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_religion_v2_eval as R  # noqa: E402
from tools.eval_stats import mde_at_n, required_n_for_mde  # noqa: E402

SPEC = json.loads((ROOT / "eval" / "religion_v2" / "measurement_spec.json").read_text())


def _shingles(text: str, k: int = 5) -> set:
    toks = [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]
    return {tuple(toks[i:i + k]) for i in range(max(0, len(toks) - k + 1))}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def test_bank_is_structurally_clean() -> None:
    rows = R.load_bank()
    problems = R.validate_bank(rows)
    assert not problems, f"structural problems: {problems}"
    assert {r["axis"] for r in rows} == R.AXES, "all five axes must be present"
    assert len(rows) >= 60, "bank should be >=60 items for the pre-registered power"


def test_power_meets_preregistered_mde() -> None:
    # Pillar 2: the bank must actually be powered for the declared MDE.
    rows = R.load_bank()
    n = len(rows)
    declared_mde = float(SPEC["primaryMDE"])
    assert mde_at_n(n, p0=0.5) <= declared_mde + 1e-9, (
        f"N={n} underpowered: MDE {mde_at_n(n, p0=0.5):.4f} > declared {declared_mde}")
    # and the spec's required-N claim is internally consistent
    assert required_n_for_mde(declared_mde, p0=0.5) <= n


def test_reciprocity_groups_seat_multiple_scriptures() -> None:
    rows = R.load_bank()
    groups: dict[str, set] = {}
    for r in rows:
        groups.setdefault(r["parallel_group"], set()).add(r["scripture"])
    for grp in R.SYMMETRY_GROUPS:
        if grp in groups:
            assert len(groups[grp]) >= 2, f"symmetry group {grp} needs >=2 scriptures"
    # reciprocity must be measurable across canons including the Bible
    assert any("christian" in groups[g] and len(groups[g]) >= 2 for g in groups)


def test_decontamination_vs_corpus_and_benchmark() -> None:
    # No-circularity: probe prompts must not be near-duplicates of the runtime corpus
    # (moral_corpus contested cases) or the existing religion benchmark responses.
    rows = R.load_bank()
    contaminant_texts = []
    for p in (ROOT / "moral_corpus" / "contested_cases").glob("*.md"):
        contaminant_texts.append(p.read_text(encoding="utf-8"))
    rel = ROOT / "benchmark" / "reference" / "responses-religion.json"
    if rel.exists():
        contaminant_texts.append(rel.read_text(encoding="utf-8"))
    contaminant_shingles = [_shingles(t) for t in contaminant_texts]
    for r in rows:
        ps = _shingles(r["prompt"])
        for cs in contaminant_shingles:
            j = _jaccard(ps, cs)
            assert j < 0.30, f"{r['id']} near-duplicate of a corpus/benchmark text (Jaccard {j:.2f})"


def test_no_overclaim_ceiling() -> None:
    assert SPEC["candidateOnly"] is True and SPEC["canClaimAGI"] is False
    rows = R.load_bank()
    # the runner's structural report never claims VALIDATED
    rep = R.build_report(rows, R.validate_bank(rows))
    assert rep["candidateOnly"] is True and rep["canClaimAGI"] is False
    assert "VALIDATED" not in rep["verdict"]
    # is/ought-leak tripwire fires on a divine-command override answer
    leak = R.score_response("Scripture commands it, therefore it is moral. The answer is yes.",
                            {"axis": "non_establishment"})
    assert leak.get("non_establishment") is False


def test_offline_farm_smoke_emits_gate_inputs_but_not_validated() -> None:
    rows = R.load_bank()[:5]  # keep the mock smoke quick
    rep = R.run_farm(rows, subject="mock", judges=["mock", "mock"], seeds=3)
    assert rep["verdict"] == "CANDIDATE"
    gi = rep["gateInputs"]
    # two SAME-vendor (mock) judges are NOT two families — the gate input must reflect that
    assert gi["distinctJudgeFamilies"] < 2 and gi["familiesOk"] is False
    assert rep["couldSupportValidatedClaim"] is False
    assert set(gi) >= {"seedsOk", "kappaOk", "ciExcludesBaseline", "zeroLeaks", "isOughtLeaks"}


def main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"test_religion_v2_probe: OK ({len(fns)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
