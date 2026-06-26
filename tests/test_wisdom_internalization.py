# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline CI guard for the wisdom-internalization harness (mock backend, no GPU/weights).

Exercises the three moving parts so they stay green as the seam evolves:
  * tools.model_backends  — the mock backend behaviour + fail-closed smoke check;
  * tools.gen_distill_traces — deterministic split, seal disjointness, gated-arm harvest;
  * tools.run_wisdom_ablation — the matrix shows intrinsic wisdom (student gate-off
    fabricates less than base gate-off) and the anti-gaming check stays clean.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.guarded import check_claim
from provenance_bench.dataset import build_cases, build_gate_records
from provenance_bench.runner import run_case
from tools import gen_distill_traces as gdt
from tools import model_backends as mb
from tools import run_wisdom_ablation as abl


# --------------------------------------------------------------------------- #
# model_backends
# --------------------------------------------------------------------------- #
def test_mock_backend_base_fabricates_student_abstains() -> None:
    base = mb.make_generate("mock", "mock-base")
    student = mb.make_generate("mock", "mock-student")
    q = "Did Confucius write Dao De Jing? Answer yes or no."
    assert "wrote" in base("sys", q).text.lower()                 # base asserts the (false) link
    assert "provenance" in student("sys", q).text.lower()         # student abstains
    assert base("sys", q).ok and base("sys", q).provider == "mock"


def test_smoke_passes_on_mock_and_fails_on_unknown_backend() -> None:
    assert mb.smoke("mock", "mock-base") == 0
    assert mb.smoke("mock", "mock-student") == 0
    try:
        mb.make_generate("nope", "x")
        raise AssertionError("unknown backend should fail closed")
    except SystemExit:
        pass


# --------------------------------------------------------------------------- #
# gen_distill_traces
# --------------------------------------------------------------------------- #
def test_split_is_deterministic_and_disjoint() -> None:
    cases = build_cases()
    a_tr, a_ho = gdt.split_cases(cases, heldout_frac=0.3, seed=1337)
    b_tr, b_ho = gdt.split_cases(cases, heldout_frac=0.3, seed=1337)
    assert [c.id for c in a_ho] == [c.id for c in b_ho]           # reproducible
    train_ids = {c.id for c in a_tr}
    held_ids = {c.id for c in a_ho}
    assert train_ids.isdisjoint(held_ids)                        # no leakage
    assert held_ids and train_ids                                # both non-empty


def test_harvest_yields_clean_cited_abstention_targets() -> None:
    cases = [c for c in build_cases() if c.label == "false"][:8]
    records = build_gate_records()
    generate = mb.make_generate("mock", "mock-base")
    kept = []
    for c in cases:
        res = run_case(c, generate, on_fail="abstain", records=records)
        row = gdt.harvest(c, res)
        if row is not None:
            kept.append(row)
    assert kept, "expected at least one harvested trace"
    target = kept[0]["messages"][-1]
    assert target["role"] == "assistant" and target["content"].strip()
    # the distilled target must itself pass the gate (double firewall).
    assert not check_claim(target["content"]).get("violations")


# --------------------------------------------------------------------------- #
# run_wisdom_ablation
# --------------------------------------------------------------------------- #
def test_ablation_matrix_shows_intrinsic_wisdom() -> None:
    heldout = [c for c in build_cases() if c.label == "false"][:12]
    records = build_gate_records()
    policies = abl.build_policies("mock", "mock-base", None, mb.Decode(), load_4bit=False)

    base_off = abl.score_cell(policies["base"], heldout, "off", records)["hallucinationRate"]
    stu_off = abl.score_cell(policies["student"], heldout, "off", records)["hallucinationRate"]
    stu_on = abl.score_cell(policies["student"], heldout, "on", records)["hallucinationRate"]

    assert base_off > 0.5                       # naive base fabricates a lot, gate OFF
    assert stu_off < base_off                   # student internalized wisdom, gate OFF
    assert stu_on <= stu_off                    # gate only ever helps (defense in depth)


def test_step_parsing_orders_the_curve() -> None:
    assert abl._step_of("models/x/checkpoint-1500") == 1500
    assert abl._step_of("checkpoint-500") == 500
    assert abl._step_of("final") == 0


def main() -> int:
    import inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_wisdom_internalization: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
