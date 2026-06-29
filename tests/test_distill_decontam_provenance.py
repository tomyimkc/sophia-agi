#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for T4 (decontaminate distilled traces vs held-out eval) + T8 (verification
provenance tagging). Offline via the mock teacher and an injected eval set."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import model as m  # noqa: E402
from provenance_bench import dataset_guard as dg  # noqa: E402
from tools import distill_export as d  # noqa: E402
from tools import build_local_sophia_dataset as b  # noqa: E402


def _mock_client():
    os.environ.pop("SOPHIA_MOCK_RESPONSE", None)
    return m.ModelClient(m.resolve_config("mock"))


def test_kept_rows_are_provenance_tagged() -> None:
    # The mock answer passes the gate; mustInclude "Decision" is present -> accepted & kept.
    data = d.distill([{"id": "g", "prompt": "give a decision", "mustInclude": ["Decision"]}],
                     _mock_client(), decontam=False)
    assert data["accepted"] == 1
    meta = data["sft"][0]["metadata"]
    assert meta["verification_provenance"] == d.PROV_PASSED
    assert meta["source"] == "distillation"


def test_exact_eval_collision_is_decontaminated() -> None:
    # Inject an eval set containing this exact prompt -> the kept trace must be diverted to
    # the decontaminated bucket, NOT the SFT bucket (no eval leak into training).
    leaked = "give a decision"
    data = d.distill([{"id": "g", "prompt": leaked, "mustInclude": ["Decision"]}],
                     _mock_client(), decontam=True, eval_prompts={dg.normalize(leaked)})
    assert data["accepted"] == 0
    assert data["decontaminated"] == 1
    assert data["decontaminatedRows"][0]["reason"]["kind"] == "exact"


def test_near_dup_eval_collision_is_decontaminated() -> None:
    eval_prompt = "what is the capital city of the country called france today"
    train_prompt = "what is the capital city of the country called france today please"
    data = d.distill([{"id": "n", "prompt": train_prompt, "mustInclude": ["Decision"]}],
                     _mock_client(), decontam=True, jaccard_thr=0.5,
                     eval_prompts={dg.normalize(eval_prompt)})
    assert data["decontaminated"] == 1
    assert data["decontaminatedRows"][0]["reason"]["kind"] == "near-dup"


def test_cost_per_verified_row_reported() -> None:
    data = d.distill([{"id": "g", "prompt": "give a decision", "mustInclude": ["Decision"]}],
                     _mock_client(), decontam=False)
    # mock teacher is free, so cost is 0.0 (not None) once at least one row is kept.
    assert data["costPerVerifiedRow"] == 0.0


def test_shingle_helpers_roundtrip() -> None:
    a = dg.shingles("the quick brown fox jumps", k=2)
    b_ = dg.shingles("the quick brown fox jumps", k=2)
    assert a == b_ and dg.jaccard(a, b_) == 1.0
    assert dg.jaccard(a, set()) == 0.0


def test_oversample_hard_repeats_only_tagged_rows() -> None:
    rows = [
        {"messages": [], "metadata": {"verification_provenance": "patched_after_failure"}},
        {"messages": [], "metadata": {"verification_provenance": "passed_first_try"}},
    ]
    out, extra = b._oversample_hard(rows, factor=3)
    # the patched row is tripled (2 extra), the passed row is untouched.
    assert extra == 2
    assert len(out) == 4
    out0, _ = b._oversample_hard(rows, factor=1)
    assert len(out0) == 2  # factor 1 is a no-op (no drift)


def main() -> int:
    test_kept_rows_are_provenance_tagged()
    test_exact_eval_collision_is_decontaminated()
    test_near_dup_eval_collision_is_decontaminated()
    test_cost_per_verified_row_reported()
    test_shingle_helpers_roundtrip()
    test_oversample_hard_repeats_only_tagged_rows()
    print("test_distill_decontam_provenance: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
