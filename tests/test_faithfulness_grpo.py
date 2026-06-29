#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the rollout-driven GRPO loop + live seam adapters. Offline, no torch.

The load-bearing property: on an all-CORRECT group, a correctness-only reward
collapses (zero within-group variance, zero advantage), but the faithfulness reward
still separates retrieval-using from weights-leaking rollouts — a non-zero learning
signal. The live gradient step stays Open; only the deterministic loop core + the
live-seam conformance are tested here.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import faithfulness_grpo as fg  # noqa: E402
from provenance_bench import faithfulness_seams as fs  # noqa: E402


# --- GRPO advantage math --------------------------------------------------- #


def test_group_advantages_sum_to_zero_and_normalize() -> None:
    adv = fg.group_advantages([0.0, 0.5, 1.0])
    assert abs(sum(adv)) < 1e-9
    assert adv[0] < 0 < adv[2]


def test_collapsed_group_has_zero_advantage() -> None:
    """Identical rewards -> zero advantage -> no learning signal (the collapse)."""
    assert fg.group_advantages([0.7, 0.7, 0.7]) == [0.0, 0.0, 0.0]
    assert fg.within_group_std([0.7, 0.7, 0.7]) == 0.0


# --- the anti-collapse property -------------------------------------------- #


def test_offline_invariants_pass() -> None:
    ok, detail = fg.offline_invariants()
    assert ok, detail
    assert all(detail["checks"].values())


def test_faithfulness_signal_where_correctness_collapses() -> None:
    """Explicitly: correctness-only std == 0 while faithfulness std > 0 on the same
    all-correct group."""
    _, detail = fg.offline_invariants()
    assert detail["correctnessOnlyStd"] == 0.0
    assert detail["withinGroupStd"] > 0.0


# --- live seam conformance ------------------------------------------------- #


def test_seam_conformance_passes() -> None:
    ok, detail = fs.conformance_check()
    assert ok, detail
    assert all(detail["checks"].values())


def test_ai_search_retrieve_returns_rollout_shaped_chunks() -> None:
    retrieve = fs.make_ai_search_retrieve(top_k=3)
    chunks = retrieve("Who wrote the Dao De Jing?")
    for c in chunks:
        assert {"chunk_id", "text", "author_confidence"} <= set(c)
        assert isinstance(c["chunk_id"], str) and c["author_confidence"]


def test_entailment_verify_maps_verdicts() -> None:
    verify = fs.make_entailment_verify(fs._mock_entailment)
    ctx = [{"chunk_id": "c1", "text": "Laozi wrote the Dao De Jing."}]
    assert verify({"text": "Laozi wrote it", "support_chunk_ids": []}, ctx) == "supported"
    assert verify({"text": "Plato wrote it", "support_chunk_ids": []}, ctx) == "unsupported"


def main() -> int:
    test_group_advantages_sum_to_zero_and_normalize()
    test_collapsed_group_has_zero_advantage()
    test_offline_invariants_pass()
    test_faithfulness_signal_where_correctness_collapses()
    test_seam_conformance_passes()
    test_ai_search_retrieve_returns_rollout_shaped_chunks()
    test_entailment_verify_maps_verdicts()
    print("test_faithfulness_grpo: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
