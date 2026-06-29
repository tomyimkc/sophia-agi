#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the retrieve-then-reason rollout harness + its run_rlvr wiring. Offline.

The harness produces the trajectory the retrieval-faithfulness reward scores; the
load-bearing invariant is that a retrieval-USING policy outscores a weights-LEAKING
one on an IDENTICAL answer (the counterfactual citation-drop signal). The live
rollout-driven GRPO loop is NOT tested here — it stays Open until a gated run.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import faithfulness_rollout as fr  # noqa: E402
from provenance_bench.retrieval_faithfulness import reward_for_trajectory  # noqa: E402

_CASE = {"prompt": "Who wrote the Project Phoenix Charter?",
         "should_retrieve": True, "answerable": True, "gold": "founding committee"}
_COMMON = dict(retrieve=fr._mock_retrieve, extract_claims=fr._mock_extract,
               verify_claim=fr._mock_verify, check_correct=fr._check_correct)


def test_offline_invariants_pass() -> None:
    ok, detail = fr.offline_invariants()
    assert ok, detail
    assert all(detail["checks"].values())


def test_faithful_claim_flips_leaky_survives() -> None:
    """The whole point: identical answers, but the claim must DEPEND on retrieval
    for the faithful policy (flips when c1 dropped) and not for the leaky one."""
    t_faithful = fr.rollout(_CASE, generate=fr._faithful_policy, **_COMMON)
    t_leaky = fr.rollout(_CASE, generate=fr._leaky_policy, **_COMMON)
    assert t_faithful["answer_text"] == t_leaky["answer_text"]  # same words...
    assert t_faithful["claims"][0]["survives_ablation"] is False  # ...different faithfulness
    assert t_leaky["claims"][0]["survives_ablation"] is True
    assert reward_for_trajectory(t_faithful)[0] > reward_for_trajectory(t_leaky)[0]


def test_rollout_is_deterministic() -> None:
    a = fr.rollout(_CASE, generate=fr._faithful_policy, **_COMMON)
    b = fr.rollout(_CASE, generate=fr._faithful_policy, **_COMMON)
    assert reward_for_trajectory(a)[0] == reward_for_trajectory(b)[0]


def test_abstain_short_circuits_with_no_claims() -> None:
    t = fr.rollout({**_CASE, "answerable": False}, generate=lambda q, c: fr.ABSTAIN, **_COMMON)
    assert t["abstained"] is True
    assert t["claims"] == []


def test_run_rlvr_faithfulness_mock_passes(tmp_path=None) -> None:
    """The task rides the same offline CI lane as the other RLVR rewards. Writes the
    report to a throwaway path so the committed artifact never drifts from a test run."""
    import tempfile

    from tools import run_rlvr

    out = Path(tmp_path) / "rlvr.json" if tmp_path else Path(tempfile.mkdtemp()) / "rlvr.json"
    rc = run_rlvr.main(["--task", "faithfulness", "--model", "mock", "--out", str(out)])
    assert rc == 0
    assert out.exists()


def main() -> int:
    test_offline_invariants_pass()
    test_faithful_claim_flips_leaky_survives()
    test_rollout_is_deterministic()
    test_abstain_short_circuits_with_no_claims()
    test_run_rlvr_faithfulness_mock_passes()
    print("test_faithfulness_rollout: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
