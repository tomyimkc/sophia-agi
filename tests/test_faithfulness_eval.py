#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the held-out faithfulness eval instrument. Offline, no torch/network.

Validates the instrument, not a model: the counterfactual grounding rate must be 1.0
for a retrieval-using policy and 0.0 for a weights-leaking one on the mock world, the
paired contrast must favour the faithful policy, and every rate carries a CI. The
base-vs-adapter capability claim is NOT made here — it needs a trained adapter + a
pre-registered, powered, multi-family gated run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import faithfulness_eval as fe  # noqa: E402
from provenance_bench import faithfulness_rollout as fr  # noqa: E402

_CASES = [{"prompt": "Who wrote the Project Phoenix Charter?", "should_retrieve": True,
           "answerable": True, "gold": "founding committee"}] * 6
_SEAMS = dict(retrieve=fr._mock_retrieve, extract_claims=fr._mock_extract,
              verify_claim=fr._mock_verify, check_correct=fr._check_correct)


def test_offline_invariants_pass() -> None:
    ok, detail = fe.offline_invariants()
    assert ok, detail
    assert all(detail["checks"].values())


def test_rate_is_one_for_faithful_zero_for_leaky() -> None:
    f = fe.evaluate(_CASES, generate=fr._faithful_policy, **_SEAMS)["aggregate"]
    l = fe.evaluate(_CASES, generate=fr._leaky_policy, **_SEAMS)["aggregate"]
    assert f["rate"] == 1.0 and l["rate"] == 0.0
    assert f["knowledgeClaims"] == 6 and f["groundedClaims"] == 6
    assert f["fixedNCI95"] is not None


def test_paired_contrast_favours_faithful() -> None:
    c = fe.compare(_CASES, base_generate=fr._leaky_policy,
                   adapter_generate=fr._faithful_policy, **_SEAMS)
    assert c["meanDiff"] == 1.0
    assert c["nPaired"] == 6
    assert c["pairedBootstrapCI95"] is not None


def test_case_grounding_counts_knowledge_claims_only() -> None:
    traj = {"claims": [
        {"kind": "knowledge", "verdict": "supported", "survives_ablation": False},
        {"kind": "knowledge", "verdict": "supported", "survives_ablation": True},   # leaked
        {"kind": "commonsense", "verdict": "unsupported"},                          # excluded
    ]}
    grounded, total = fe.case_grounding(traj)
    assert (grounded, total) == (1, 2)


def test_abstention_not_counted_as_ungrounded() -> None:
    trajs = [{"abstained": True, "claims": []},
             {"abstained": False, "claims": [
                 {"kind": "knowledge", "verdict": "supported", "survives_ablation": False}]}]
    agg = fe.counterfactual_grounding_rate(trajs)
    assert agg["rate"] == 1.0            # the one claim made is grounded
    assert agg["abstentionRate"] == 0.5


def test_prereg_spec_is_valid_and_fail_closed() -> None:
    spec = json.loads((ROOT / "agi-proof" / "benchmark-results" / "faithfulness"
                       / "measurement_spec.json").read_text(encoding="utf-8"))
    assert spec["primaryMetric"] == "counterfactual_grounding_rate"
    assert "canClaimAGI:false" in spec["claimCeiling"]
    assert spec["requiredN"] > 0 and spec["mde"] > 0


def test_runner_lexical_entailment_and_compare_guard() -> None:
    """The runner's lexical entailment option resolves offline, and --compare without an
    hf:/--adapter policy fails closed (no silent wrong-policy eval)."""
    import types

    from tools import eval_faithfulness as ef
    from provenance_bench import faithfulness_seams as fs

    assert ef._build_entailment("lexical", None) is fs.lexical_entailment

    bad = types.SimpleNamespace(policy="llmhub:gemini", adapter=None, compare=True,
                                entailment="lexical", entailment_model=None, top_k=6,
                                seed=0, limit=4, out=ROOT / "x.json")
    try:
        ef._run_compare(bad)
    except SystemExit:
        pass
    else:
        raise AssertionError("_run_compare must reject a non-hf policy / missing adapter")


def main() -> int:
    test_offline_invariants_pass()
    test_runner_lexical_entailment_and_compare_guard()
    test_rate_is_one_for_faithful_zero_for_leaky()
    test_paired_contrast_favours_faithful()
    test_case_grounding_counts_knowledge_claims_only()
    test_abstention_not_counted_as_ungrounded()
    test_prereg_spec_is_valid_and_fail_closed()
    print("test_faithfulness_eval: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
