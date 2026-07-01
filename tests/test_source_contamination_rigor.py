# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline smoke for the source-contamination RIGOR pass (source-contamination-live-multifamily-rigor).

Deterministic — no network, no API keys. Exercises the multi-run aggregation + bootstrap CI
plumbing that the live >=3-run answer!=judge protocol depends on, and locks the honest framing:

  (a) run_multi(runs=3) returns per_run + a bootstrap CI for both headline rates;
  (b) --fake is CI plumbing ONLY — the pre-registration and PENDING report both say so, and the
      PENDING artifact must stay status:not_run (a --fake run is never a capability result);
  (c) the pre-registration pins answer!=judge and an open-world --retrieve arm as GO/NO-GO.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_source_contamination_bench import (  # noqa: E402
    load_pack,
    make_fake_entailment,
    run_multi,
)

SPEC = ROOT / "agi-proof" / "source-verifier" / "measurement_spec.json"
PENDING = ROOT / "agi-proof" / "source-verifier" / "rigor-multifamily.PENDING.public-report.json"
_PACK = load_pack()


def _fake_factories():
    def entail_factory(case):
        return make_fake_entailment(case["false_token"], case["true_token"])

    def complete_factory(case):
        answer = case["fake_answer"]
        def C(system, user, *, max_tokens=180):  # noqa: ARG001
            return answer
        return C

    return entail_factory, complete_factory


def test_run_multi_emits_per_run_and_ci() -> None:
    """runs=3 -> 3 per-run entries + a bootstrap CI for both headline rates."""
    ef, cf = _fake_factories()
    agg = run_multi(_PACK, ef, cf, runs=3)
    assert agg["runs"] == 3
    assert len(agg["per_run"]) == 3
    for metric in ("contamination_caught_rate", "clean_over_blocked_rate"):
        ci = agg["ci"][metric]
        assert {"mean", "lo", "hi"} <= set(ci), ci
        assert ci["lo"] <= ci["mean"] <= ci["hi"]
    # --fake plumbing: fails closed without destroying recall.
    assert agg["ci"]["contamination_caught_rate"]["mean"] == 1.0
    assert agg["ci"]["clean_over_blocked_rate"]["mean"] == 0.0


def test_preregistration_pins_the_rigor_protocol() -> None:
    """The spec must require >=3 runs, answer!=judge, and an open-world --retrieve arm."""
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    assert spec["candidateOnly"] is True and spec["level3Evidence"] is False
    proto = spec["protocol"]
    assert proto["runsPerFamily"] >= 3
    assert proto["answerJudgeSeparated"] is True
    assert "retrieve" in proto["openWorldRefs"].lower()
    assert spec["guardrails"]["answerNeJudge"]["use"] == "GO/NO-GO"


def test_pending_artifact_is_not_run_and_candidate_only() -> None:
    """The rigor result stays PENDING/not_run until a real answer!=judge run exists."""
    rep = json.loads(PENDING.read_text(encoding="utf-8"))
    assert rep["status"] == "not_run"
    assert rep["go"] is False and rep["canClaimAGI"] is False
    assert rep["results"] is None and rep["ci"] is None
    assert rep["ledgerRef"] == "source-contamination-live-multifamily-2026-06-28"
