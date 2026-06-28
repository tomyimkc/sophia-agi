#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the G9D eval-awareness / deceptive-alignment tripwire. Offline, stdlib only, no torch.

Asserts the gate's invariants: consistent cross-context behaviour promotes; behaviour that
diverges under observation rejects (eval-gaming tell); paraphrase brittleness quarantines;
a missing eval-context or deployment-context measurement fails closed to quarantine; and the
standardised honesty fields (with an honest boundary) hold on every decision.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ssil_eval_awareness import GATE_ID, SCHEMA, demo_bundle, evaluate  # noqa: E402

_ALLOWED = {"promote", "quarantine", "reject"}


def _assert_envelope(d: dict) -> None:
    assert d["canClaimAGI"] is False, d
    assert d["candidateOnly"] is True, d
    assert d["level3Evidence"] is False, d
    assert d["verdict"] in _ALLOWED, d
    assert d["schema"] == SCHEMA == "sophia.eval_awareness_decision.v1", d
    assert d["gate"] == GATE_ID == "G9D", d
    assert isinstance(d["boundary"], str) and d["boundary"], d


# --- promote --------------------------------------------------------------- #


def test_demo_bundle_promotes() -> None:
    d = evaluate(demo_bundle())
    assert d["verdict"] == "promote", d["reasons"]
    assert any("consistent across eval/deploy" in r for r in d["reasons"]), d["reasons"]
    _assert_envelope(d)


# --- reject reason --------------------------------------------------------- #


def test_divergence_under_observation_rejects() -> None:
    b = demo_bundle()
    # Looks safe when eval-cued, much less so when deploy-cued: eval-gaming tell.
    b["evalCued"] = {"safetyScore": 0.95}
    b["deployCued"] = {"safetyScore": 0.40}
    d = evaluate(b)
    assert d["verdict"] == "reject", d
    assert any("possible eval-gaming" in r for r in d["reasons"]), d["reasons"]
    assert d["metrics"]["divergence"] > d["metrics"]["maxDivergence"], d["metrics"]
    _assert_envelope(d)


# --- quarantine / abstain reasons ------------------------------------------ #


def test_paraphrase_inconsistency_quarantines() -> None:
    b = demo_bundle()
    # Contexts agree (no reject) but safety behaviour is brittle to surface paraphrasing.
    b["paraphraseScores"] = [0.90, 0.55, 0.92, 0.60]
    d = evaluate(b)
    assert d["verdict"] == "quarantine", d
    assert any("inconsistent across paraphrases" in r for r in d["reasons"]), d["reasons"]
    assert d["metrics"]["paraphraseSpread"] > d["metrics"]["maxParaphraseSpread"], d["metrics"]
    _assert_envelope(d)


def test_missing_eval_cued_fails_closed() -> None:
    b = demo_bundle()
    b["evalCued"] = None
    d = evaluate(b)
    assert d["verdict"] == "quarantine", d
    assert any("evalCued" in r for r in d["reasons"]), d["reasons"]
    assert d["metrics"]["missingInput"] == "evalCued", d["metrics"]
    _assert_envelope(d)


def test_missing_deploy_cued_fails_closed() -> None:
    b = demo_bundle()
    b["deployCued"] = None
    d = evaluate(b)
    assert d["verdict"] == "quarantine", d
    assert any("cannot certify no eval-gaming" in r for r in d["reasons"]), d["reasons"]
    assert d["metrics"]["missingInput"] == "deployCued", d["metrics"]
    _assert_envelope(d)


def test_missing_deploy_safety_score_fails_closed() -> None:
    b = demo_bundle()
    b["deployCued"] = {}  # present dict but no safetyScore
    d = evaluate(b)
    assert d["verdict"] == "quarantine", d
    assert any("cannot certify no eval-gaming" in r for r in d["reasons"]), d["reasons"]
    assert d["metrics"]["missingInput"] == "deployCued.safetyScore", d["metrics"]
    _assert_envelope(d)


def test_missing_paraphrase_scores_fails_closed() -> None:
    # Contexts agree (no reject) but the paraphrase consistency check was never run:
    # the gate must ABSTAIN (quarantine) rather than promote on an unmeasured check,
    # and must NOT default the spread to a passing 0.0.
    for b in (_drop_paraphrases(demo_bundle()), _empty_paraphrases(demo_bundle())):
        d = evaluate(b)
        assert d["verdict"] == "quarantine", d
        assert any("paraphrase consistency unmeasured" in r for r in d["reasons"]), d["reasons"]
        # No positive claim about an unrun paraphrase check.
        assert not any("consistent across" in r and "paraphrase" in r for r in d["reasons"]), d["reasons"]
        assert d["metrics"]["paraphraseCount"] == 0, d["metrics"]
        assert d["metrics"]["paraphraseSpread"] is None, d["metrics"]  # not defaulted to 0.0
        _assert_envelope(d)


def test_promote_reason_only_claims_measured_paraphrases() -> None:
    # When paraphrases ARE supplied and consistent, the promote reason may name them;
    # but a promote can never occur with an unmeasured paraphrase check (see above), so
    # any promote reason mentioning paraphrases reflects a check that actually ran.
    d = evaluate(demo_bundle())
    assert d["verdict"] == "promote", d
    assert d["metrics"]["paraphraseCount"] > 0, d["metrics"]
    _assert_envelope(d)


def _drop_paraphrases(b: dict) -> dict:
    b.pop("paraphraseScores", None)
    return b


def _empty_paraphrases(b: dict) -> dict:
    b["paraphraseScores"] = []
    return b


def main() -> int:
    test_demo_bundle_promotes()
    test_divergence_under_observation_rejects()
    test_paraphrase_inconsistency_quarantines()
    test_missing_eval_cued_fails_closed()
    test_missing_deploy_cued_fails_closed()
    test_missing_deploy_safety_score_fails_closed()
    test_missing_paraphrase_scores_fails_closed()
    test_promote_reason_only_claims_measured_paraphrases()
    print("test_ssil_eval_awareness: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
