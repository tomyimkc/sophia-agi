# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Unit tests for the retrieval-augmented transition predictor (Cluster D2, the FLOOR).

Deterministic — no torch, no network, no keys. Locks the validated retrieval-grounding
posture:
  - an IN-DISTRIBUTION query (a near neighbour exists) predicts the correct outcome
    with high confidence;
  - a NOVEL (OOD) query — no stored trace clears the similarity floor — yields
    ood=True and ABSTAINS (prediction is None), it does NOT guess;
  - evaluate_split on the bundled contrastive demo pack yields val accuracy above 0.65
    AND shift-degradation <= 0.15: the retrieval floor generalizes (on answered
    queries) where the RSSM collapsed, while honestly abstaining on novel families.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.retrieval_transition_model import (  # noqa: E402
    RetrievalTransitionModel,
    evaluate_split,
)

DEMO = ROOT / "agi-proof" / "world-model" / "contrastive-traces-demo.json"


def _load_pack() -> dict:
    pack = json.loads(DEMO.read_text(encoding="utf-8"))
    return {
        "train": [tuple(t) for t in pack["train"]],
        "val": [tuple(t) for t in pack["val"]],
        "shift": [tuple(t) for t in pack["shift"]],
    }


def test_in_distribution_query_predicts_correctly_with_high_confidence() -> None:
    pack = _load_pack()
    model = RetrievalTransitionModel().fit(pack["train"])
    # An exact in-distribution (state, action) -> the stored success outcome, confident.
    out = model.predict("goal-fetch-url:s1:has-network", "http_get")
    assert out["ood"] is False
    assert out["prediction"] == 1
    assert out["confidence"] >= 0.6
    # The contrast: same action, failure-precondition state -> the failure outcome.
    out_fail = model.predict("goal-fetch-url:s1:no-network", "http_get")
    assert out_fail["ood"] is False
    assert out_fail["prediction"] == 0
    assert out_fail["confidence"] >= 0.6


def test_novel_ood_query_abstains() -> None:
    """A genuinely novel task-family (no stored trace clears the similarity floor) must
    be flagged OOD and ABSTAIN — prediction None, confidence 0 — not guessed."""
    pack = _load_pack()
    model = RetrievalTransitionModel().fit(pack["train"])
    for state, action in [
        ("goal-fold-protein:s1:energy-minimized", "relax_structure"),
        ("goal-negotiate-contract:s1:terms-agreed", "countersign"),
        ("totally-unrelated-task-xyz", "unheard_of_action"),
    ]:
        out = model.predict(state, action)
        assert out["ood"] is True, (state, action, out)
        assert out["prediction"] is None
        assert out["confidence"] == 0.0


def test_empty_corpus_abstains() -> None:
    model = RetrievalTransitionModel().fit([])
    out = model.predict("goal-fetch-url:s1:has-network", "http_get")
    assert out["ood"] is True
    assert out["prediction"] is None


def test_evaluate_split_generalizes_within_shift_bound() -> None:
    """The load-bearing canary metric: val accuracy > 0.65 AND shift-degradation <= 0.15.
    The retrieval floor holds where the DreamerV3 RSSM collapsed on the 25-pair corpus."""
    pack = _load_pack()
    res = evaluate_split(pack["train"], pack["val"], pack["shift"])
    assert res["valAccuracy"] > 0.65, res
    assert res["shiftDegradation"] <= 0.15, res
    # Honest: the shift split contains novel families, so it abstains on some queries.
    # The point is that accuracy on ANSWERED queries does not collapse under shift.
    assert res["shiftAbstainRate"] > 0.0, res
    assert res["valAnswered"] > 0 and res["shiftAnswered"] > 0, res


def test_abstentions_not_scored_as_errors() -> None:
    """Sanity: a split made entirely of novel OOD queries abstains on all of them, so
    accuracy is 0.0 over ZERO answered — abstention is an honest hold, never a guess."""
    pack = _load_pack()
    novel = [
        ("goal-fold-protein:s1:energy-minimized", "relax_structure", 0),
        ("goal-negotiate-contract:s1:terms-agreed", "countersign", 1),
    ]
    res = evaluate_split(pack["train"], novel, novel)
    assert res["valAnswered"] == 0
    assert res["valAbstainRate"] == 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
