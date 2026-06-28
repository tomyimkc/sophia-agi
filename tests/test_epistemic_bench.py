"""CI test for the epistemic benchmark scorer (Phase 0).

Verifies the deterministic scoring seam without any model or GPU: a disciplined
grounded/abstaining agent must dominate a fabricating one on the axes the bench
is built to measure (E1 grounding, E2 abstention, E5 deception robustness).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.epistemic_bench import score  # noqa: E402


def test_self_test_disciplined_dominates_fabricator():
    out = score.self_test()
    good, bad = out["good"], out["bad"]
    # E1: fewer hallucinated attributions.
    assert good["E1_hallucinated_attribution"]["rate"] < bad["E1_hallucinated_attribution"]["rate"]
    # E2: abstains on the unanswerable split where the fabricator does not.
    assert good["E2_abstain_recall"]["rate"] > bad["E2_abstain_recall"]["rate"]
    # E5: robust under the adversarial (fabricated-source) split.
    assert good["E5_adversarial_abstain_recall"]["rate"] > bad["E5_adversarial_abstain_recall"]["rate"]


def test_sealed_dataset_loads_and_scores():
    cases_path = _REPO_ROOT / "eval" / "epistemic_bench" / "data" / "cases.jsonl"
    cases = score.load_cases(cases_path)
    assert len(cases) >= 7
    assert {c["split"] for c in cases.values()} == {"answerable", "unanswerable", "adversarial"}

    # A disciplined completion for every case (abstain where appropriate, ground otherwise).
    comps = {}
    for cid, c in cases.items():
        if c["split"] == "answerable":
            comps[cid] = {"id": cid, "completion": f"The attributed figure is {c.get('gold', 'recorded in the register')}.", "confidence": 0.8, "correct": True}
        else:
            comps[cid] = {"id": cid, "completion": "I can't verify that against a source, so I won't assert it.", "confidence": 0.2, "correct": True}
    scores = [score.score_case(cases[cid], comps[cid]) for cid in cases]
    report = score.aggregate(scores)
    assert report["n"] == len(cases)
    # Disciplined agent: full abstain-recall on the unanswerable split.
    assert report["E2_abstain_recall"]["rate"] == 1.0
    # No hallucinated attributions when the agent stays in the register / abstains.
    assert report["E1_hallucinated_attribution"]["rate"] == 0.0


def test_axis_is_none_when_inputs_absent():
    # A case with no sources => E4 must be N/A (None), never a guessed pass.
    case = {"id": "x", "split": "answerable"}
    comp = {"id": "x", "completion": "Some grounded claim."}
    s = score.score_case(case, comp)
    assert s["E4_citation_faithful"] is None


if __name__ == "__main__":
    test_self_test_disciplined_dominates_fabricator()
    test_sealed_dataset_loads_and_scores()
    test_axis_is_none_when_inputs_absent()
    print("all epistemic_bench tests passed")
