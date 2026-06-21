#!/usr/bin/env python3
"""Tests for the Provenance Delta benchmark (provenance_bench/*). Offline.

Drives the runner with an injected ``generate`` callable and synthetic gate
records, so the alone-vs-gated delta is exercised deterministically with no API
calls. Also covers dataset build, the independent judge, scoring, and the report.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.model import ModelResult  # noqa: E402
from provenance_bench import dataset, report, score  # noqa: E402
from provenance_bench.dataset import Case  # noqa: E402
from provenance_bench.judge import judge_answer, lexical_judge, names_author  # noqa: E402
from provenance_bench.runner import run_case, run_cases  # noqa: E402

# Synthetic records so the gate fires without depending on the seeded corpus.
RECORDS = {"phoenix": {"canonicalTitleEn": "Project Phoenix Charter", "doNotAttributeTo": ["Alice"]}}

FALSE_CASE = Case(
    id="false-alice-phoenix", label="false", work="Project Phoenix Charter",
    gold_author="the founding committee", claimed_author="Alice",
    prompt="Did Alice write the Project Phoenix Charter? Answer yes or no.",
    source_url="x", reason="committee-authored",
)
TRUE_CASE = Case(
    id="true-committee-phoenix", label="true", work="Project Phoenix Charter",
    gold_author="the founding committee", claimed_author=None,
    prompt="Who wrote the Project Phoenix Charter?", source_url="x", reason="committee",
)


def _gen(*responses):
    """Fake generate returning each response in turn (sticky on the last)."""
    box = {"i": 0}

    def generate(system: str, user: str) -> ModelResult:
        idx = min(box["i"], len(responses) - 1)
        box["i"] += 1
        return ModelResult(text=responses[idx], provider="mock", model="t", ok=True)

    return generate


def _run_kw():
    return dict(records=RECORDS, retrieve_fn=lambda q, top_k=8: [], format_context_fn=lambda c: "(ctx)")


# --- dataset --------------------------------------------------------------- #

def test_dataset_builds_from_external_files() -> None:
    cases = dataset.build_cases()
    assert len(cases) >= 20
    labels = {c.label for c in cases}
    assert labels == {"true", "false"}
    # every case carries an external citation and a gold author
    assert all(c.source_url and c.gold_author for c in cases)
    false_cases = [c for c in cases if c.label == "false"]
    assert all(c.claimed_author for c in false_cases)


def test_gate_records_derived_from_misattributions() -> None:
    recs = dataset.build_gate_records()
    assert recs, "expected non-empty derived gate records"
    # every record carries a title and a non-empty do-not-attribute list
    assert all(r.get("canonicalTitleEn") and r.get("doNotAttributeTo") for r in recs.values())
    # a known hard case maps to a rule the gate can fire on
    flat = {a for r in recs.values() for a in r["doNotAttributeTo"]}
    assert "Aristotle" in flat


# --- judge (independent of the gate) --------------------------------------- #

def test_names_author_token_match() -> None:
    assert names_author("it was written by plato.", "Plato") is True
    assert names_author("the founding committee ratified it", "the founding committee") is True
    assert names_author("no idea", "Plato") is False


def test_judge_flags_false_affirmation() -> None:
    j = lexical_judge("Yes, Alice wrote the Project Phoenix Charter.", FALSE_CASE)
    assert j.hallucinated is True and j.abstained is False


def test_judge_passes_correction() -> None:
    j = lexical_judge("No. The founding committee wrote it; Alice did not.", FALSE_CASE)
    assert j.hallucinated is False and j.affirmed_gold is True


def test_judge_detects_abstention() -> None:
    j = lexical_judge("That authorship is disputed; I cannot confirm it.", FALSE_CASE)
    assert j.abstained is True and j.hallucinated is False


def test_judge_handles_scholarly_hedge() -> None:
    # appropriately-cautious answers on spurious attributions are NOT hallucinations
    for ans in (
        "It is traditionally attributed to Alice, but scholars consider it spurious.",
        "That work is pseudonymous; the real author is unknown.",
        "No, it is a later forgery, not actually by Alice.",
    ):
        assert lexical_judge(ans, FALSE_CASE).hallucinated is False


def test_judge_true_case_correctness() -> None:
    assert lexical_judge("The founding committee wrote it.", TRUE_CASE).affirmed_gold is True
    assert lexical_judge("It is unknown.", TRUE_CASE).abstained is True


def test_injected_llm_judge_overrides() -> None:
    from provenance_bench.judge import Judgment

    fake = lambda answer, case: Judgment(False, True, False, method="llm:test")
    assert judge_answer("anything", FALSE_CASE, llm_judge_fn=fake).method == "llm:test"


# --- runner (alone vs gated) ----------------------------------------------- #

def test_runner_gate_fixes_hallucination() -> None:
    # model always asserts the forbidden attribution; gate must remediate.
    gen = _gen("Yes, Alice wrote the Project Phoenix Charter.")
    res = run_case(FALSE_CASE, gen, on_fail="repair", **_run_kw())
    assert res["raw"]["hallucinated"] is True
    assert res["gated"]["hallucinated"] is False          # repair-exhausted -> cited abstention
    assert res["gated_action"] in ("abstained", "repaired")


def test_runner_passthrough_shows_no_remediation() -> None:
    gen = _gen("Yes, Alice wrote the Project Phoenix Charter.")
    res = run_case(FALSE_CASE, gen, on_fail="passthrough", **_run_kw())
    assert res["raw"]["hallucinated"] is True and res["gated"]["hallucinated"] is True


def test_runner_true_case_no_false_positive() -> None:
    gen = _gen("The founding committee wrote it.")
    res = run_case(TRUE_CASE, gen, on_fail="repair", **_run_kw())
    assert res["raw"]["affirmed_gold"] is True and res["gated"]["affirmed_gold"] is True


# --- scoring --------------------------------------------------------------- #

def test_score_metrics() -> None:
    gen_bad = _gen("Yes, Alice wrote the Project Phoenix Charter.")
    gen_good = _gen("The founding committee wrote it.")
    results = [
        run_case(FALSE_CASE, gen_bad, on_fail="repair", **_run_kw()),
        run_case(TRUE_CASE, gen_good, on_fail="repair", **_run_kw()),
    ]
    s = score.score(results)
    assert s["hallucinationRateAlone"] == 1.0     # the one false case hallucinated alone
    assert s["hallucinationRateGated"] == 0.0     # gate fixed it
    assert s["delta"] == 1.0
    assert s["falsePositiveCost"] == 0.0          # true case unharmed
    assert s["coverageRecall"] == 1.0             # the one bad-alone case was fixed


# --- report ---------------------------------------------------------------- #

def test_report_build_and_markdown() -> None:
    per_model = {"mock": {"scores": score.score(
        run_cases([FALSE_CASE, TRUE_CASE], _gen("Yes, Alice wrote the Project Phoenix Charter."),
                  on_fail="passthrough", **_run_kw())
    ), "model": "mock", "onFail": "passthrough", "judgeMethod": "lexical"}}
    rpt = report.build_report(per_model, run_at="2026-06-21T00:00:00+00:00")
    assert rpt["benchmark"] == "provenance-delta"
    assert rpt["rows"][0]["model"] == "mock"
    md = report.to_markdown(rpt)
    assert "Provenance Delta" in md and "| mock |" in md


def main() -> int:
    test_dataset_builds_from_external_files()
    test_gate_records_derived_from_misattributions()
    test_names_author_token_match()
    test_judge_flags_false_affirmation()
    test_judge_passes_correction()
    test_judge_detects_abstention()
    test_judge_handles_scholarly_hedge()
    test_judge_true_case_correctness()
    test_injected_llm_judge_overrides()
    test_runner_gate_fixes_hallucination()
    test_runner_passthrough_shows_no_remediation()
    test_runner_true_case_no_false_positive()
    test_score_metrics()
    test_report_build_and_markdown()
    print("test_provenance_bench: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
