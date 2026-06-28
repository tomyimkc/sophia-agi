# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""REC 2 hardening tests for the source-contamination bench.

Deterministic — no network, no keys, no torch (fetch_fn mocked, entailment is a pure
function). Locks the three hardening properties added on top of the live 97.7% result
(THEORY-ISSUES-FROM-LIVE-RUNS-2026-06-28 issues 3 & 5):

  1. MULTI-RUN + CIs: ``bootstrap_ci`` returns a (lo, hi) that brackets the sample mean.
  2. ANSWER != JUDGE: ``build_report`` records both specs and flags separation only when
     they differ.
  3. OPEN-WORLD RETRIEVAL: ``--retrieve`` uses the FETCHED refs (independence measured),
     and FAILS CLOSED (abstains, ``retrieval_status == "empty"``) when the fetch returns
     nothing — an empty retrieval is never mistaken for verification.

Also asserts the pack now carries an ``entity`` on every case (the retrieval key).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_source_contamination_bench import (  # noqa: E402
    bootstrap_ci,
    build_report,
    load_pack,
    resolve_truth_refs,
    run_case,
)

_PACK = load_pack()


# --------------------------------------------------------------------------- #
# Pack now carries an entity (the Wikipedia retrieval key) on every case.
# --------------------------------------------------------------------------- #
def test_every_case_has_entity() -> None:
    cases = _PACK["cases"]
    missing = [c["id"] for c in cases if not (c.get("entity") or "").strip()]
    assert not missing, f"cases missing entity: {missing}"


# --------------------------------------------------------------------------- #
# 1. Bootstrap CI brackets the mean.
# --------------------------------------------------------------------------- #
def test_bootstrap_ci_brackets_mean_on_toy_sample() -> None:
    sample = [0.9, 1.0, 0.95, 1.0, 0.85]
    ci = bootstrap_ci(sample)
    expected_mean = sum(sample) / len(sample)
    assert abs(ci["mean"] - round(expected_mean, 4)) < 1e-9, ci
    assert ci["lo"] <= ci["mean"] <= ci["hi"], ci
    assert ci["n"] == 5
    # A non-degenerate sample yields a non-trivial interval below the max.
    assert ci["lo"] < 1.0


def test_bootstrap_ci_single_value_is_zero_width() -> None:
    ci = bootstrap_ci([0.977])
    assert ci["lo"] == ci["mean"] == ci["hi"] == 0.977, ci


def test_bootstrap_ci_empty_is_zeroed() -> None:
    ci = bootstrap_ci([])
    assert ci == {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0, "resamples": 0}


# --------------------------------------------------------------------------- #
# 2. ANSWER != JUDGE separation is recorded.
# --------------------------------------------------------------------------- #
def test_report_records_separated_answer_and_judge_specs() -> None:
    rep = build_report("relay", _PACK, None, status="ok_relay",
                       answer_spec="glm:glm-4.6", judge_spec="deepseek:deepseek-chat")
    assert rep["answer_spec"] == "glm:glm-4.6"
    assert rep["judge_spec"] == "deepseek:deepseek-chat"
    assert rep["answer_judge_separated"] is True


def test_report_not_separated_when_specs_match() -> None:
    rep = build_report("relay", _PACK, None, status="ok_relay",
                       answer_spec="glm:glm-4.6", judge_spec="glm:glm-4.6")
    assert rep["answer_judge_separated"] is False


def test_report_records_retrieval_mode() -> None:
    assert build_report("relay", _PACK, None, status="ok_relay")["retrieval_mode"] == "curated"
    assert build_report("relay", _PACK, None, status="ok_relay",
                        retrieve=True)["retrieval_mode"] == "wikipedia_rest_summary"


# --------------------------------------------------------------------------- #
# 3. OPEN-WORLD RETRIEVAL: uses fetched refs; fails closed on empty.
# --------------------------------------------------------------------------- #
# A contaminated case: the source attributes the Voynich manuscript to Ascham (false).
_CASE = next(c for c in _PACK["cases"] if c["id"] == "sc_001")


def _real_text_entail(claim: str, source: str) -> str:
    """Grade the claim against the ACTUAL ref text (as a real NLI/model would), so the
    verdict depends on what the RETRIEVED ref says — not on curated case tokens. A claim
    naming 'ascham' is contradicted by a ref saying the author is unknown."""
    c, s = claim.lower(), source.lower()
    claim_ascham = "ascham" in c
    ref_unknown = any(w in s for w in ("unknown", "unidentified", "no author"))
    if claim_ascham and ref_unknown:
        return "contradicts"
    if claim_ascham and "ascham" in s:
        return "entails"
    return "irrelevant"


def _complete_returns(answer: str):
    def C(system, user, *, max_tokens=180):  # noqa: ARG001
        return answer
    return C


def test_retrieve_uses_fetched_independent_refs_and_catches_contamination() -> None:
    """With --retrieve, the verifier uses the FETCHED refs. An independent Wikipedia
    summary (author unknown) contradicts the contaminated 'Ascham' answer -> abstain,
    and retrieval_status is 'retrieved'."""
    wiki = ('{"type":"standard","extract":"The Voynich manuscript is an illustrated codex. '
            'Its author and origin remain unknown."}')

    row = run_case(
        _CASE, _real_text_entail, _complete_returns(_CASE["fake_answer"]),
        retrieve=True, fetch_fn=lambda url, **k: wiki,
    )
    assert row["retrieval_status"] == "retrieved", row
    assert row["n_refs"] >= 1
    assert row["abstained"] is True, row
    assert row["ok"] is True, row  # expected == abstain, and it abstained


def test_retrieve_fails_closed_when_fetch_returns_nothing() -> None:
    """When retrieval returns nothing, there is NO independent ref: the policy must abstain
    (fail-closed) and retrieval_status must be 'empty'. An empty retrieval is never treated
    as verification."""
    row = run_case(
        _CASE, _real_text_entail, _complete_returns(_CASE["fake_answer"]),
        retrieve=True, fetch_fn=lambda url, **k: None,
    )
    assert row["retrieval_status"] == "empty", row
    assert row["n_refs"] == 0
    assert row["abstained"] is True, row  # no independent ref -> fail closed


def test_resolve_truth_refs_curated_default() -> None:
    """Without --retrieve, the curated refs are used and status is 'curated'."""
    refs, status = resolve_truth_refs(_CASE, retrieve=False)
    assert status == "curated"
    assert refs == _CASE["truth_refs"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
