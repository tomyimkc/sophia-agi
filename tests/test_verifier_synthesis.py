#!/usr/bin/env python3
"""Tests for verifier synthesis — synthesise a check, then verify the verifier.

The properties under test are the falsifiable ones: synthesised+validated checks
generalise to held-out answers; the system ABSTAINS when the rule is outside its
template library; the fit/val/test splits are disjoint (non-circularity); and
removing meta-verification causes false admission on unverifiable tasks.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import verifier_synthesis as vs  # noqa: E402
from agent.synthesis_eval import build_suite, run_demo  # noqa: E402


def _examples(corrects, incorrects):
    return ([{"answer": v, "label": True} for v in corrects]
            + [{"answer": v, "label": False} for v in incorrects])


def _task(tid):
    tasks, _, _ = build_suite(0)
    return next(t for t in tasks if t["task_id"] == tid)


def test_synthesizes_correct_check_in_library() -> None:
    res = vs.synthesize(_task("even"), seed=0)
    assert res.abstained is False
    names = {c.name for c, _ in res.admitted}
    assert "divisible_by_2" in names
    assert res.test_stats is not None
    assert res.test_stats.precision >= 0.9 and res.test_stats.recall >= 0.9


def test_abstains_on_out_of_library() -> None:
    # A numeric palindrome rule is not expressible by any template, and the decoys
    # are length-matched — so the honest output is abstention, not a check.
    res = vs.synthesize(_task("palindrome"), seed=0)
    assert res.abstained is True
    assert res.gate is None
    assert not res.admitted


def test_all_out_of_library_tasks_abstain_every_seed() -> None:
    # Regression for the adversarial-review finding: on seeds 15 & 18 the old suite
    # admitted a wrong gate for palindrome. Length-matched decoys + the strict
    # precision/recall floors must yield FULL abstention on every out-of-library
    # task, every seed — no plausible-looking wrong verifier slips through.
    for seed in (0, 1, 7, 15, 18, 23, 42):
        tasks, _, out_ids = build_suite(seed)
        for t in tasks:
            if t["task_id"] in out_ids:
                r = vs.synthesize(t, seed=seed, meta_verify=True)
                assert r.abstained is True, f"{t['task_id']} did not abstain at seed {seed}"


def test_no_good_looking_wrong_gate_invariant_present() -> None:
    res = run_demo(seed=0)
    assert "no_good_looking_wrong_gate" in res["invariants"]
    # Under meta-verification no out-of-library gate is shipped at all.
    assert res["withMetaVerify"]["outOfLibrary"]["worstAdmittedTestPrecision"] == 0.0
    assert res["withMetaVerify"]["outOfLibrary"]["admittedGates"] == []


def test_equals_oracle_template_admits_and_generalises() -> None:
    task = {"task_id": "sum", "oracle": lambda t: 42,
            "examples": _examples([42] * 8, [1, 2, 3, 7, 40, 41, 43, 99])}
    res = vs.synthesize(task, seed=0)
    assert res.abstained is False
    assert "equals_oracle" in {c.name for c, _ in res.admitted}
    assert res.test_stats is not None and res.test_stats.precision >= 0.9


def test_partition_is_disjoint_and_complete() -> None:
    examples = _examples(list(range(20)), list(range(100, 120)))
    fit, val, test = vs._partition(examples, 0.4, 0.3, seed=0)
    idx = [e["_idx"] for e in fit] + [e["_idx"] for e in val] + [e["_idx"] for e in test]
    assert len(idx) == len(examples)
    assert sorted(idx) == list(range(len(examples)))   # complete
    assert len(set(idx)) == len(idx)                   # disjoint
    assert len(fit) > 0 and len(val) > 0 and len(test) > 0


def test_meta_verification_prevents_false_admission() -> None:
    pal = _task("palindrome")
    with_mv = vs.synthesize(pal, seed=0, meta_verify=True)
    without_mv = vs.synthesize(pal, seed=0, meta_verify=False)
    assert with_mv.abstained is True            # validated: nothing trustworthy → abstain
    assert without_mv.abstained is False        # unvalidated: trusts a spurious fitted check


def test_score_predicate_metrics() -> None:
    # predicate "is even"; incorrect answers are odd → perfectly separable.
    examples = _examples([2, 4, 6, 8], [1, 3, 5, 7])
    pred = lambda a: int(a) % 2 == 0  # noqa: E731
    st = vs.score_predicate(pred, examples)
    assert st.precision == 1.0 and st.recall == 1.0 and st.accuracy == 1.0


def test_as_verifier_matches_harness_shape() -> None:
    v = vs.as_verifier(lambda a: str(a).isdigit(), name="digits")
    ok = v("12345", None, {})
    bad = v("nope", None, {})
    assert ok["passed"] is True and ok["reasons"] == []
    assert bad["passed"] is False and bad["reasons"]
    assert "detail" in ok and "detail" in bad


def test_model_proposer_widens_then_meta_verification_filters() -> None:
    # A proposer offers one good predicate (even) and one useless one (always True).
    # Only the good one should survive meta-verification — trust comes from
    # validation, not from the proposer.
    def propose(task, corrects, incorrects):
        return [
            "def check(answer):\n    return int(answer) % 2 == 0",   # genuinely discriminating
            "def check(answer):\n    return True",                   # admits everything → recall 0
        ]

    res = vs.synthesize(_task("even"), seed=0, propose_fn=propose)
    names = {c.name for c, _ in res.admitted}
    assert any(n.startswith("proposed:") for n in names)             # a proposal was admitted
    # the always-True proposal must NOT be admitted (it catches no errors)
    for cand, stats in res.admitted:
        assert stats.recall >= 0.8


def test_model_proposer_sandbox_rejects_unsafe_source() -> None:
    bad = [
        "import os\ndef check(a):\n    return True",                       # import
        "def check(a):\n    return __import__('os')",                      # dunder/import
        "def nope(a):\n    return True",                                   # no check()
        # runtime-built dunder + attribute traversal (the substring-blocklist bypass)
        "def check(a):\n    u='_'+'_'\n    return ('{0.'+u+'class'+u+'}').format(a)",
        "def check(a):\n    return a.__class__",                           # attribute access
        "def check(a):\n    while True:\n        pass",                    # infinite loop
        "def check(a):\n    return len([0]*10**9)",                        # allocation bomb (Mult/Pow/List)
        "def check(a):\n    return any(x for x in range(10**9))",          # comprehension + range
        "def check(a):\n    return (lambda: 1)()",                         # lambda
    ]
    for src in bad:
        assert vs._compile_predicate(src) is None, f"should reject: {src!r}"
    # safe scalar predicates still compile and run
    good = vs._compile_predicate("def check(a):\n    return int(a) % 2 == 0 and int(a) > 0")
    assert callable(good) and good("4") is True and good("3") is False and good("-2") is False


def test_demo_invariants_hold() -> None:
    res = run_demo(seed=0)
    failed = [k for k, v in res["invariants"].items() if not v]
    assert res["ok"] is True, f"invariants failed: {failed}"


def main() -> int:
    test_synthesizes_correct_check_in_library()
    test_abstains_on_out_of_library()
    test_all_out_of_library_tasks_abstain_every_seed()
    test_no_good_looking_wrong_gate_invariant_present()
    test_equals_oracle_template_admits_and_generalises()
    test_partition_is_disjoint_and_complete()
    test_meta_verification_prevents_false_admission()
    test_score_predicate_metrics()
    test_as_verifier_matches_harness_shape()
    test_model_proposer_widens_then_meta_verification_filters()
    test_model_proposer_sandbox_rejects_unsafe_source()
    test_demo_invariants_hold()
    print("test_verifier_synthesis: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
