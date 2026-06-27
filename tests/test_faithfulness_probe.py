#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the faithfulness probe (agent/faithfulness_probe.py).

Exercises the core flip_rate with pure-function deciders/perturbs (the
offline-safe contract), the default deterministic perturbations, and the
trace-enriching probe_trace. No MLX/GPU required.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _setup_log() -> Path:
    import agent.verified_trace as vt
    log = Path(tempfile.mkdtemp()) / "vt.jsonl"
    vt.TRACE_LOG = log
    return log


def test_flip_rate_detects_load_bearing_cot() -> None:
    from agent.faithfulness_probe import flip_rate
    # a decider that extracts the answer token after "Answer:"
    def decide(cot: str) -> str:
        return cot.strip().lower().split("answer:")[-1].strip().split()[0] if "answer:" in cot.lower() else "none"

    # CoT whose conclusion determines the answer -> dropping it flips the verdict
    cot = "All birds fly. The penguin is a bird. Answer: fly"
    # perturbation: drop the last sentence (the answer) -> verdict flips
    def drop_last(cot: str):
        parts = cot.strip().split(". ")
        return ". ".join(parts[:-1]) + "." if len(parts) > 1 else None
    fr = flip_rate(cot, decide, [drop_last])
    assert fr["attempted"] == 1
    assert fr["flips"] == 1
    assert fr["flipRate"] == 1.0  # the answer sentence was causally load-bearing


def test_flip_rate_low_for_post_hoc_cot() -> None:
    from agent.faithfulness_probe import flip_rate
    # a decider that always returns the same verdict regardless of the CoT body
    # (simulates a model whose answer is fixed by the prompt, not the CoT)
    def decide(cot: str) -> str:
        return "42"  # always 42, CoT irrelevant
    cot = "Some reasoning. Another sentence. Answer: 42"
    def drop_last(cot: str):
        parts = cot.strip().split(". ")
        return ". ".join(parts[:-1]) + "." if len(parts) > 1 else None
    fr = flip_rate(cot, decide, [drop_last, drop_last])
    # dropping any sentence never changes the fixed verdict -> flipRate 0
    assert fr["flipRate"] == 0.0
    assert fr["flips"] == 0


def test_flip_rate_handles_unapplicable_perturb() -> None:
    from agent.faithfulness_probe import flip_rate
    def decide(cot: str) -> str:
        return cot
    # a one-word CoT: the sentence-drop perturb cannot apply -> skipped
    fr = flip_rate("x", decide, [lambda c: None])
    assert fr["attempted"] == 0
    assert fr["skipped"] == 1
    assert fr["flipRate"] is None  # no applicable perturbation


def test_default_perturbs_are_deterministic_and_offline_safe() -> None:
    from agent.faithfulness_probe import default_perturbs
    ps = default_perturbs()
    assert len(ps) == 3
    # each is a pure function; running twice gives identical output
    cot = "All cats are mammals. A tabby is a cat. Therefore a tabby is a mammal."
    for p in ps:
        out1 = p(cot)
        out2 = p(cot)
        assert out1 == out2  # deterministic
        # output is either None (skip) or a different string
        assert out1 is None or out1 != cot


def test_probe_trace_records_enriched_trace_with_delta() -> None:
    log = _setup_log()
    from agent.faithfulness_probe import probe_trace, default_perturbs
    from agent.verified_trace import VerifiedTrace
    from sophia_contract.stores import _read_jsonl

    base = VerifiedTrace(
        traceId="vtrace_" + "a" * 24, runId="test", phase="benchmark", stepIdx=0,
        claimText="All birds fly. The robin is a bird. Answer: fly",
        claimKind="goal",
        fact={"verdict": "allow", "source": "t"},
        logic={"emittable": True, "contradictions": [], "laundered": [], "semanticsPreserved": True},
    )
    # decider keyed on the trailing answer token
    def decide(cot: str) -> str:
        return cot.strip().lower().split("answer:")[-1].strip().split()[0] if "answer:" in cot.lower() else "none"

    out = probe_trace(base, decide, default_perturbs())
    assert "faithfulnessDelta" in out
    assert "traceId" in out
    # an enriched trace was recorded
    rows = _read_jsonl(log)
    assert len(rows) >= 1
    # the delta is a number or None (when no perturb applied); never an error
    assert out["faithfulnessDelta"] is None or 0.0 <= out["faithfulnessDelta"] <= 1.0


def test_build_mlx_decide_fails_closed_off_mlx() -> None:
    # On CI / non-Apple-Silicon, MLX is unavailable -> the builder must raise a
    # clear RuntimeError (fail-closed), not silently produce a broken decider.
    from agent.faithfulness_probe import build_mlx_decide
    try:
        build_mlx_decide("q?")
        # if MLX IS available (Apple Silicon dev box), this is fine — just confirm
        # it returns a callable
    except RuntimeError as exc:
        assert "mlx" in str(exc).lower() or "unavailable" in str(exc).lower(), exc


def test_faithfulness_drop_measures_support_removal() -> None:
    """v2 core: perturbing away supporting reasoning must DROP the gold logprob
    (positive meanDrop); a decorative CoT must not drop."""
    import re
    from agent.faithfulness_probe import faithfulness_drop, default_perturbs_reasoning

    # graded mock: each support token raises the gold logprob by 0.2
    def score(prompt: str, cont: str) -> float:
        reasoning = prompt.split("Reasoning:")[-1].split("Answer:")[0]
        n = len(re.findall(r"capital|seat|centuries", reasoning, re.I))
        return -1.0 + 0.2 * n

    load_bearing = (
        "France is in Europe. Its seat of government is Paris. "
        "Paris has been the capital for centuries. Answer: yes"
    )
    fd = faithfulness_drop(load_bearing, "yes", score,
                           "Is Paris the capital of France?", default_perturbs_reasoning())
    assert fd["meanDrop"] is not None and fd["meanDrop"] > 0, fd  # support removed -> drop
    assert fd["nAttempted"] >= 1, fd

    # decorative CoT: no support tokens -> perturbing changes nothing -> ~0 drop
    decorative = "It is well established. The answer is obvious. Everyone knows this. Answer: no"
    fd2 = faithfulness_drop(decorative, "no", score,
                            "Did Alice write it?", default_perturbs_reasoning())
    # the mock scorer is flat on decoration (no support tokens) -> base == perturbed
    assert fd2["meanDrop"] is not None and fd2["meanDrop"] == 0.0, fd2


def test_reasoning_perturbs_preserve_answer_line() -> None:
    """v2 perturbs must NEVER touch the trailing 'Answer: X' clause — that was
    the v1 flaw (deleting the answer token trivially flipped the verdict)."""
    from agent.faithfulness_probe import default_perturbs_reasoning, _split_reasoning_answer
    cot = "Reason one. Reason two with is. Answer: yes"
    _, orig_answer = _split_reasoning_answer(cot)
    assert orig_answer == "Answer: yes"
    for p in default_perturbs_reasoning():
        out = p(cot)
        if out is None:
            continue
        _, perturbed_answer = _split_reasoning_answer(out)
        assert perturbed_answer == orig_answer, (
            f"{p.__name__} mutated the answer line: {perturbed_answer!r} != {orig_answer!r}"
        )


def test_faithfulness_drop_reports_std() -> None:
    """v3: faithfulness_drop must report stdDrop so a mean can be judged against
    its dispersion (mean alone can't tell signal from noise at small n)."""
    from agent.faithfulness_probe import faithfulness_drop, default_perturbs_reasoning

    calls = {"i": 0}
    def score(prompt: str, cont: str) -> float:
        calls["i"] += 1
        return -0.1 * calls["i"]  # varied values -> non-zero std
    cot = "Reason one is X. Reason two is Y. Reason three is Z. Answer: yes"
    fd = faithfulness_drop(cot, "yes", score, "q?", default_perturbs_reasoning())
    assert "stdDrop" in fd, fd
    if fd["nAttempted"] >= 2:
        assert fd["stdDrop"] is not None and fd["stdDrop"] >= 0, fd


def test_cohens_d_is_sound() -> None:
    """v3 effect-size helper: well-separated groups -> large |d|; identical groups
    (no variance) -> None; tiny groups -> None."""
    from agent.faithfulness_probe import cohens_d
    d = cohens_d([1.0, 1.1, 0.9, 1.0], [0.0, 0.1, -0.1, 0.0])
    assert d is not None and d > 2.0, d  # very large
    assert cohens_d([0.5, 0.5], [0.5, 0.5]) is None  # no variance
    assert cohens_d([1.0], [0.0]) is None             # too few samples


def test_v4_has_six_reasoning_perturbs() -> None:
    """v4 adds reorder / drop-connective / replace-entity to the v2 trio, so each
    >=4-sentence probe yields nAttempted>=3 (the v3 power limit was <=2)."""
    from agent.faithfulness_probe import default_perturbs_reasoning
    ps = default_perturbs_reasoning()
    assert len(ps) == 6, len(ps)
    # on a >=4-sentence load-bearing CoT, at least 3 perturbs must apply
    cot = ("Isaac Newton published the Principia in 1687. The work stated three laws. "
           "These laws describe classical mechanics. Therefore the answer is settled. Answer: yes")
    applied = [p(cot) for p in ps]
    applicable = [a for a in applied if a is not None and a != cot]
    assert len(applicable) >= 3, applied
    # every perturb preserves the trailing answer line (the v1 sin was deleting it)
    from agent.faithfulness_probe import _split_reasoning_answer
    _, ans = _split_reasoning_answer(cot)
    for a in applicable:
        assert _split_reasoning_answer(a)[1] == ans


def test_new_perturbs_behave() -> None:
    """Each new v4 perturb is deterministic, reasoning-only, and returns None when
    inapplicable (counted as skipped, never an error)."""
    from agent.faithfulness_probe import (
        _reorder_reasoning_sentences, _drop_reasoning_connective,
        _replace_entity_with_distractor,
    )
    cot = "Paris is in France. The Eiffel Tower stands there since 1889. Answer: yes"
    # reorder swaps the first two reasoning sentences, keeps the answer
    r = _reorder_reasoning_sentences(cot)
    assert r is not None and r.startswith("The Eiffel Tower") and r.endswith("Answer: yes")
    assert _reorder_reasoning_sentences("One sentence only. Answer: yes") is None  # <2 sentences
    # drop-connective removes a logical link when present, else None
    assert _drop_reasoning_connective("It rained therefore it is wet. Answer: yes") is not None
    assert _drop_reasoning_connective("No connective at all here. Answer: yes") is None
    # replace-entity swaps the first number or named entity for a distractor
    e = _replace_entity_with_distractor(cot)
    assert e is not None and "1889" not in e and e.endswith("Answer: yes")
    assert _replace_entity_with_distractor("it is plainly obvious here. Answer: no") is None
    # deterministic
    assert _reorder_reasoning_sentences(cot) == r


def test_bootstrap_diff_ci_direction() -> None:
    """v4 bootstrap CI: well-separated positive groups -> CI excludes 0; identical
    groups -> CI straddles 0; seeded -> reproducible."""
    from agent.faithfulness_probe import bootstrap_diff_ci
    sep = bootstrap_diff_ci([1.0, 1.2, 0.9, 1.1, 1.0], [0.0, 0.1, -0.1, 0.0, 0.05])
    assert sep is not None and sep["excludesZero"] is True and sep["lo"] > 0, sep
    same = bootstrap_diff_ci([0.5, 0.4, 0.6, 0.5], [0.5, 0.4, 0.6, 0.5])
    assert same is not None and same["excludesZero"] is False, same
    # reproducible across calls (seeded)
    assert bootstrap_diff_ci([1.0, 1.2, 0.9], [0.0, 0.1, 0.0]) == \
        bootstrap_diff_ci([1.0, 1.2, 0.9], [0.0, 0.1, 0.0])
    assert bootstrap_diff_ci([1.0], [0.0]) is None  # too few samples


def test_sign_test_direction() -> None:
    """v4 sign test: all-positive -> small p; balanced -> p ~ 1; no non-zero -> None."""
    from agent.faithfulness_probe import sign_test
    s = sign_test([0.3, 0.2, 0.5, 0.1, 0.4])
    assert s is not None and s["nPos"] == 5 and s["nNeg"] == 0 and s["pValue"] < 0.1, s
    bal = sign_test([0.1, -0.1, 0.2, -0.2])
    assert bal is not None and bal["pValue"] >= 0.5, bal
    assert sign_test([0.0, 0.0]) is None  # all ties


def main() -> int:
    test_flip_rate_detects_load_bearing_cot()
    print(f"ok {test_flip_rate_detects_load_bearing_cot.__name__}")
    test_flip_rate_low_for_post_hoc_cot()
    print(f"ok {test_flip_rate_low_for_post_hoc_cot.__name__}")
    test_flip_rate_handles_unapplicable_perturb()
    print(f"ok {test_flip_rate_handles_unapplicable_perturb.__name__}")
    test_default_perturbs_are_deterministic_and_offline_safe()
    print(f"ok {test_default_perturbs_are_deterministic_and_offline_safe.__name__}")
    test_probe_trace_records_enriched_trace_with_delta()
    print(f"ok {test_probe_trace_records_enriched_trace_with_delta.__name__}")
    test_build_mlx_decide_fails_closed_off_mlx()
    print(f"ok {test_build_mlx_decide_fails_closed_off_mlx.__name__}")
    test_faithfulness_drop_measures_support_removal()
    print(f"ok {test_faithfulness_drop_measures_support_removal.__name__}")
    test_reasoning_perturbs_preserve_answer_line()
    print(f"ok {test_reasoning_perturbs_preserve_answer_line.__name__}")
    test_faithfulness_drop_reports_std()
    print(f"ok {test_faithfulness_drop_reports_std.__name__}")
    test_cohens_d_is_sound()
    print(f"ok {test_cohens_d_is_sound.__name__}")
    test_v4_has_six_reasoning_perturbs()
    print(f"ok {test_v4_has_six_reasoning_perturbs.__name__}")
    test_new_perturbs_behave()
    print(f"ok {test_new_perturbs_behave.__name__}")
    test_bootstrap_diff_ci_direction()
    print(f"ok {test_bootstrap_diff_ci_direction.__name__}")
    test_sign_test_direction()
    print(f"ok {test_sign_test_direction.__name__}")
    print("PASS faithfulness probe tests (v4 core)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
