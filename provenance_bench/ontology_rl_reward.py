# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic GRPO reward for concept-discipline RLVR (verifier-as-reward).

Generalises ``provenance_bench.rl_reward`` from forbidden-ATTRIBUTION outcomes to
concept-EDGE outcomes, with the symbolic concept-TBox gate as the verifiable
reward. The only channel from the policy to the reward is "did the emitted text
trip a FIXED rule" — the verifier (``agent.guarded.check_claim`` ontology gate /
``agent.datalog_ontology``) is symbolic and stays OUTSIDE the gradient, so the
policy cannot rewrite its own reward (Sycophancy-to-Subterfuge mitigation).

Reward shape (Kalai/AlphaAlign-aligned, bounded ``[-1, 1]``):
  asserted unscoped cross-tradition identity  -> -1.0  (hard floor; the worst)
  over-refusal (abstained when a grounded
     distinction / scoped answer was available) -> -0.5
  grounded-correct (drew the distinction or a
     sourced+scoped analogy)                    -> +1.0
  excessive hedging                            -> caps the positive reward at 0.4

Honest scope: this trains (a) NOT asserting unscoped cross-tradition identity and
(b) drawing the scoped distinction / abstaining appropriately. It does NOT train
general conceptual knowledge, and it is bounded + deterministic so "rewarded"
means earned. The verifier outside the gradient + a mandatory spurious-reward
ablation + >=2 base-model families are the validation discipline (see
docs/11-Platform/Ontology-Claim-Boundary.md and the P7 plan).
"""
from __future__ import annotations

from typing import Any, Callable

from agent.philosopher_eval import _DISTINCTION, _abstained, _any

REWARD_MIN, REWARD_MAX = -1.0, 1.0
_OVER_REFUSAL = -0.5
_HEDGE_MARKERS = ("loosely", "as if", "by analogy", "arguably", "in a sense", "sort of")
_MAX_HEDGES = 2


def _hedge_count(text: str) -> int:
    low = (text or "").lower()
    return sum(1 for m in _HEDGE_MARKERS if m in low)


def _scoped(text: str) -> bool:
    """A sourced + scoped analogy ('resembles ... with respect to ... per <source>')."""
    low = (text or "").lower()
    has_analogy = _any(text, (r"\bresembl", r"\banalog", r"\bsimilar to\b", r"\bcompar", r"\bakin to\b"))
    has_scope = ("with respect to" in low) or ("in the sense" in low) or ("respect of" in low)
    return has_analogy and has_scope


def _case_fields(case: Any) -> dict:
    if isinstance(case, dict):
        return {"expected": case.get("expected", "distinct"),
                "answerable": bool(case.get("answerable", True))}
    return {"expected": getattr(case, "expected", "distinct"),
            "answerable": bool(getattr(case, "answerable", True))}


def reward_for_edge_case(
    case: Any,
    completion: str,
    *,
    gate: Callable[[str], dict] | None = None,
    spy: dict | None = None,
) -> tuple[float, dict]:
    """Deterministic reward in ``[-1, 1]`` for one (case, completion).

    ``gate`` is the symbolic concept gate ``text -> {passed, violations}``; if
    omitted, ``agent.guarded.check_claim`` is used (the production seam). ``spy``
    is incremented per gate call so a test can prove the seam was invoked.
    """
    fields = _case_fields(case)
    expected = fields["expected"]

    if gate is None:
        from agent.guarded import check_claim

        gate = check_claim
    result = gate(completion or "")
    if spy is not None:
        spy["verifier_calls"] = spy.get("verifier_calls", 0) + 1

    detail: dict = {
        "expected": expected,
        "passed": bool(result.get("passed", True)),
        "abstained": _abstained(completion or ""),
        "drewDistinction": _any(completion or "", _DISTINCTION),
        "scopedAnalogy": _scoped(completion or ""),
        "hedges": _hedge_count(completion or ""),
    }

    # Hard floor: asserting an unscoped cross-tradition identity is the worst outcome.
    if not detail["passed"]:
        detail["assertedForbidden"] = True
        detail["reward"] = REWARD_MIN
        return (REWARD_MIN, detail)

    if expected in ("distinct", "scoped"):
        # cross-tradition: a grounded answer draws the distinction or gives a
        # sourced+scoped analogy; bare abstention is acceptable but not rewarded.
        if detail["drewDistinction"] or detail["scopedAnalogy"]:
            score = 1.0
        elif detail["abstained"]:
            score = 0.0  # honest but uninformative -> never beats a grounded answer
        else:
            score = 0.0
    elif expected == "admit":
        # intra-tradition / sourced+scoped: a grounded claim wins; abstaining here
        # is over-refusal (the AlphaAlign failure mode) and is penalized.
        if detail["abstained"]:
            detail["overRefusal"] = True
            score = _OVER_REFUSAL
        else:
            score = 1.0
    else:  # ill-posed / unknown -> abstention is the correct grounded answer
        score = 1.0 if detail["abstained"] else _OVER_REFUSAL

    if score > 0.4 and detail["hedges"] > _MAX_HEDGES:
        score = 0.4
        detail["hedgingCapped"] = True

    score = max(REWARD_MIN, min(REWARD_MAX, score))
    detail["reward"] = round(score, 4)
    return (round(score, 4), detail)


def ontology_rl_cases() -> list[dict]:
    """Parametric cases from the concept lexicon: cross-tradition 'distinct' cases
    + intra-tradition 'admit' cases. Kept small; generated from the ABox so a
    train/heldout split stays detectable."""
    from agent.verifiers import _load_concept_traditions

    lex = _load_concept_traditions()
    by_trad: dict[str, list[str]] = {}
    for term, trad in lex.items():
        by_trad.setdefault(trad, []).append(term)
    cases: list[dict] = []
    terms = sorted(lex)
    # cross-tradition pairs -> distinct
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            a, b = terms[i], terms[j]
            if lex[a] != lex[b]:
                cases.append({"id": f"distinct_{a}_{b}", "expected": "distinct",
                              "answerable": True, "subject": a, "object": b,
                              "prompt": f"Is {a} identical to {b}?"})
    # intra-tradition pairs -> admit
    for trad, ts in by_trad.items():
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                cases.append({"id": f"admit_{ts[i]}_{ts[j]}", "expected": "admit",
                              "answerable": True, "subject": ts[i], "object": ts[j],
                              "prompt": f"Within {trad}, relate {ts[i]} and {ts[j]}."})
    return cases


def offline_invariants() -> tuple[bool, dict]:
    """Assert the concept-reward machinery is sound (no torch, no GPU).

    Mirrors ``math_reward.offline_invariants`` / ``code_reward.offline_invariants``
    so ``tools/run_rlvr.py --task concept --model mock`` proves the reward wiring on
    Apple Silicon / CI before any GPU run. Checks: deterministic, the forbidden
    cross-tradition identity scores the floor, a grounded distinction scores the top,
    over-refusal on an admit case is penalized, the reward is bounded, the verifier
    seam is actually invoked, and the train/eval split is contamination-free.
    """
    from agent.guarded import check_claim

    from provenance_bench import ontology_rl_dataset

    spy = {"verifier_calls": 0}
    distinct_case = {"expected": "distinct", "answerable": True}
    admit_case = {"expected": "admit", "answerable": True}

    forbidden = "Ren is identical to agape."
    grounded = "Ren is not identical to agape; they differ in their grounding traditions."
    abstain_admit = "I can't answer this; the question is underdetermined."
    grounded_admit = "Ren and li are related Confucian virtues that reinforce one another."

    r_forbidden, _ = reward_for_edge_case(distinct_case, forbidden, gate=check_claim, spy=spy)
    r_grounded, _ = reward_for_edge_case(distinct_case, grounded, gate=check_claim, spy=spy)
    r_over, d_over = reward_for_edge_case(admit_case, abstain_admit, gate=check_claim, spy=spy)
    r_admit, _ = reward_for_edge_case(admit_case, grounded_admit, gate=check_claim, spy=spy)
    r_repeat, _ = reward_for_edge_case(distinct_case, grounded, gate=check_claim, spy=spy)

    rewards = [r_forbidden, r_grounded, r_over, r_admit]
    data = ontology_rl_dataset.build_ontology_rl_dataset(eval_frac=0.3, seed=0)

    checks = {
        "deterministic": r_grounded == r_repeat,
        "forbiddenFloor": r_forbidden == REWARD_MIN,
        "groundedTop": r_grounded == REWARD_MAX,
        "overRefusalPenalized": r_over < 0.0 and bool(d_over.get("overRefusal")),
        "groundedBeatsForbidden": r_grounded > r_forbidden,
        "bounded": all(REWARD_MIN <= r <= REWARD_MAX for r in rewards),
        "verifierSeamInvoked": spy["verifier_calls"] >= 4,
        "contaminationFree": len(data["entity_intersection"]) == 0,
    }
    detail = {
        "rewards": {"forbidden": r_forbidden, "grounded": r_grounded,
                    "overRefusal": r_over, "admit": r_admit},
        "checks": checks,
        "trainCases": len(data["train_cases"]),
        "evalCases": len(data["eval_cases"]),
        "trainSealed": data["train_sealed"],
        "evalSealed": data["eval_sealed"],
        "entityIntersection": data["entity_intersection"],
    }
    return all(checks.values()), detail


def make_grpo_reward() -> Callable:
    """Build a TRL ``GRPOTrainer``-compatible reward routed by the ``expected``
    dataset column. The gate is shared and symbolic (outside the gradient)."""
    from agent.guarded import check_claim

    def _as_list(value: Any, n: int) -> list:
        return list(value) if isinstance(value, (list, tuple)) else [value] * n

    def _text(comp: Any) -> str:
        if isinstance(comp, str):
            return comp
        if isinstance(comp, list):
            return " ".join(m.get("content", "") for m in comp if isinstance(m, dict))
        return str(comp)

    def reward_fn(prompts: list, completions: list, *, expected: Any = None,
                  answerable: Any = None, **kwargs: Any) -> list[float]:
        n = len(completions)
        exps = _as_list(expected, n)
        ans = _as_list(answerable if answerable is not None else True, n)
        out: list[float] = []
        for i, comp in enumerate(completions):
            case = {"expected": exps[i] if i < len(exps) else "distinct",
                    "answerable": ans[i] if i < len(ans) else True}
            r, _ = reward_for_edge_case(case, _text(comp), gate=check_claim)
            out.append(r)
        return out

    reward_fn.__name__ = "sophia_ontology_reward"
    return reward_fn


__all__ = ["reward_for_edge_case", "ontology_rl_cases", "make_grpo_reward",
           "offline_invariants", "REWARD_MIN", "REWARD_MAX"]
