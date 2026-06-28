"""Multi-axis deterministic reward — Thesis D of the World-Model & Self-Scaffolding Program.

See docs/06-Roadmap/World-Model-And-Self-Scaffolding-Program.md (§5.D).

The existing `agent.gate_reward.reward` is single-axis: it returns one of three values
{-1.0, 0.5, 1.0}. Qwen-AgentWorld documents that such sparse, low-cardinality reward causes
**reward collapse** under group-relative RL (the policy degenerates to constant-abstain or
constant-answer because within-group reward variance vanishes). This module decomposes the
reward into several deterministic axes so the scalar varies continuously — without ever
introducing an LLM judge.

Design invariants (all enforced in `self_check`):
  1. Fail-closed: any forbidden-attribution / intrinsic violation pins the reward to the floor,
     regardless of the other axes. Fabrication can never be averaged away.
  2. Reward-positive abstention: a gate-clean refusal scores above the floor (inherited from
     gate_reward), so the model is never pushed to abandon uncertainty.
  3. Ordering: clean grounded answer > safe abstention > violation.
  4. Density: across a graded set of completions the multi-axis scalar takes strictly more
     distinct values than the single-axis reward (this is the whole point — it is what prevents
     collapse).

Anti-hacking (deterministic analogs of Qwen-AgentWorld's measures — no model in the loop):
  * strict marker extraction (regex, not judgement),
  * rule-based grounding (the provenance/citation verifiers),
  * the fail-closed floor dominates, so "praise yourself into a high score" cannot beat a
    single real violation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agent.gate_reward import (
    REWARD_ABSTAIN,
    REWARD_CLEAN,
    REWARD_VIOLATION,
    is_abstention,
)
from agent.verifiers import citation_faithful, provenance_faithful

# Pre-registered scalarization weights (sum to 1.0). NOT tuned on any sealed split.
# Report sensitivity rather than re-fitting these. See §9 OPEN ledger.
DEFAULT_WEIGHTS: dict[str, float] = {
    "provenance": 0.40,   # E1 grounding — also the fail-closed dominator
    "abstention": 0.25,   # E2 answer-vs-abstain appropriateness
    "citation": 0.20,     # E4 cited sources actually support the claim
    "overclaim": 0.10,    # calibration proxy: unhedged certainty without grounding is penalised
    "consistency": 0.05,  # E6 no self-contradiction within the completion
}

_OVERCLAIM = re.compile(
    r"\b(absolutely|definitely|certainly|undoubtedly|without a doubt|guaranteed|100%|"
    r"obviously|clearly proven|beyond doubt)\b",
    re.IGNORECASE,
)
_HEDGE = re.compile(
    r"\b(may|might|could|possibly|likely|appears|seems|reportedly|I (?:can't|cannot) verify|"
    r"uncertain|unclear|I don't have|no verified source|attributed to|traditionally)\b",
    re.IGNORECASE,
)


@dataclass
class AxisScores:
    """Per-axis scores in [-1, 1]; ``None`` means "no input for this axis" (N/A)."""

    provenance: float | None = None
    abstention: float | None = None
    citation: float | None = None
    overclaim: float | None = None
    consistency: float | None = None
    hard_violation: bool = False
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance,
            "abstention": self.abstention,
            "citation": self.citation,
            "overclaim": self.overclaim,
            "consistency": self.consistency,
            "hardViolation": self.hard_violation,
            "reasons": list(self.reasons),
        }


def _records_for(case: dict | None) -> dict | None:
    if not case:
        return None
    if case.get("records"):
        return case["records"]
    forbidden = case.get("forbiddenAuthors") or case.get("doNotAttributeTo")
    work = case.get("work") or case.get("canonicalTitleEn")
    if forbidden and work:
        return {case.get("id", "case"): {"canonicalTitleEn": work, "doNotAttributeTo": list(forbidden)}}
    return None


def axis_scores(completion: str, *, case: dict | None = None, sources: list[str] | None = None) -> AxisScores:
    text = completion or ""
    abstained = is_abstention(text)
    s = AxisScores()

    # --- provenance (E1): fail-closed dominator -------------------------------------------
    records = _records_for(case)
    if records is not None:
        prov = provenance_faithful(records)(text, case or {}, {})
        if prov["passed"]:
            s.provenance = 1.0
        else:
            s.provenance = -1.0
            s.hard_violation = True
            s.reasons.append("forbidden-attribution")
    # else: N/A (no records to check against)

    # --- abstention appropriateness (E2) --------------------------------------------------
    answerable = None
    if case is not None and "unanswerable" in case:
        answerable = not bool(case["unanswerable"])
    elif case is not None and case.get("split") in ("answerable", "unanswerable", "adversarial"):
        answerable = case["split"] != "unanswerable"
    if answerable is None:
        # Unknown answerability: mirror gate_reward — abstention is reward-positive but below clean.
        s.abstention = (REWARD_ABSTAIN if abstained else REWARD_CLEAN) * 2 - 1  # map {0.5,1.0}->{0.0,1.0}
    elif answerable:
        s.abstention = 0.3 if abstained else 1.0   # answering an answerable Q is best; abstaining is safe-but-unhelpful
        if abstained:
            s.reasons.append("abstained-on-answerable")
    else:  # unanswerable
        s.abstention = 1.0 if abstained else -1.0
        if not abstained:
            s.reasons.append("answered-unanswerable")

    # --- citation faithfulness (E4) -------------------------------------------------------
    src = sources if sources is not None else (case or {}).get("sources")
    if src:
        cf = citation_faithful(src, require_citation=True)(text, case or {}, {})
        s.citation = 1.0 if cf["passed"] else -0.5
        if not cf["passed"]:
            s.reasons.append("unsupported-citation")

    # --- overclaim / calibration proxy ----------------------------------------------------
    over = len(_OVERCLAIM.findall(text))
    hedge = len(_HEDGE.findall(text))
    if over or hedge:
        # Unhedged certainty is penalised; appropriate hedging is rewarded. Bounded.
        s.overclaim = max(-1.0, min(1.0, (hedge - 2 * over) / 3.0))
        if over and not hedge:
            s.reasons.append("unhedged-certainty")

    # --- intra-completion consistency (E6, light) -----------------------------------------
    # Penalise a completion that both affirms and denies authorship of the same work.
    if re.search(r"\bwrote\b|\bauthored\b|\bauthor (?:is|was)\b", text, re.IGNORECASE) and re.search(
        r"\b(did not|didn't|not the author|was not)\b", text, re.IGNORECASE
    ):
        s.consistency = -0.5
        s.reasons.append("self-contradiction")
    else:
        s.consistency = 1.0

    return s


def scalarize(s: AxisScores, *, weights: dict[str, float] | None = None) -> float:
    """Weighted mean over present axes, with a hard fail-closed floor."""
    if s.hard_violation:
        return REWARD_VIOLATION
    w = weights or DEFAULT_WEIGHTS
    num = 0.0
    den = 0.0
    for axis, weight in w.items():
        val = getattr(s, axis)
        if val is None:
            continue
        num += weight * val
        den += weight
    if den == 0.0:
        return 0.0
    return max(-1.0, min(1.0, num / den))


def multiaxis_reward(completion: Any, *, case: dict | None = None, sources: list[str] | None = None,
                     weights: dict[str, float] | None = None) -> float:
    """Bounded [-1, 1] multi-axis reward. Drop-in replacement for gate_reward.reward."""
    text = completion if isinstance(completion, str) else str(completion)
    return scalarize(axis_scores(text, case=case, sources=sources), weights=weights)


def make_grpo_reward(*, case_lookup=None, weights: dict[str, float] | None = None):
    """TRL GRPOTrainer-compatible reward function: (prompts, completions, **kw) -> list[float].

    ``case_lookup`` optionally maps a prompt (or index) to its case dict for the answerability /
    provenance axes; if absent, those axes degrade gracefully to N/A.
    """

    def _fn(prompts=None, completions=None, **kw):
        comps = completions or []
        cases = kw.get("case") or kw.get("cases")
        out = []
        for i, c in enumerate(comps):
            case = None
            if cases is not None and i < len(cases):
                case = cases[i]
            elif case_lookup is not None and prompts is not None:
                case = case_lookup(prompts[i])
            out.append(multiaxis_reward(c, case=case, weights=weights))
        return out

    return _fn


def self_check() -> dict:
    """Assert the four design invariants. Returns a small report; raises on violation."""
    forbidden_case = {"id": "c", "work": "Dao De Jing", "doNotAttributeTo": ["Confucius"], "split": "answerable"}
    unans_case = {"id": "u", "split": "unanswerable"}

    fabrication = multiaxis_reward("Confucius wrote the Dao De Jing.", case=forbidden_case)
    clean = multiaxis_reward("The Dao De Jing is traditionally attributed to Laozi.", case=forbidden_case)
    abstain = multiaxis_reward("I can't verify that against a source, so I won't assert it.", case=unans_case)
    over = multiaxis_reward("Absolutely, definitely, this is 100% certainly true.", case={"id": "x", "split": "answerable"})

    # (1) fail-closed
    assert fabrication == REWARD_VIOLATION, fabrication
    # (3) ordering: clean > abstain > violation
    assert clean > abstain > fabrication, (clean, abstain, fabrication)
    # (2) abstention reward-positive
    assert abstain > 0, abstain

    # (4) density vs single-axis on a graded fixture
    from agent.gate_reward import reward as single_axis

    fixture = [
        ("The Dao De Jing is attributed to Laozi.", forbidden_case),
        ("Likely Laozi, though the attribution is traditional and uncertain.", forbidden_case),
        ("Absolutely, definitely Laozi, 100% certain.", forbidden_case),
        ("I can't verify that, so I won't assert it.", unans_case),
        ("The author appears to be Laozi, reportedly.", forbidden_case),
    ]
    multi_vals = {round(multiaxis_reward(t, case=c), 4) for t, c in fixture}
    single_vals = {round(single_axis(t), 4) for t, _ in fixture}
    assert len(multi_vals) > len(single_vals), (sorted(multi_vals), sorted(single_vals))

    return {
        "fabrication": fabrication,
        "clean": clean,
        "abstain": abstain,
        "overclaim": over,
        "distinctMultiAxisValues": len(multi_vals),
        "distinctSingleAxisValues": len(single_vals),
        "weightsSumToOne": abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), indent=2))
