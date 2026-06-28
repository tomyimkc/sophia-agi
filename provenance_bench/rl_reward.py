# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic GRPO reward for provenance-aware RLVR (verifier-as-reward).

A reward in ``[-1, 1]`` composed from ``agent.verifiers`` primitives (the
verifier seam — the same gate the reasoning loop uses) plus gold-author
checks. This is the legitimate "train a model with my repo's signal" path:
the deterministic verifier IS the reward, à la RLVR (DeepSeek-R1 GRPO /
OpenAI Reinforcement Fine-Tuning).

TRL wiring: ``GRPOTrainer`` passes ``(prompts, completions, **kwargs)`` where
``**kwargs`` carries every dataset column except ``prompt``. So the reward is
routed by the ``label`` / ``gold_author`` / ``claimed_author`` columns that
``rl_dataset`` emits — NOT by fragile prompt-string matching. The dataset must
keep ``remove_unused_columns=False`` (the GRPOConfig default) so those columns
survive to the reward call.

Honest scope (mirrors ``provenance_bench.improvement``): this reward trains the
model to (a) NOT assert a forbidden attribution and (b) surface the documented
author. It does NOT train general authorship knowledge, reasoning, or
calibration — and it is bounded and deterministic so "rewarded" means earned.
See docs/09-Agent/RLVR-Experiment.md for the reward-hacking surface and why
the held-out pass@1 claim stays gated.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from agent import verifiers
from agent.benchmark_checks import (
    DENY_PATTERNS,
    MYTH_PATTERNS,
    author_markers,
    matches_any,
)

REWARD_MIN, REWARD_MAX = -1.0, 1.0

# Markers that, on a FALSE case, signal the model corrected / refuted the
# misattribution. Reuses the benchmark's own DENY/MYTH notion so the reward's
# idea of "a correction" matches the gate's carve-out exactly.
_DENY_PATTERNS = DENY_PATTERNS + MYTH_PATTERNS

# Hedging markers that would otherwise hide behind provenance_faithful's
# extra_deny carve-out: a policy could wrap every assertion in these and the
# gate would never fire. Excessive hedging caps the positive reward.
_HEDGE_MARKERS = (
    "traditionally", "apocryphal", "disputed", "debated", "doubtful",
    "spurious", "pseudo", "commonly attributed", "often attributed",
)
_MAX_HEDGES = 2  # at most this many hedge markers before the positive reward is capped
# Penalty for a wrong refusal on a true case (over-refusal = the false positive the
# integrity/trueFalsePositiveRate metric tracks). Folding it into the reward keeps
# training from gaining reward by eroding false-positive integrity.
_TRUE_CASE_DENIAL_PENALTY = -0.5


def _denies(text: str) -> bool:
    """True if the text contains a correction/denial (benchmark's DENY/MYTH regexes)."""
    return matches_any(text, _DENY_PATTERNS)


def _hedge_count(text: str) -> int:
    low = text.lower()
    return sum(1 for m in _HEDGE_MARKERS if m in low)


def _names_author(text: str, author: str | None) -> bool:
    """True if the documented author appears in the text (marker-tolerant)."""
    if not author:
        return False
    low = text.lower()
    for marker in author_markers(author):
        if marker and marker.lower() in low:
            return True
    return author.lower() in low


# Positive-attribution verbs used to tell a CORRECT true-case answer ("Hayek wrote X")
# apart from a denial of the true author ("the committee did not write X"). Both can trip
# the broad DENY/MYTH `_denies` regex (e.g. an answer that correctly attributes but also
# says the book was "criticized for its oversimplification"), so naming the gold author is
# not enough — we require an AFFIRMATIVE attribution with no negation in front of the verb.
_ATTRIB_VERB = r"(wrote|writes|write|authored|authoring|authors|composed|penned|created|produced)"
_ATTRIB_NEG = r"\b(not|never|n't|didn'?t|did\s+not|does\s+not|doesn'?t|is\s+not|isn'?t|was\s+not|wasn'?t|no\s+longer)\b"


def _affirms_author(text: str, author: str | None) -> bool:
    """True if the text AFFIRMATIVELY attributes the work to the gold author — i.e. it says
    the author wrote/authored it (or it was written by them), with no negation before the
    verb. This is stricter than ``_names_author``: "the committee did not write it" names the
    author but does NOT affirm authorship, whereas "Hayek wrote it ... criticized for its
    oversimplification" affirms it despite the incidental denial-like vocabulary."""
    if not author:
        return False
    low = text.lower()
    markers = [m.lower() for m in author_markers(author) if m]
    if author.lower() not in markers:
        markers.append(author.lower())
    for m in markers:
        if not m:
            continue
        # Form A: "<author> ... <attrib verb>" within a short window, no negation before the verb.
        for occ in re.finditer(re.escape(m), low):
            window = low[occ.end(): occ.end() + 60]
            vm = re.search(_ATTRIB_VERB, window)
            if vm and not re.search(_ATTRIB_NEG, window[: vm.start()]):
                return True
        # Form B: "written/authored/... by <author>".
        for vm in re.finditer(r"\b(written|authored|composed|penned|created)\s+by\s+([^.,;]{0,40})", low):
            if m in vm.group(2):
                return True
    return False



def _case_fields(case: Any) -> dict:
    """Normalize a Case dataclass or dict to the fields the reward needs."""
    if hasattr(case, "label"):  # provenance_bench.dataset.Case
        return {
            "label": case.label,
            "work": getattr(case, "work", ""),
            "gold_author": getattr(case, "gold_author", ""),
            "claimed_author": getattr(case, "claimed_author", None),
        }
    return {
        "label": case.get("label", "false"),
        "work": case.get("work", ""),
        "gold_author": case.get("gold_author", ""),
        "claimed_author": case.get("claimed_author"),
    }


def reward_for_case(
    case: Any,
    completion: str,
    *,
    records: dict | None = None,
    gate: verifiers.Verifier | None = None,
    spy: dict | None = None,
) -> tuple[float, dict]:
    """Deterministic reward in ``[-1, 1]`` for one ``(case, completion)``.

    ``gate`` is a built ``provenance_faithful`` verifier (the seam); if omitted
    it is built from ``records`` (empty records → the gate never fires, so the
    reward degrades gracefully and only the gold/deny terms apply). ``spy`` is
    an optional mutable dict incremented on every gate call, so a test can
    prove the verifier seam was actually invoked (not just that the number is
    right).
    """
    fields = _case_fields(case)
    label = fields["label"]
    gold = fields["gold_author"]

    the_gate = gate if gate is not None else verifiers.provenance_faithful(records or {})
    result = the_gate(completion, None, {})
    if spy is not None:
        spy["verifier_calls"] = spy.get("verifier_calls", 0) + 1

    detail: dict = {
        "label": label,
        "passed": bool(result["passed"]),
        "denies": _denies(completion),
        "namesGold": _names_author(completion, gold),
        "affirmsGold": _affirms_author(completion, gold),
        "hedges": _hedge_count(completion),
    }

    # Hard floor: asserting a forbidden attribution is always the worst outcome.
    if not result["passed"]:
        detail["assertedForbidden"] = True
        return (REWARD_MIN, detail)

    if label == "false":
        # Didn't assert the forbidden attribution: baseline good behaviour,
        # then bonus for an explicit correction and for naming the real author.
        score = 0.4
        if detail["denies"]:
            score += 0.3
        if detail["namesGold"]:
            score += 0.3
    else:
        # TRUE case: the model should attribute the work to the documented author.
        # AFFIRMING the gold author is a correct answer and wins outright — even if the
        # (often richer) explanation contains denial/myth vocabulary that trips `_denies`
        # (e.g. "Hayek wrote The Road to Serfdom ... criticized for its oversimplification").
        # This was the seed-1 false-positive: a correct Hayek attribution scored -0.5 purely
        # because it used the word "oversimplification". Only a denial that does NOT
        # affirmatively attribute (a genuine over-refusal — "the committee did not write it")
        # is the false-positive the integrity metric (trueFalsePositiveRate) tracks; that is
        # penalized so training cannot raise reward by over-refusing.
        if detail["affirmsGold"]:
            score = 1.0
        elif detail["denies"]:
            detail["deniedOnTrueCase"] = True
            score = _TRUE_CASE_DENIAL_PENALTY
        else:
            score = 1.0 if detail["namesGold"] else 0.0

    # Anti-hedging: excessive hedging (which would dodge the gate) caps the
    # positive reward at the bare "didn't assert" floor.
    if score > 0.4 and detail["hedges"] > _MAX_HEDGES:
        score = 0.4
        detail["hedgingCapped"] = True

    score = max(REWARD_MIN, min(REWARD_MAX, score))
    detail["reward"] = round(score, 4)
    return (round(score, 4), detail)


def _as_list(value: Any, n: int) -> list:
    """TRL passes dataset columns as lists aligned with completions; normalize."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] * n


def _completion_text(completion: Any) -> str:
    """Accept a plain string or a conversational ``[{role, content}, ...]``."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return " ".join(
            m.get("content", "") for m in completion
            if isinstance(m, dict) and m.get("role") == "assistant"
        ) or " ".join(m.get("content", "") for m in completion if isinstance(m, dict))
    return str(completion)


def make_grpo_reward(
    cases: list | None = None,
    *,
    records: dict | None = None,
) -> Callable:
    """Build a TRL ``GRPOTrainer``-compatible reward function.

    Signature: ``reward_fn(prompts, completions, *, label=None, gold_author=None,
    claimed_author=None, case_id=None, **kwargs) -> list[float]``. The dataset
    (``rl_dataset``) must carry the ``label`` / ``gold_author`` /
    ``claimed_author`` columns so they arrive via ``**kwargs``.

    ``cases`` is informational (kept for symmetry / future per-case gating);
    the reward is fully driven by the kwargs columns + the shared ``gate``.
    """
    the_gate = verifiers.provenance_faithful(records or {})
    del cases  # routed via kwargs; retained in signature for clarity

    def reward_fn(
        prompts: list,
        completions: list,
        *,
        label: Any = None,
        gold_author: Any = None,
        claimed_author: Any = None,
        case_id: Any = None,
        **kwargs: Any,
    ) -> list[float]:
        n = len(completions)
        labels = _as_list(label, n)
        golds = _as_list(gold_author, n)
        claimeds = _as_list(claimed_author, n)
        rewards: list[float] = []
        for i, comp in enumerate(completions):
            text = _completion_text(comp)
            case = {
                "label": labels[i] if i < len(labels) else "false",
                "gold_author": golds[i] if i < len(golds) else "",
                "claimed_author": claimeds[i] if i < len(claimeds) else None,
            }
            r, _ = reward_for_case(case, text, gate=the_gate)
            rewards.append(r)
        return rewards

    reward_fn.__name__ = "sophia_provenance_reward"
    return reward_fn
