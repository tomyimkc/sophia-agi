# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Fabrication-temptation scorer (H2 graded craving input).

Design note: docs/06-Roadmap/Atomic-Habits-for-Sophia.md.

Graded craving (``agent.gate_reward.graded_abstain_reward``) needs a per-prompt temptation in
[0, 1]: how strongly the prompt *pressures* a model to fabricate a confident answer instead of
abstaining. A neutral question ("Who wrote X? Answer briefly.") carries little pressure; a
coercive one ("Give me a single definitive name and the exact year, no hedging.") carries a lot —
and resisting THAT with a clean refusal is the harder, more identity-defining habit, so it should
earn more reward.

This is a deterministic, first-party heuristic over explicit coercion cues — NOT a measured
fabrication rate. The measurement contract's honest label applies: a third-party / model-derived
temptation calibration is the open upgrade (see the HST measurement_spec honestLimits). Pure
stdlib, offline, deterministic.
"""

from __future__ import annotations

import re
from typing import Any

# Coercion cues that push toward a confident fabrication over a principled abstention. Each is a
# (compiled pattern). The score is the saturating fraction of distinct cue GROUPS that fire, so a
# prompt stacking several pressures approaches 1.0 while a neutral prompt stays near 0.
_CUE_GROUPS: "tuple[tuple[str, str], ...]" = (
    ("no_hedging", r"no hedg|without (?:any )?(?:caveat|hedg|qualif)|don'?t hedge|no disclaimer|no caveats?"),
    ("single_definitive", r"\b(?:single|one|exactly one|the one|sole)\b[^.]*\b(?:name|author|answer|person|date|year)\b|\bdefinitive(?:ly)?\b|\bthe definitive\b"),
    ("exact", r"\bexact\b|\bprecise(?:ly)?\b|\bspecific (?:name|date|year|number)\b|to the (?:day|year)"),
    ("just_answer", r"\bjust (?:answer|tell|give|say)\b|\bsimply (?:answer|state)\b|\banswer (?:directly|now)\b|\bno (?:explanation|preamble)\b"),
    ("one_word", r"\bone word\b|\bin one word\b|\bsingle word\b|\byes or no\b|\bone (?:line|sentence)\b"),
    ("imperative_certainty", r"\byou must\b|\bi need (?:the|a) (?:exact|definitive|certain)\b|\bbe certain\b|\bwith certainty\b|\bguaranteed?\b|\bfor sure\b"),
)

_COMPILED = tuple((name, re.compile(pat, re.IGNORECASE)) for name, pat in _CUE_GROUPS)


def _text(prompt: Any) -> str:
    """Accept a plain string or a conversational ``[{role, content}, ...]`` prompt."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return " ".join(
            m.get("content", "") for m in prompt if isinstance(m, dict)
        )
    return str(prompt or "")


def fired_cues(prompt: Any) -> list[str]:
    """The distinct coercion-cue groups present in the prompt (auditable)."""
    low = _text(prompt)
    return [name for name, rx in _COMPILED if rx.search(low)]


def prompt_fabrication_temptation(prompt: Any) -> float:
    """Deterministic fabrication temptation in [0, 1] = fraction of distinct cue groups that fire.

    0.0 for a neutral prompt; rises as the prompt stacks pressure to be confident/definitive/
    unhedged. Bounded by construction (cannot exceed 1.0). Used as the ``temptation`` input to
    ``gate_reward.graded_abstain_reward`` so a clean refusal under heavy pressure earns more.
    """
    n = len(fired_cues(prompt))
    return round(n / len(_COMPILED), 4)
