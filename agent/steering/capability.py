"""Spec D — deterministic capability-retention scorer (pure stdlib).

Produces B's SSA capability inputs: does steering degrade reasoning? Accuracy
is answer-vs-gold; soundness reuses arithmetic_sound; coherence is a
deterministic degeneracy proxy (the failure mode high-alpha steering produces).
No model, no judge, no network.
"""
from __future__ import annotations

import re

from agent.verifiers import arithmetic_sound

_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_ARITH_SOUND = arithmetic_sound()
_MARKERS = ("answer is", "answer:", "answer =", "answer", "=")


def extract_final_number(text: str) -> "float | None":
    """The number the response commits to: the value after the last answer
    marker if present, else the last standalone number. None if none parseable."""
    if not text:
        return None
    low = text.lower()
    for marker in _MARKERS:
        idx = low.rfind(marker)
        if idx != -1:
            m = _NUM.search(text[idx + len(marker):])
            if m:
                return float(m.group())
    nums = _NUM.findall(text)
    return float(nums[-1]) if nums else None


def answer_correct(text: str, gold: float, *, tol: float = 1e-6) -> bool:
    got = extract_final_number(text)
    return got is not None and abs(got - gold) <= tol


def coherence_proxy(text: str) -> float:
    """Deterministic 0-100 coherence. Penalizes degeneracy: emptiness, immediate
    token repetition, low type-token diversity, pathological length."""
    t = (text or "").strip()
    if not t:
        return 0.0
    toks = t.split()
    if len(toks) < 2:
        return 40.0
    score = 100.0
    reps = sum(1 for i in range(1, len(toks)) if toks[i] == toks[i - 1])
    score -= 100.0 * reps / len(toks)
    ttr = len(set(toks)) / len(toks)
    if ttr < 0.5:
        score -= (0.5 - ttr) * 120.0
    if len(toks) > 200:
        score -= 20.0
    return max(0.0, min(100.0, score))


def score_response(text: str, gold: float) -> dict:
    # `sound` is a per-item diagnostic (did the response state any FALSE arithmetic?);
    # capability_cell aggregates only `correct`/`coherence` (the SSA inputs). `sound`
    # is carried for inspection/reporting, not folded into the cell verdict.
    return {
        "correct": answer_correct(text, gold),
        "sound": bool(_ARITH_SOUND(text or "", None, {})["passed"]),
        "coherence": coherence_proxy(text),
    }


def _accuracy(scored: "list[dict]") -> float:
    return round(sum(1 for s in scored if s["correct"]) / len(scored), 4) if scored else 0.0


def _mean_coh(scored: "list[dict]") -> float:
    return round(sum(s["coherence"] for s in scored) / len(scored), 2) if scored else 0.0


def capability_cell(base_scored: "list[dict]", steered_scored: "list[dict]") -> dict:
    """Assemble the SSA capability cell from per-item scores. capability_drop is
    the RELATIVE accuracy drop; retains mirrors ssa_verdict's capability check."""
    base_acc = _accuracy(base_scored)
    steer_acc = _accuracy(steered_scored)
    drop = max(0.0, (base_acc - steer_acc) / base_acc) if base_acc > 0 else 0.0
    coh = _mean_coh(steered_scored)
    return {
        "n": len(steered_scored),
        "base_accuracy": base_acc,
        "steered_accuracy": steer_acc,
        "capability_drop": round(drop, 4),
        "coherence": coh,
        "base_coherence": _mean_coh(base_scored),
        "retains": bool(drop < 0.05 and coh >= 75.0),
    }
