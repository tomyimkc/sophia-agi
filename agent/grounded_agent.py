# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Grounded answering for the Sophia agent — CPQA's gated hybrid, end to end.

Ties the pieces validated in the CPQA benchmark into one runtime call the agent can use:

    question --(controller routes)--> OKF page --(retrieve source)--> answer_with_policy

- routing: a controller picks the OKF page the question is about (None -> abstain);
- retrieval: the page's source (optionally its k-hop neighborhood);
- answering: the typed hybrid gate — strict if the source is answer-bearing, attribution-
  safe parametric fallback if thin, and the fallback is verified by Sophia's real
  attribution gate (fail-closed). Traps with no routed page never reach a model call.

Pure orchestration over already-tested modules; the LLM is injected as ``complete`` so
this is offline-testable and provider-agnostic.
"""

from __future__ import annotations

from typing import Any

from agent.continual_qa_answer import (
    ABSTAIN_TEXT, build_neighborhood_source_map, build_source_map,
)
from agent.continual_qa_controller import LexicalController
from agent.continual_qa_hybrid import FALLBACK, STRICT
from agent.graded_decision import answer_confidence, decide
from agent.grounded_answer_policy import answer_with_policy
from tools.audit_cpqa_recall import classify_source

#: Policies that produced a real, gate-cleared answer (a "pass" for the graded router).
#: Everything else (the abstain_* / fallback_gated_abstain labels) is already fail-closed.
_PASSED_POLICIES = frozenset({STRICT, FALLBACK})


def apply_graded_decision(out, *, confidence=None, corroboration_evidence=None,
                          self_consistency_samples=None, thresholds=None) -> "dict[str, Any]":
    """Overlay the calibrated answer/hedge/abstain router on a policy result, in place.

    Wires the previously-unwired :func:`agent.graded_decision.decide` into the live path:
    it maps ``(gate_passed, confidence)`` onto ``answer | hedge | abstain`` and we apply it
    **downgrade-only** — graded can make an answer *more* conservative, never less:

      - a gate-passing answer whose confidence ``< lo`` becomes an **abstain** (a
        low-confidence pass is suspicious — the gate may have missed something);
      - a gate-passing answer whose confidence is in ``[lo, hi)`` is **hedged** (the
        answer is surfaced, flagged low-confidence, the original kept in ``rawAnswer``);
      - a high-confidence pass is left untouched.

    An already-abstaining policy is **never** upgraded (fail-closed): the router's
    high-confidence-near-miss ``hedge`` branch cannot resurrect an answer the policy
    already suppressed, because that text was never returned.

    Confidence source (precedence): explicit ``confidence`` float > ``answer_confidence``
    over ``corroboration_evidence`` / ``self_consistency_samples``. When **no** confidence
    signal is supplied, grading is skipped (``applied: False``) so opting in without a
    signal is a guaranteed no-op — zero drift for existing callers/benchmarks.
    """
    has_signal = (confidence is not None or bool(corroboration_evidence)
                  or bool(self_consistency_samples))
    if not has_signal:
        out["graded"] = {"applied": False, "reason": "no confidence signal supplied"}
        return out

    conf = (float(confidence) if confidence is not None
            else answer_confidence(corroboration_evidence, self_consistency_samples))
    gate_passed = out.get("policy") in _PASSED_POLICIES
    d = decide(gate_passed=gate_passed, confidence=conf, thresholds=thresholds)
    out["graded"] = {"applied": True, "action": d["action"], "confidence": d["confidence"],
                     "reason": d["reason"], "thresholds": d["thresholds"]}

    # Downgrade-only: never touch an already-fail-closed (abstaining) result.
    if not gate_passed:
        return out
    if d["action"] == "abstain":
        out["rawAnswer"] = out["answer"]
        out["answer"] = ABSTAIN_TEXT
        out["policy"] = "graded_abstain_low_confidence"
    elif d["action"] == "hedge":
        out["rawAnswer"] = out["answer"]
        out["answer"] = f"(low confidence) {out['answer']}"
        out["policy"] = f"{out['policy']}_hedged"
    return out


def vocab_for_pages(pages) -> "dict[str, str]":
    """id -> searchable text (id words + title + aliases + type/domain tag) for routing."""
    vocab: dict[str, str] = {}
    for p in pages:
        parts = [p.id.replace("_", " ")]
        title = p.meta.get("canonicalTitleEn")
        if title:
            parts.append(str(title))
        for alias in p.meta.get("aliases", []) or []:
            parts.append(str(alias).replace("_", " "))
        tag = " ".join(str(p.meta.get(k)) for k in ("pageType", "domain") if p.meta.get(k))
        vocab[p.id] = " ".join(parts) + (f" [{tag}]" if tag else "")
    return vocab


def grounded_answer(question: str, complete, *, pages, controller=None, retrieval: str = "single",
                    hops: int = 1, attribution_check=None, gap_log_path=None,
                    graded: bool = False, confidence=None, corroboration_evidence=None,
                    self_consistency_samples=None, thresholds=None) -> "dict[str, Any]":
    """Answer ``question`` from the OKF corpus via routing + the gated hybrid policy.

    Returns {answer, policy, target, gated}. ``policy`` is one of: abstain_no_route (the
    controller found no page), abstain_no_source, grounded_strict, grounded_fallback,
    fallback_gated_abstain. If ``gap_log_path`` is set, knowledge gaps (anything but a clean
    grounded answer) are appended there to feed the self-improving corpus worklist.

    When ``graded=True`` the calibrated answer/hedge/abstain router
    (:func:`apply_graded_decision`) is overlaid on the result: a gate-passing answer is
    downgraded to a hedge or abstain when its confidence is low, and an ``out["graded"]``
    block records the decision. It is **downgrade-only and fail-closed** (never upgrades an
    abstain). Supply a confidence signal via ``confidence`` (a float in ``[0,1]``),
    ``corroboration_evidence`` (``agent.corroboration.Evidence`` list), or
    ``self_consistency_samples`` (sampled answers); with no signal, grading is a no-op so
    existing callers and benchmarks see zero drift. ``thresholds`` overrides the ``hi``/``lo``
    cut points (default ``{"hi": 0.7, "lo": 0.4}``).
    """
    controller = controller or LexicalController()
    target = controller.route(question, vocab_for_pages(pages))
    if target is None:
        out = {"answer": ABSTAIN_TEXT, "policy": "abstain_no_route", "target": None, "gated": False}
    else:
        source_map = (build_neighborhood_source_map(pages, hops=hops)
                      if retrieval == "neighborhood" else build_source_map(pages))
        by_id = {p.id: p for p in pages}
        answer_bearing = classify_source(by_id[target])["answerBearing"] if target in by_id else False
        out = answer_with_policy(question, source_map.get(target), complete,
                                 answer_bearing=answer_bearing, attribution_check=attribution_check)
        out["target"] = target

    if graded:
        apply_graded_decision(out, confidence=confidence,
                              corroboration_evidence=corroboration_evidence,
                              self_consistency_samples=self_consistency_samples,
                              thresholds=thresholds)

    if gap_log_path is not None:
        from agent.knowledge_gap_log import log_gap  # noqa: PLC0415
        log_gap(question, target=out.get("target"), policy=out["policy"], path=gap_log_path)
    return out


__all__ = ["grounded_answer", "vocab_for_pages", "apply_graded_decision"]
