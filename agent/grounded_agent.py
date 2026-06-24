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
from agent.grounded_answer_policy import answer_with_policy
from tools.audit_cpqa_recall import classify_source


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
                    hops: int = 1, attribution_check=None) -> "dict[str, Any]":
    """Answer ``question`` from the OKF corpus via routing + the gated hybrid policy.

    Returns {answer, policy, target, gated}. ``policy`` is one of: abstain_no_route (the
    controller found no page), abstain_no_source, grounded_strict, grounded_fallback,
    fallback_gated_abstain.
    """
    controller = controller or LexicalController()
    target = controller.route(question, vocab_for_pages(pages))
    if target is None:
        return {"answer": ABSTAIN_TEXT, "policy": "abstain_no_route", "target": None, "gated": False}

    source_map = (build_neighborhood_source_map(pages, hops=hops)
                  if retrieval == "neighborhood" else build_source_map(pages))
    by_id = {p.id: p for p in pages}
    answer_bearing = classify_source(by_id[target])["answerBearing"] if target in by_id else False
    out = answer_with_policy(question, source_map.get(target), complete,
                             answer_bearing=answer_bearing, attribution_check=attribution_check)
    out["target"] = target
    return out


__all__ = ["grounded_answer", "vocab_for_pages"]
