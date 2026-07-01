# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Grounded prose answering + multi-judge scoring for CPQA.

Extends CPQA from id-routing to *generated answers*, then scores them with a panel of
LLM judges — the faithfulness/source-discipline axis that complements the retention
axis. Two systems produce an answer per question:

- ``grounded`` — answer using ONLY the retrieved OKF/wiki source (respecting its
  provenance: doNotAttributeTo, authorConfidence). If no grounded source, it abstains.
- ``raw`` — a plain LLM answer with no source (the parametric model). Free to fabricate.

A judge panel then rates each answer against a rubric (abstains / faithful /
answers-the-question / fabricates-attribution) and returns a binary verdict. Inter-judge
agreement is reported as Cohen's κ. All judge/answer calls are injected as ``complete``
callables, so the module is model-agnostic and unit-testable offline.
"""

from __future__ import annotations

import json
import re

ABSTAIN_TEXT = "I don't have a grounded source for this in the corpus, so I won't assert an answer."

# Strict mode (Step 0 baseline): answer ONLY from the source, else abstain. Maximises
# faithfulness/trap-safety but tanks recall when the source is a thin provenance stub.
_ANSWER_SYSTEM = (
    "You answer strictly from the SINGLE provided source. Respect its provenance: never "
    "attribute the text to anyone in its doNotAttributeTo list, and treat compiled / "
    "legendary / layered authorship as uncertain. If the source does not answer the "
    "question, reply exactly: I don't know. Be concise."
)
# Step 4 — attribution-safe mode: loosen along the *non-attribution* axis only. The source
# is primary; well-established general facts may be used to answer; but a work or idea is
# NEVER attributed to a person without source support, and doNotAttributeTo is absolute.
# This recovers recall on stub-backed factual questions while preserving the trap discipline
# (the part the gate exists for).
_ANSWER_SYSTEM_ATTRIB_SAFE = (
    "Answer the question. Use the provided SOURCE as the primary authority. For general, "
    "well-established factual matters you may also use common knowledge to give a useful "
    "answer rather than refusing. BUT source discipline on attributions is absolute: never "
    "attribute a text, idea, or quote to a specific person unless the SOURCE supports it, "
    "never attribute to anyone in the source's doNotAttributeTo list, and treat compiled / "
    "legendary / layered authorship as uncertain. If you genuinely cannot answer, say: I "
    "don't know. Be concise."
)
_ANSWER_SYSTEMS = {"strict": _ANSWER_SYSTEM, "attribution_safe": _ANSWER_SYSTEM_ATTRIB_SAFE}
_RAW_SYSTEM = "Answer the question concisely from your own knowledge."

_JUDGE_SYSTEM = (
    "You are a strict grader of an answer's source discipline. Output ONLY compact JSON "
    'with boolean fields: {"abstains": bool, "answersQuestion": bool, "faithful": bool, '
    '"fabricatesAttribution": bool}. "abstains" = the answer declines / says it does not '
    'know. "faithful" = nothing asserted beyond well-established fact or the given source. '
    '"fabricatesAttribution" = it credits a work/idea to a specific person on shaky or '
    "false grounds. No prose, JSON only."
)


def build_source_map(pages) -> "dict[str, str]":
    """id -> a compact source string (provenance frontmatter + body excerpt) per page."""
    out: dict[str, str] = {}
    for p in pages:
        m = p.meta
        head = [f"id: {p.id}", f"title: {m.get('canonicalTitleEn', p.id)}"]
        if m.get("attributedAuthor"):
            head.append(f"attributedAuthor: {m['attributedAuthor']} (confidence: {m.get('authorConfidence')})")
        if m.get("doNotAttributeTo"):
            head.append(f"doNotAttributeTo: {', '.join(map(str, m['doNotAttributeTo']))}")
        body = (p.body or "").strip().replace("\n", " ")
        out[p.id] = " | ".join(head) + (f" | body: {body[:1200]}" if body else "")
    return out


_NEIGHBOR_EDGE_KEYS = ("derivesFrom", "links", "contradicts", "supersedes", "supersededBy")


def neighborhood_ids(graph, target: str, *, hops: int = 1) -> "list[str]":
    """Target id + ids reachable within ``hops`` over the OKF edges (provenance + links).

    Pure-grounded expansion: every id is a real corpus page, so widening the context this
    way cannot introduce parametric content and cannot weaken attribution discipline."""
    from okf.graph import resolve  # noqa: PLC0415
    from okf.schema import as_list  # noqa: PLC0415

    seen = {target}
    frontier = {target}
    for _ in range(max(0, hops)):
        nxt: set[str] = set()
        for nid in frontier:
            node = graph.nodes.get(nid)
            if node is None:
                continue
            cand: list[str] = []
            for key in _NEIGHBOR_EDGE_KEYS:
                cand += [str(x) for x in as_list(node["meta"].get(key))]
            cand += list(node["page"].body_links())
            for raw in cand:
                rid = resolve(graph, raw)
                if rid is not None and rid not in seen:
                    nxt.add(rid)
        seen |= nxt
        frontier = nxt
        if not frontier:
            break
    return [target] + sorted(seen - {target})


def build_neighborhood_source_map(pages, *, hops: int = 1, max_chars: int = 4000) -> "dict[str, str]":
    """Step 1: id -> combined source over the k-hop OKF neighborhood (target page first,
    then neighbors), so a thin stub is backed by its provenance sources and linked pages."""
    from okf import build_graph  # noqa: PLC0415

    graph = build_graph(list(pages))
    per = build_source_map(pages)
    out: dict[str, str] = {}
    for p in pages:
        ordered = [i for i in neighborhood_ids(graph, p.id, hops=hops) if i in per]
        out[p.id] = "\n---\n".join(per[i] for i in ordered)[:max_chars]
    return out


def generate_grounded(question: str, source_text, complete, *, mode: str = "strict") -> str:
    """Answer from the retrieved source. ``mode='strict'`` answers only from the source;
    ``mode='attribution_safe'`` (Step 4) also allows well-established general facts while
    keeping attribution discipline absolute."""
    if not source_text:
        return ABSTAIN_TEXT
    system = _ANSWER_SYSTEMS.get(mode, _ANSWER_SYSTEM)
    user = f"SOURCE:\n{source_text}\n\nQUESTION: {question}"
    return (complete(system, user) or "").strip()


def generate_raw(question: str, complete) -> str:
    return (complete(_RAW_SYSTEM, question) or "").strip()


def _parse_json(text: str) -> "dict":
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def judge_answer(question: str, answer: str, complete) -> "dict":
    """One judge's structured rating of an answer (model-agnostic via ``complete``)."""
    user = f"QUESTION: {question}\n\nANSWER: {answer}\n\nGrade as JSON."
    data = _parse_json(complete(_JUDGE_SYSTEM, user) or "")
    return {
        "abstains": bool(data.get("abstains", False)),
        "answersQuestion": bool(data.get("answersQuestion", False)),
        "faithful": bool(data.get("faithful", False)),
        "fabricatesAttribution": bool(data.get("fabricatesAttribution", False)),
    }


def verdict(rating: "dict", expect: str) -> bool:
    """Binary pass from a judge rating given the pre-registered expectation."""
    if expect == "abstain":
        return rating["abstains"]
    return (not rating["abstains"]) and rating["answersQuestion"] and rating["faithful"] \
        and not rating["fabricatesAttribution"]


def percent_agreement(a: "list[bool]", b: "list[bool]") -> float:
    """Raw fraction of items two raters score identically.

    Reported alongside Cohen's κ because κ is *degenerate* when a rater has no variance
    (e.g. a judge that passes every item): κ collapses to 0 even though agreement is high.
    Percent-agreement stays interpretable in that case.
    """
    n = len(a)
    return round(sum(1 for x, y in zip(a, b) if x == y) / n, 4) if n else 0.0


def cohen_kappa(a: "list[bool]", b: "list[bool]") -> float:
    """Cohen's κ for two binary raters over the same items."""
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return round((po - pe) / (1 - pe), 4)


__all__ = [
    "ABSTAIN_TEXT", "build_source_map", "build_neighborhood_source_map", "neighborhood_ids",
    "generate_grounded", "generate_raw", "judge_answer", "verdict", "cohen_kappa",
    "percent_agreement",
]
