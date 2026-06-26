# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Surprise capture for the OKF belief-dynamics layer — a REAL, measured signal.

This is the piece that ``okf.belief_state_projection``'s HONESTY CONTRACT names as the
one missing signal: ``surprise`` — "how unexpected a belief is given current memory."
Until now it was an UNRECORDED PLACEHOLDER (``0.0``, "unmeasured, NOT unsurprising"), so
the decay/surprise-gating logic was a no-op over the real corpus. This module measures
it from the corpus content, honestly and deterministically, with no GPU and no fabrication.

==============================================================================
WHAT "surprise" MEANS HERE — read before trusting any score.
==============================================================================
Surprise = the per-token cross-entropy (negative log-likelihood, in nats) of a belief's
content under a smoothed term-likelihood model estimated from the REST of the corpus,
focused on the belief's provenance neighborhood (graph neighbours + same tradition /
domain) with a global-corpus backoff. High NLL == the rest of memory predicts this belief
poorly == novel / surprising. Low NLL == the belief is well-explained by what is already
known == a routine confirmation.

This is exactly the "retrieval/likelihood over the existing graph" that
``agi-proof/okf-consistency/RESEARCH_FOLLOWUP.md`` scopes as the honest predictive model
for this signal — NOT a neural model. It is named for what it is: a smoothed unigram /
retrieval-likelihood estimator. (The repo already learned, the hard way, not to call a
count table a "predictive world model" — see ``agent/tabular_transition_model.py``.)

HONESTY CAVEAT — the leave-one-out substitution (do not skip this):
The engram signal the literature describes is "surprise AT FIRST OBSERVATION": how
unexpected a belief was *when it arrived*, given memory *at that time*. Computing that
needs an arrival-time ordering of the corpus — i.e. ``written_at`` — which the OKF
frontmatter DOES NOT record (see RESEARCH_FOLLOWUP signal #1). So we do NOT claim the
temporal first-observation signal. Instead we compute a **leave-one-out** predictive
surprise: ``P(belief | the rest of memory)``, excluding the belief's own counts so it
cannot predict itself. That answers a related, weaker, but genuinely measured question —
"how redundant vs. novel is this belief given everything else in the corpus" — and it is
the most honest predictive surprise obtainable from a static, append-only, un-timestamped
corpus. When ``written_at`` lands, this same machinery can be restricted to the
arrival-time prefix to recover the true first-observation signal.

NORMALISED surprise is RELATIVE to this corpus. ``surprise == 0.5`` means "about as
predictable as the average belief"; ``> 0.5`` means "less predictable than average"
(more surprising). The mapping is a standard z-score -> logistic squash with no free
parameters tuned to produce a particular outcome — whatever falls out, falls out. The
raw per-token NLL is always reported alongside so a reader is never forced to trust the
squash.
==============================================================================
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

# Tokeniser: ASCII word/number runs of length >= 2, lower-cased. Deterministic and
# dependency-free. The corpus bodies are predominantly English prose; CJK appears mostly
# in titles and is intentionally not modelled (it would add noise, not signal, to a
# unigram term model). This is a documented, honest limitation, not a hidden one.
_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")

# Interpolation weight: how much to trust the local provenance neighbourhood vs. the
# global corpus background. 0.7 leans on the neighbourhood (a belief is "surprising"
# chiefly relative to its own context) while keeping a global backoff so an isolated
# node still gets a defined score. Fixed and documented — NOT tuned to a target result.
DEFAULT_LAMBDA_LOCAL = 0.7
# Additive (Lidstone) smoothing mass per vocabulary item — handles out-of-context terms
# without driving any probability to zero (which would make NLL infinite).
DEFAULT_ADD_K = 1.0


@dataclass(frozen=True)
class SurpriseScore:
    """One belief's measured surprise. ``raw_nll`` is the primary, un-squashed measure."""
    node_id: str
    raw_nll: float            # per-token cross-entropy under the rest-of-corpus model (nats)
    surprise: float           # raw_nll squashed to (0,1), RELATIVE to the corpus (0.5 == average)
    token_count: int          # number of modelled tokens in this belief's body
    neighborhood_size: int    # provenance neighbours that informed the local model (0 == global-only)

    def as_dict(self) -> dict:
        return {
            "nodeId": self.node_id,
            "rawNll": round(self.raw_nll, 6),
            "surprise": round(self.surprise, 6),
            "tokenCount": self.token_count,
            "neighborhoodSize": self.neighborhood_size,
        }


def tokenize(text: str) -> "list[str]":
    """Deterministic ASCII-word tokeniser (lower-cased, length >= 2). See module note."""
    return _TOKEN_RE.findall((text or "").lower())


def _neighbors(graph, node_id: str) -> "set[str]":
    """Provenance neighbourhood of one node: resolved out-links + in-links + same
    tradition + same domain. The belief itself is never included (leave-one-out)."""
    from okf.graph import out_link_targets, resolve

    if graph is None or node_id not in getattr(graph, "nodes", {}):
        return set()
    nbrs: set[str] = set()
    node = graph.nodes[node_id]
    # out-links (body [[wikilinks]] + frontmatter `links`)
    for t in out_link_targets(node):
        rid = resolve(graph, t)
        if rid and rid != node_id:
            nbrs.add(rid)
    meta = node["meta"]
    tradition = meta.get("tradition")
    domain = meta.get("domain")
    for other_id, other in graph.nodes.items():
        if other_id == node_id:
            continue
        om = other["meta"]
        # in-links: another page that links to us
        if node_id in {resolve(graph, t) for t in out_link_targets(other)}:
            nbrs.add(other_id)
        # shared provenance context
        if tradition and om.get("tradition") == tradition:
            nbrs.add(other_id)
        if domain and om.get("domain") == domain:
            nbrs.add(other_id)
    nbrs.discard(node_id)
    return nbrs


def corpus_surprise(
    pages,
    *,
    graph=None,
    lambda_local: float = DEFAULT_LAMBDA_LOCAL,
    add_k: float = DEFAULT_ADD_K,
) -> "dict[str, SurpriseScore]":
    """Measure leave-one-out predictive surprise for every page. Deterministic.

    ``pages`` is an iterable of ``okf.page.Page``. ``graph`` (optional) is an
    ``okf.graph.Graph`` over the same pages; when supplied, surprise is focused on each
    belief's provenance neighbourhood, else it is purely global. Returns ``{id: score}``.
    """
    from okf.graph import build as build_graph

    pages = list(pages)
    if graph is None and pages:
        graph = build_graph(pages)

    # Per-page token counts and the global background (sum of all pages).
    tokens_by_id: dict[str, list[str]] = {}
    counts_by_id: dict[str, Counter] = {}
    global_counts: Counter = Counter()
    for p in pages:
        toks = tokenize(p.body)
        tokens_by_id[p.id] = toks
        c = Counter(toks)
        counts_by_id[p.id] = c
        global_counts.update(c)
    vocab_size = max(1, len(global_counts))
    n_global = sum(global_counts.values())

    raw_by_id: dict[str, float] = {}
    nbrhood_by_id: dict[str, int] = {}
    for p in pages:
        nid = p.id
        toks = tokens_by_id[nid]
        if not toks:
            continue  # nothing to model; excluded from normalisation, scored as average below
        c_self = counts_by_id[nid]

        # Global model with THIS belief left out (it must not predict itself).
        g_minus = global_counts.copy()
        g_minus.subtract(c_self)
        n_global_minus = max(0, n_global - len(toks))

        # Local provenance-neighbourhood model (already excludes self).
        nbrs = _neighbors(graph, nid)
        local_counts: Counter = Counter()
        for o in nbrs:
            local_counts.update(counts_by_id.get(o, Counter()))
        n_local = sum(local_counts.values())
        nbrhood_by_id[nid] = len(nbrs)
        lam = lambda_local if n_local > 0 else 0.0

        denom_g = n_global_minus + add_k * vocab_size
        denom_l = n_local + add_k * vocab_size

        nll_sum = 0.0
        for t in toks:  # document order is fixed -> deterministic
            p_global = (g_minus.get(t, 0) + add_k) / denom_g
            if lam > 0.0:
                p_local = (local_counts.get(t, 0) + add_k) / denom_l
                p_t = lam * p_local + (1.0 - lam) * p_global
            else:
                p_t = p_global
            nll_sum += -math.log(p_t)
        raw_by_id[nid] = nll_sum / len(toks)

    # Normalise raw per-token NLL to (0,1) via z-score -> logistic. Relative to THIS
    # corpus: 0.5 == as predictable as the average belief. No free parameter tuned to a
    # target. Pages with no modellable tokens get the neutral 0.5 (honestly "unscored").
    scored_ids = sorted(raw_by_id)  # sorted for byte-reproducible mean/std accumulation
    out: dict[str, SurpriseScore] = {}
    if scored_ids:
        vals = [raw_by_id[i] for i in scored_ids]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var)
    else:
        mean = std = 0.0

    for p in pages:
        nid = p.id
        toks = tokens_by_id[nid]
        if nid not in raw_by_id:
            out[nid] = SurpriseScore(nid, raw_nll=mean, surprise=0.5,
                                     token_count=len(toks), neighborhood_size=0)
            continue
        raw = raw_by_id[nid]
        z = (raw - mean) / std if std > 1e-12 else 0.0
        surprise = 1.0 / (1.0 + math.exp(-z))
        out[nid] = SurpriseScore(nid, raw_nll=raw, surprise=surprise,
                                 token_count=len(toks),
                                 neighborhood_size=nbrhood_by_id.get(nid, 0))
    return out


def surprise_by_id(pages, *, graph=None) -> "dict[str, float]":
    """Convenience: ``{node_id: normalised_surprise}`` for projection into BeliefState."""
    return {nid: s.surprise for nid, s in corpus_surprise(pages, graph=graph).items()}


__all__ = [
    "SurpriseScore",
    "tokenize",
    "corpus_surprise",
    "surprise_by_id",
    "DEFAULT_LAMBDA_LOCAL",
    "DEFAULT_ADD_K",
]
