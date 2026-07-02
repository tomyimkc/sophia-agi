# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""NLI-entailment backend for the fact-check gate's `EntailmentFn` seam.

Validated primitive (agi-proof/benchmark-results/coherence-reframes/NLI-ENTAILMENT.public-report.json):
textual entailment decisively beats the coherence/lexical baseline at telling SUPPORTING from
REFUTING evidence — FEVER n=400 AUROC 0.962 vs coherence 0.650 (paired Δ+0.31, CI excludes 0, 3
seeds), because coherence/similarity treats refuting evidence as "topical" while NLI does not.

This packages that primitive as a drop-in `agent.fact_check_gate.EntailmentFn`:
    (AtomicClaim, EvidenceSource) -> "entails" | "contradicts" | "irrelevant"
so a caller can do `external_ground(claim, retriever, entailment=build_nli_entailment())`. It
changes NO defaults — with no backend injected the gate keeps its conservative lexical screen.

Honest bounds (this is a MECHANISM, not a solved-verification claim): the default model is a
purpose-built NLI specialist; validation used clean gold evidence (real retrieval is noisier).
Fail-closed: if the model can't load, `build_nli_entailment` raises so the caller falls back to
the lexical screen (never a silent, over-confident pass). The scorer is injectable for
deterministic, download-free tests.
"""
from __future__ import annotations

from typing import Callable

# scorer(premise, hypothesis) -> (p_contradiction, p_entailment, p_neutral)
Scorer = Callable[[str, str], "tuple[float, float, float]"]

_DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-base"  # trained on MNLI/SNLI, NOT FEVER (fair)
_CACHE: dict[str, Scorer] = {}


def _load_cross_encoder(model_name: str) -> Scorer:
    """Load a cross-encoder NLI model as a scorer. Raises (fail-closed) if unavailable."""
    if model_name in _CACHE:
        return _CACHE[model_name]
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import numpy as np
    import scipy.special as sp
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)

    def scorer(premise: str, hypothesis: str) -> tuple[float, float, float]:
        logits = model.predict([(premise, hypothesis)])
        p = sp.softmax(np.atleast_2d(logits), axis=1)[0]  # cols: [contradiction, entailment, neutral]
        return float(p[0]), float(p[1]), float(p[2])

    _CACHE[model_name] = scorer
    return scorer


def build_nli_entailment(
    scorer: Scorer | None = None,
    *,
    model_name: str = _DEFAULT_MODEL,
    entail_threshold: float = 0.5,
    contradict_threshold: float = 0.5,
):
    """Return an `EntailmentFn` (claim, source) -> entails|contradicts|irrelevant.

    Conservative / fail-closed: a label is only 'entails'/'contradicts' when that class is the
    argmax AND clears its threshold; otherwise 'irrelevant' (so the gate abstains rather than
    over-admitting). Empty evidence -> 'irrelevant'. `scorer` is injectable for tests.
    """
    score = scorer or _load_cross_encoder(model_name)

    def entail(claim, source) -> str:
        premise = f"{getattr(source, 'title', '')} {getattr(source, 'snippet', '')}".strip()
        hypothesis = getattr(claim, "text", "") or ""
        if not premise or not hypothesis:
            return "irrelevant"
        p_contra, p_entail, _p_neutral = score(premise, hypothesis)
        top = max(p_contra, p_entail, _p_neutral)
        if p_entail == top and p_entail >= entail_threshold:
            return "entails"
        if p_contra == top and p_contra >= contradict_threshold:
            return "contradicts"
        return "irrelevant"

    return entail


def build_hybrid_entailment(lexical_fn, scorer: Scorer | None = None, *,
                            model_name: str = _DEFAULT_MODEL, contradict_threshold: float = 0.5):
    """Contradiction-only hybrid EntailmentFn: admission stays on the cheap `lexical_fn`; NLI adds
    ONLY contradiction-catching.

    Motivated by the acceptance-gate NO-GO: NLI *entailment* over-abstains, but the lexical screen's
    negation cues + NLI's strong *contradiction* detection are the load-bearing signal. So:
      label = 'contradicts'  iff NLI detects a strong contradiction (argmax + threshold)
      else   = lexical_fn(claim, source)   (the incumbent admission screen drives entail/irrelevant)
    This cannot introduce NLI's admission over-abstention (lexical drives admission); it only lets
    NLI reject false claims the lexical screen would otherwise admit. `lexical_fn` is injected to keep
    this module decoupled from fact_check_gate.
    """
    score = scorer or _load_cross_encoder(model_name)

    def entail(claim, source) -> str:
        premise = f"{getattr(source, 'title', '')} {getattr(source, 'snippet', '')}".strip()
        hypothesis = getattr(claim, "text", "") or ""
        if premise and hypothesis:
            p_contra, p_entail, p_neutral = score(premise, hypothesis)
            if p_contra == max(p_contra, p_entail, p_neutral) and p_contra >= contradict_threshold:
                return "contradicts"
        return lexical_fn(claim, source)

    return entail


# --- healthy semantic-similarity incumbent (for a fair NLI-vs-coherence comparison) ---
_BIENCODER: dict = {}


def _load_bi_encoder(model_name: str):
    if model_name in _BIENCODER:
        return _BIENCODER[model_name]
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer, util
    m = SentenceTransformer(model_name)

    def cos(a: str, b: str) -> float:
        ea = m.encode(a, normalize_embeddings=True)
        eb = m.encode(b, normalize_embeddings=True)
        return float(util.cos_sim(ea, eb))

    _BIENCODER[model_name] = cos
    return cos


def build_semantic_entailment(cos_fn=None, *, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                              threshold: float = 0.45):
    """A HEALTHY coherence/similarity incumbent EntailmentFn (the fair baseline for NLI).

    The token-overlap lexical screen COLLAPSES on non-verbatim evidence (FEVER paraphrase coverage
    0.003; sophia hedged provenance coverage 0.0), making NLI-vs-lexical unfalsifiable. A semantic
    similarity admission fires on topical evidence (healthy coverage), so NLI-vs-semantic isolates the
    mechanism question: does ENTAILMENT beat SIMILARITY as a grounding signal? Returns 'entails' iff
    cos(claim, evidence) >= threshold, else 'irrelevant' (similarity cannot detect contradiction — which
    is exactly the limitation NLI is hypothesized to fix). cos_fn injectable for tests.
    """
    cos = cos_fn or _load_bi_encoder(model_name)

    def entail(claim, source) -> str:
        premise = f"{getattr(source, 'title', '')} {getattr(source, 'snippet', '')}".strip()
        hypothesis = getattr(claim, "text", "") or ""
        if not premise or not hypothesis:
            return "irrelevant"
        return "entails" if cos(hypothesis, premise) >= threshold else "irrelevant"

    return entail
