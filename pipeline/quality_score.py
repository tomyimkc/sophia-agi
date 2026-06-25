# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Provenance-aware document quality scoring (Phase 1).

This is the pipeline's differentiator: instead of a black-box "quality classifier", a
document's training-worthiness is scored by *composing Sophia's existing trust layer* with
cheap, deterministic content heuristics. The score then drives crawl prioritization
(`pipeline.link_priority`) and corpus filtering — the JD's
"根据数据清洗反馈信号设计全网数据选取 ... 链接质量预估 ... 数据优先级".

Two signal families, both offline and stdlib-only:

  1. PROVENANCE (reused, not reinvented):
     - ``agent.poison_resistant_ingestion.assess_item`` — trust-shrunk, k-independent
       corroboration pooling. A single low-trust source can never alone clear the bar.
     - ``agent.grounded_confidence.AUTHOR_CONFIDENCE_PRIOR`` — maps an OKF authorConfidence
       tier (consensus/disputed/legendary/...) to a prior on being well-grounded.

  2. CONTENT HEURISTICS (cheap, language-aware): length adequacy, alphanumeric ratio,
     boilerplate/spam markers, script purity, and lexical repetition.

Fail-closed by design: a document with NO provenance is capped below the top tier (it can be
*good* but not *trusted*), and a document whose sources are quarantined by the poison gate is
penalized. ``score_document`` is pure and deterministic — same input, same score.
"""

from __future__ import annotations

import re

from agent.grounded_confidence import AUTHOR_CONFIDENCE_PRIOR, _DEFAULT_PRIOR
from agent.poison_resistant_ingestion import SourceTrust, assess_item
from pipeline import document as _doc

# --------------------------------------------------------------------------- #
# Content heuristics — each returns a sub-score in [0,1], higher == better.
# --------------------------------------------------------------------------- #

_WORD_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
_CJK_RE = re.compile(r"[一-鿿]")
_LATIN_RE = re.compile(r"[a-zA-Z]")
# Spam / boilerplate markers common on low-value pages.
_BOILERPLATE_MARKERS = (
    "click here",
    "buy now",
    "sign up",
    "subscribe",
    "accept all cookies",
    "% off",
    "limited time",
    "act now",
    "$$$",
)
#: Documents below this many word-tokens are too short to be useful pretraining text.
_MIN_TOKENS_FULL_CREDIT = 120


def _tokens(text: str) -> list[str]:
    """Word tokens: latin/numeric runs plus each CJK character counted individually."""
    return _WORD_RE.findall(text.lower()) + _CJK_RE.findall(text)


def length_score(text: str) -> float:
    """Saturating credit for length: full credit at >= ~120 tokens."""
    n = len(_tokens(text))
    return round(min(1.0, n / _MIN_TOKENS_FULL_CREDIT), 6)


def alpha_ratio_score(text: str) -> float:
    """Fraction of non-space chars that are alphanumeric/CJK (spam pages are symbol-heavy)."""
    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return 0.0
    good = sum(1 for c in non_space if c.isalnum())
    return round(good / len(non_space), 6)


def boilerplate_score(text: str) -> float:
    """1.0 == clean; drops as spam markers and exclamation runs accumulate."""
    low = text.lower()
    hits = sum(low.count(m) for m in _BOILERPLATE_MARKERS)
    hits += len(re.findall(r"!{2,}", text))  # "!!!" runs
    return round(max(0.0, 1.0 - hits / 4.0), 6)


def script_purity_score(text: str) -> float:
    """Dominant-script fraction among letters (mixed-script junk scores lower)."""
    latin = len(_LATIN_RE.findall(text))
    cjk = len(_CJK_RE.findall(text))
    total = latin + cjk
    if total == 0:
        return 0.0
    return round(max(latin, cjk) / total, 6)


def repetition_score(text: str) -> float:
    """Unique-token ratio (repeated keyword stuffing scores lower)."""
    toks = _tokens(text)
    if not toks:
        return 0.0
    return round(len(set(toks)) / len(toks), 6)


#: (sub-score fn, weight). Weights sum to 1. ``script_purity`` is trivially ~1.0 for any
#: monolingual text, so it carries little weight; boilerplate and length discriminate most.
_HEURISTICS = {
    "length": (length_score, 0.25),
    "alpha_ratio": (alpha_ratio_score, 0.20),
    "boilerplate": (boilerplate_score, 0.30),
    "script_purity": (script_purity_score, 0.10),
    "repetition": (repetition_score, 0.15),
}


# --------------------------------------------------------------------------- #
# Provenance signal — reuse the existing trust layer.
# --------------------------------------------------------------------------- #


def _provenance_signal(doc: dict, *, trust, k: int) -> tuple[float | None, str, list[str]]:
    """Return ``(prov_score, decision, reasons)``.

    ``prov_score`` is ``None`` when the document declares no sources (caller treats this as
    "good but untrusted" via the no-provenance cap). Otherwise it blends the poison-gate's
    pooled, trust-weighted confidence with the OKF authorConfidence prior.
    """
    prov = doc.get("provenance") or {}
    sources = _doc.to_sources(doc)
    tier = prov.get("authorConfidence")
    tier_prior = AUTHOR_CONFIDENCE_PRIOR.get(tier, _DEFAULT_PRIOR)

    if not sources:
        reasons = ["no provenance sources (capped below top tier, fail-closed)"]
        if tier:
            reasons.append(f"authorConfidence tier {tier!r} -> prior {tier_prior:.2f}")
            return tier_prior, "no_sources", reasons
        return None, "no_sources", reasons

    item = {"claimId": doc.get("url"), "value": doc.get("content"), "sources": sources}
    verdict = assess_item(item, trust=trust, k=k)
    pooled = float(verdict["pooledConfidence"])
    # Blend pooled corroboration with the tier prior (equal weight).
    prov_score = round(0.5 * pooled + 0.5 * tier_prior, 6)
    reasons = [
        f"poison-gate: {verdict['decision']} "
        f"({verdict['independentCorroborations']} indep group(s), pooled {pooled:.3f})",
        f"authorConfidence tier {tier!r} -> prior {tier_prior:.2f}",
    ]
    return prov_score, verdict["decision"], reasons


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

#: Weight split between content heuristics and provenance when provenance exists.
_HEURISTIC_WEIGHT = 0.5
_PROVENANCE_WEIGHT = 0.5
#: A document with no provenance cannot exceed this score (good != trusted).
_NO_PROVENANCE_CAP = 0.7
#: Keep/drop threshold for the ``keep`` flag.
DEFAULT_KEEP_THRESHOLD = 0.5


def score_document(
    doc: dict,
    *,
    trust=None,
    k: int = 2,
    keep_threshold: float = DEFAULT_KEEP_THRESHOLD,
) -> dict:
    """Score one document. Returns a ``quality`` block ``{score, keep, signals, reasons}``.

    ``trust`` is a ``SourceTrust`` or a ``{sourceId: trust}`` mapping (unknown sources get a
    conservative default). The returned block is also suitable to assign to ``doc['quality']``.
    """
    if trust is None:
        trust = SourceTrust(scores={})

    text = doc.get("content") or ""
    signals = {name: fn(text) for name, (fn, _w) in _HEURISTICS.items()}
    heuristic_score = round(
        sum(signals[name] * w for name, (_fn, w) in _HEURISTICS.items()), 6
    )

    prov_score, decision, prov_reasons = _provenance_signal(doc, trust=trust, k=k)
    reasons = list(prov_reasons)

    if prov_score is None:
        # No provenance at all: capped — good content, but unverifiable.
        score = round(min(heuristic_score, _NO_PROVENANCE_CAP), 6)
        reasons.append(f"heuristics-only score capped at {_NO_PROVENANCE_CAP}")
    else:
        score = round(_HEURISTIC_WEIGHT * heuristic_score + _PROVENANCE_WEIGHT * prov_score, 6)

    # Fail-closed penalty: a quarantined / single-source-poison verdict caps the score.
    penalized = False
    if decision == "quarantine":
        score = round(min(score, _NO_PROVENANCE_CAP), 6)
        penalized = True
        reasons.append("poison-gate quarantine -> score capped (fail-closed)")

    keep = score >= keep_threshold and not penalized
    signals["heuristic_mean"] = heuristic_score
    if prov_score is not None:
        signals["provenance"] = prov_score

    return {
        "score": score,
        "keep": bool(keep),
        "signals": signals,
        "reasons": reasons,
    }


def score_documents(docs, *, trust=None, k: int = 2, keep_threshold: float = DEFAULT_KEEP_THRESHOLD):
    """Score an iterable of documents in place, yielding each with ``doc['quality']`` set."""
    if trust is None:
        trust = SourceTrust(scores={})
    for doc in docs:
        doc["quality"] = score_document(doc, trust=trust, k=k, keep_threshold=keep_threshold)
        yield doc


__all__ = ["score_document", "score_documents", "DEFAULT_KEEP_THRESHOLD"]
