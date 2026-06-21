"""Contamination control — catch NEAR-EXACT train↔eval overlap beyond exact-ID holdout.

`tools/prepare_lora_dataset.py` holds out eval items by ID/trap; that misses an eval
item being copied near-verbatim into training under a different file. This measures
that with word-n-gram **shingle containment**: for each eval item, the maximum
fraction of its n-word shingles found in any single train item.

Honest scope: this detects **near-exact / verbatim-span** duplication (≈``n``
consecutive identical words), NOT semantic paraphrase — a fully reworded item with
no shared n-gram run scores ~0. It is a precise near-dup detector, not a
semantic-overlap model. For short eval items the shingle size adapts down so a short
verbatim subset is still detected. Deterministic, no model.
"""

from __future__ import annotations

import re


def _words(text: str) -> list:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _shingles(words: list, k: int) -> set:
    if not words:
        return set()
    if len(words) <= k:
        return {tuple(words)}
    return {tuple(words[i:i + k]) for i in range(len(words) - k + 1)}


def _containment(a: set, b: set) -> float:
    """Fraction of a's shingles also in b (asymmetric — how much of eval is in train)."""
    return (len(a & b) / len(a)) if a else 0.0


def overlap_report(train_texts: list, eval_texts: list, *, n: int = 8, threshold: float = 0.6) -> dict:
    """Max near-duplicate containment of each eval item against the train set. The
    shingle size adapts to ``min(n, eval-length)`` per item so a short verbatim
    subset is detected (both sides are shingled at the same size for comparison)."""
    train_words = [_words(t) for t in train_texts]
    rows = []
    for et in eval_texts:
        ew = _words(et)
        k = min(n, len(ew)) or 1
        es = _shingles(ew, k)
        mc = max((_containment(es, _shingles(tw, k)) for tw in train_words), default=0.0)
        rows.append({"containment": round(mc, 4), "contaminated": mc >= threshold})
    n_eval = len(eval_texts) or 1
    return {
        "n": len(eval_texts),
        "threshold": threshold,
        "shingleSize": n,
        "contaminationRate": round(sum(r["contaminated"] for r in rows) / n_eval, 4),
        "maxContainment": round(max((r["containment"] for r in rows), default=0.0), 4),
        "rows": rows,
    }


def assert_clean(train_texts: list, eval_texts: list, *, n: int = 8,
                 threshold: float = 0.6, max_rate: float = 0.0) -> dict:
    """Report + an ``ok`` flag: contamination rate must be ≤ ``max_rate``."""
    rep = overlap_report(train_texts, eval_texts, n=n, threshold=threshold)
    rep["ok"] = rep["contaminationRate"] <= max_rate
    return rep
