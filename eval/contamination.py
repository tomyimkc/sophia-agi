"""Contamination control — catch train↔eval overlap beyond exact-ID holdout.

`tools/prepare_lora_dataset.py` already holds out eval items by ID/trap, but that
misses **paraphrased / near-duplicate** leakage (the same content reworded slips
into training). This module measures near-duplicate overlap with word-n-gram
**shingles**: for each eval item, the maximum *containment* of its shingles within
any single train item. A high containment means the eval item is substantially
present in training — contamination — even if the surface text differs.

Falsifiable: a planted near-duplicate is flagged (positive control); a genuinely
held-out split is below threshold. Deterministic, no model.
"""

from __future__ import annotations

import re


def _shingles(text: str, n: int = 8) -> set:
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _containment(a: set, b: set) -> float:
    """Fraction of a's shingles also in b (asymmetric — how much of eval is in train)."""
    return (len(a & b) / len(a)) if a else 0.0


def overlap_report(train_texts: list, eval_texts: list, *, n: int = 8, threshold: float = 0.6) -> dict:
    """Max near-duplicate containment of each eval item against the train set."""
    train_sh = [_shingles(t, n) for t in train_texts]
    rows = []
    for et in eval_texts:
        es = _shingles(et, n)
        mc = max((_containment(es, ts) for ts in train_sh), default=0.0)
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
