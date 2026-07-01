# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Streaming decontamination + temporal-validity gate for live-data ingestion.

Three fail-closed predicates a live web fact must pass before it may be written to
the belief store or a training pack. They make "freshness vs contamination-free"
an explicit, auditable admission decision rather than a batch scrub:

  * CONTENT decontamination — exact/normalized overlap plus word-k-shingle Jaccard
    near-duplicate scan against the committed eval surfaces (same contract as
    ``tools/assert_decontam.py`` and ``provenance_bench.dataset_guard``). A freshly
    scraped item that *paraphrases* a held-out eval prompt is rejected before it can
    inflate a later measurement; n-gram alone is bypassable, so the shingle layer
    backstops exact match.
  * TEMPORAL decontamination — reject any item whose source publish timestamp
    postdates the frozen evaluation cutoff. Date-filtered web retrieval still leaks
    post-cutoff content, so this turns the cutoff into a first-class admission rule.
  * VALID-TIME — a fact is an interval ``(validFrom, validUntil)``, not a scalar; a
    claim is only servable when the query's as-of date falls inside its interval.

Pure stdlib, deterministic, offline. Fail-closed by construction: an unparseable
date or an ambiguous check returns ``ok=False`` (do not ingest) rather than
guessing. Nothing here changes weights or writes canonical records.
"""
from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from provenance_bench.dataset_guard import ROOT, eval_prompt_set, normalize

__all__ = [
    "parse_date",
    "shingles",
    "jaccard",
    "content_decontam",
    "temporal_decontam",
    "valid_time",
    "eval_surface",
]


def parse_date(value: Any) -> date | None:
    """Parse an ISO-8601 date/datetime string to a ``date``; ``None`` if unparseable.

    Fail-closed helper: callers treat ``None`` as "cannot verify" and refuse.
    """
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    # Accept a trailing Z (UTC) and space-separated datetimes.
    text = text.replace("Z", "+00:00").replace(" ", "T", 1) if "T" not in text and " " in text else text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None


@lru_cache(maxsize=8192)
def _shingle_cache(text: str, k: int) -> frozenset[str]:
    """Cached word k-shingle set. Memoized so a fixed eval surface is shingled once
    across many claims, not once per claim (the hot path in content_decontam)."""
    toks = normalize(text).split()
    if len(toks) < k:
        return frozenset({" ".join(toks)}) if toks else frozenset()
    return frozenset(" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1))


def shingles(text: str, k: int) -> set[str]:
    """Word k-shingle set over the normalized text (mirrors tools/assert_decontam)."""
    return set(_shingle_cache(text, k))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def content_decontam(
    text: str,
    eval_prompts: set[str] | None,
    *,
    k: int = 5,
    jaccard_threshold: float = 0.9,
) -> dict[str, Any]:
    """Reject a candidate whose content is an exact or near-duplicate of any eval prompt.

    Returns ``{ok, reason, exact, maxJaccard, nearMatch}``. ``ok`` is True only when
    there is no exact/normalized overlap AND the max shingle-Jaccard against the eval
    surface is below ``jaccard_threshold``. An empty eval surface is ``ok`` (nothing to
    leak) but flagged in ``reason`` so a caller can require a non-empty surface.
    """
    if not eval_prompts:
        return {"ok": True, "reason": "no eval surface supplied; nothing to contaminate", "exact": False, "maxJaccard": 0.0, "nearMatch": ""}
    npr = normalize(text)
    if npr in eval_prompts:
        return {"ok": False, "reason": "exact/normalized overlap with an eval prompt", "exact": True, "maxJaccard": 1.0, "nearMatch": npr[:120]}
    tsh = _shingle_cache(text, k)
    if not tsh:
        return {"ok": True, "reason": "too short to shingle; exact check passed", "exact": False, "maxJaccard": 0.0, "nearMatch": ""}
    best = 0.0
    best_match = ""
    for e in eval_prompts:
        j = jaccard(tsh, _shingle_cache(e, k))
        if j > best:
            best, best_match = j, e
            if best >= 1.0:
                break
    ok = best < jaccard_threshold
    return {
        "ok": ok,
        "reason": ("clean vs eval surface" if ok else f"near-duplicate of an eval prompt (J={round(best, 3)} >= {jaccard_threshold})"),
        "exact": False,
        "maxJaccard": round(best, 3),
        "nearMatch": "" if ok else best_match[:120],
    }


def temporal_decontam(source_timestamp: Any, eval_cutoff: Any) -> dict[str, Any]:
    """Reject any item whose source publish date postdates the frozen eval cutoff.

    Fail-closed: an unparseable/missing source timestamp is ``ok=False`` (provenance
    cannot be verified, so it must not enter a surface that will later be measured).
    A missing cutoff is ``ok=True`` but flagged — the CLI requires a cutoff for a
    strict run.
    """
    cutoff = parse_date(eval_cutoff)
    if cutoff is None:
        return {"ok": True, "reason": "no eval cutoff configured (unstrict)", "sourceDate": None, "cutoff": None}
    src = parse_date(source_timestamp)
    if src is None:
        return {"ok": False, "reason": "missing/unparseable source timestamp; cannot verify freshness", "sourceDate": None, "cutoff": cutoff.isoformat()}
    if src > cutoff:
        return {"ok": False, "reason": f"source date {src.isoformat()} postdates eval cutoff {cutoff.isoformat()} (temporal leakage)", "sourceDate": src.isoformat(), "cutoff": cutoff.isoformat()}
    return {"ok": True, "reason": f"source date {src.isoformat()} <= cutoff {cutoff.isoformat()}", "sourceDate": src.isoformat(), "cutoff": cutoff.isoformat()}


def valid_time(valid_from: Any, valid_until: Any, as_of: Any) -> dict[str, Any]:
    """Check the query's as-of date falls inside the fact's validity interval.

    Open bounds are allowed (missing ``validFrom`` = valid since forever; missing
    ``validUntil`` = still valid). Fail-closed: an unparseable as-of date is
    ``ok=False``.
    """
    asof = parse_date(as_of)
    if asof is None:
        return {"ok": False, "reason": "missing/unparseable as-of date", "asOf": None}
    vf = parse_date(valid_from)
    vu = parse_date(valid_until)
    if vf is not None and asof < vf:
        return {"ok": False, "reason": f"as-of {asof.isoformat()} precedes validFrom {vf.isoformat()}", "asOf": asof.isoformat(), "validFrom": vf.isoformat(), "validUntil": vu.isoformat() if vu else None}
    if vu is not None and asof > vu:
        return {"ok": False, "reason": f"as-of {asof.isoformat()} is after validUntil {vu.isoformat()} (stale)", "asOf": asof.isoformat(), "validFrom": vf.isoformat() if vf else None, "validUntil": vu.isoformat()}
    return {"ok": True, "reason": "as-of within validity interval", "asOf": asof.isoformat(), "validFrom": vf.isoformat() if vf else None, "validUntil": vu.isoformat() if vu else None}


def eval_surface(root: Path | None = None) -> set[str]:
    """The committed eval-prompt surface to decontaminate against (dataset_guard)."""
    return eval_prompt_set(root=root or ROOT)
