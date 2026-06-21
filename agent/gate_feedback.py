"""Active-learning feedback: turn live judge-vs-gate disagreements into
candidate ``doNotAttributeTo`` records.

The provenance gate (``agent.guarded.check_claim`` / ``verifiers.provenance_faithful``)
can only fire on rules it already knows. When the LLM judge catches a
hallucination on a case the gate let through ``clean`` (a *gate MISS*), that
disagreement is exactly the signal the continual-learning loop needs: the gate
is missing a rule it should have.

This module mines those misses into *candidate* records and appends them to a
pending JSONL queue. It deliberately does **not** mutate the live, frozen gate
records — a human (or a separate promotion step) adopts a candidate before it
becomes a treatment rule, which preserves the non-circularity guarantee.

Pure stdlib, deterministic, offline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from provenance_bench.dataset import _alt_titles, _author_marker


def _rid(work: str) -> str:
    """Same record-id derivation as ``dataset.build_gate_records``."""
    return re.sub(r"[^a-z0-9]+", "_", work.lower()).strip("_")


def detect_miss(case_result: dict) -> Optional[dict]:
    """Return a candidate record when the JUDGE caught a hallucination the GATE
    missed, else ``None``.

    A gate MISS is: the gate passed the answer unchanged (``gated_action ==
    'clean'``) yet the judge flagged the *gated* answer as hallucinated. That is
    a forbidden attribution slipping past the gate.

    ``case_result`` is a ``runner.run_case``-style dict with at least
    ``work``, ``gated_action`` and ``gated: {hallucinated: bool}``. The wrong
    author is taken from ``claimed_author`` when present, otherwise from
    ``gold_author`` is *not* used (gold is correct); if no claimed author is
    available the function returns ``None`` (nothing actionable to forbid).
    """
    if (case_result.get("gated_action") or "") != "clean":
        return None
    gated = case_result.get("gated") or {}
    if not gated.get("hallucinated"):
        return None

    work = (case_result.get("work") or "").strip()
    claimed = (case_result.get("claimed_author") or "").strip()
    if not work or not claimed:
        return None

    return candidate_record(work, claimed)


def candidate_record(work: str, claimed_author: str) -> dict:
    """Synthesize a ``{rid: {...}}`` candidate gate record forbidding the
    attribution of ``work`` to ``claimed_author``.

    Mirrors the shape produced by ``dataset.build_gate_records`` so a promotion
    step can merge it into the live records dict directly.
    """
    marker = _author_marker(claimed_author)
    rid = _rid(work)
    return {
        rid: {
            "canonicalTitleEn": work,
            "altTitlesEn": _alt_titles(work),
            "doNotAttributeTo": [marker],
        }
    }


def _record_key(candidate: dict) -> tuple:
    """Stable dedupe key for a single-record candidate: (rid, frozenset of
    forbidden markers). Title/alt-title differences for the same rid+markers are
    treated as the same pending item."""
    (rid, rec), = candidate.items()
    markers = tuple(sorted(rec.get("doNotAttributeTo", [])))
    return (rid, markers)


def append_pending(candidate: dict, path) -> int:
    """Append ``candidate`` to the pending JSONL queue at ``path`` (deduped),
    returning the total number of pending records afterwards.

    Does NOT touch the live frozen records — this is a staging queue only. A
    candidate already present (same rid + forbidden markers) is not re-appended.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    seen: set = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict) or not rec:
                continue
            existing.append(rec)
            seen.add(_record_key(rec))

    key = _record_key(candidate)
    if key not in seen:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(candidate, ensure_ascii=False) + "\n")
        existing.append(candidate)

    return len(existing)
