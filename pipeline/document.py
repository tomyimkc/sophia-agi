# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Document contract for the pipeline (Phase 0).

A lightweight, dependency-free validator for the canonical document record defined in
``pipeline/schemas/document.schema.json``. We deliberately do NOT import ``jsonschema``
here so the contract can be checked under ``SOPHIA_PROFILE=airgap`` with only the stdlib;
the JSON Schema file remains the authoritative spec for external consumers.

A *valid* document needs only ``url`` and ``content``; the ``provenance``, ``quality``, and
``dedup`` blocks are progressively filled by later stages. ``validate`` is fail-closed: it
returns a list of human-readable problems (empty == valid) rather than raising, so a batch
job can quarantine bad rows and keep going.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

#: authorConfidence tiers accepted in provenance (mirrors grounded_confidence priors).
_AUTHOR_CONFIDENCE_TIERS = frozenset(
    {
        "consensus",
        "attributed",
        "compiled",
        "layered",
        "disputed",
        "legendary",
        "anachronism_risk",
        "none_extant",
    }
)


def _is_prob(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and 0.0 <= float(x) <= 1.0


def validate(doc: dict) -> list[str]:
    """Return a list of problems with ``doc`` (empty list == valid).

    Checks required fields, types, and the value ranges that downstream scoring relies on.
    Unknown extra keys are allowed (the schema is ``additionalProperties: true``).
    """
    problems: list[str] = []
    if not isinstance(doc, dict):
        return [f"document must be an object, got {type(doc).__name__}"]

    url = doc.get("url")
    if not isinstance(url, str) or not url.strip():
        problems.append("missing or empty 'url'")
    content = doc.get("content")
    if not isinstance(content, str):
        problems.append("missing or non-string 'content'")

    prov = doc.get("provenance")
    if prov is not None:
        if not isinstance(prov, dict):
            problems.append("'provenance' must be an object")
        else:
            ac = prov.get("authorConfidence")
            if ac is not None and ac not in _AUTHOR_CONFIDENCE_TIERS:
                problems.append(f"unknown provenance.authorConfidence tier {ac!r}")
            sources = prov.get("sources", [])
            if sources is not None and not isinstance(sources, list):
                problems.append("'provenance.sources' must be an array")
            else:
                for i, src in enumerate(sources or []):
                    if not isinstance(src, dict):
                        problems.append(f"provenance.sources[{i}] must be an object")
                        continue
                    for field in ("trust", "confidence"):
                        if field in src and not _is_prob(src[field]):
                            problems.append(
                                f"provenance.sources[{i}].{field} must be a probability in [0,1]"
                            )

    quality = doc.get("quality")
    if quality is not None:
        if not isinstance(quality, dict):
            problems.append("'quality' must be an object")
        elif "score" in quality and not _is_prob(quality["score"]):
            problems.append("quality.score must be a probability in [0,1]")

    return problems


def is_valid(doc: dict) -> bool:
    """True iff ``doc`` has no validation problems."""
    return not validate(doc)


def to_sources(doc: dict, *, default_trust: float | None = None) -> list[dict]:
    """Extract the ``provenance.sources`` list in the shape ``assess_item`` expects.

    Returns an empty list when no provenance is present, so callers can fall back to a
    heuristic-only score. ``default_trust`` (if given) backfills sources missing a trust.
    """
    prov = doc.get("provenance") or {}
    sources = prov.get("sources") or []
    out: list[dict] = []
    for i, src in enumerate(sources):
        if not isinstance(src, dict):
            continue
        entry = dict(src)
        entry.setdefault("sourceId", f"_src{i}")
        if default_trust is not None and entry.get("trust") is None:
            entry["trust"] = default_trust
        out.append(entry)
    return out


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield document dicts from a JSONL file (skips blank lines)."""
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, docs: Iterable[dict]) -> int:
    """Write ``docs`` to a JSONL file; returns the row count."""
    n = 0
    with Path(path).open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


__all__ = ["validate", "is_valid", "to_sources", "read_jsonl", "write_jsonl"]
