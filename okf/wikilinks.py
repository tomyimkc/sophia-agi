"""Parse and normalize ``[[wikilinks]]`` in OKF page bodies.

Supports ``[[target]]``, ``[[target#anchor]]`` and ``[[target|alias]]``. Targets
are normalized to a slug form so a link written as ``[[Dao De Jing]]`` resolves to
the page id ``dao_de_jing``.
"""

from __future__ import annotations

import re

WIKILINK_RE = re.compile(r"\[\[\s*([^\]\|#]+?)\s*(?:#[^\]\|]+)?\s*(?:\|[^\]]+)?\]\]")


def normalize_target(target: str) -> str:
    """Slugify a link target so '[[Dao De Jing]]' and '[[dao-de-jing]]' both match."""
    slug = target.strip().lower()
    slug = re.sub(r"[\s\-]+", "_", slug)
    slug = re.sub(r"[^a-z0-9_]", "", slug)
    return slug.strip("_")


def extract_links(body: str) -> "list[str]":
    """Return normalized link targets in order of first appearance (deduped)."""
    seen: list[str] = []
    for match in WIKILINK_RE.finditer(body or ""):
        target = normalize_target(match.group(1))
        if target and target not in seen:
            seen.append(target)
    return seen


def raw_links(body: str) -> "list[str]":
    """Return the raw (un-normalized) link targets, for diagnostics."""
    return [m.group(1).strip() for m in WIKILINK_RE.finditer(body or "")]
