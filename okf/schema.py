"""The OKF page contract for Sophia: page types, provenance vocabulary, validation.

Mirrors data/schema.json's provenance ontology (authorConfidence enum,
doNotAttributeTo, doNotMergeWith) so the wiki and the structured corpus share one
vocabulary rather than diverging. A page is YAML frontmatter (the provenance
record) plus a Markdown body (the prose).
"""

from __future__ import annotations

# Page types in the Sophia OKF profile. The first five mirror data/schema.json
# recordType; the rest are wiki-native (prose disputes, hubs, the schema itself).
PAGE_TYPES = (
    "text",
    "concept",
    "event",
    "figure",
    "figure_source_seat",
    "school",
    "tradition",
    "domain",
    "dispute",
    "index",
    "schema",
    "memory",
)

DOMAINS = ("philosophy", "psychology", "history", "religion")

# authorConfidence enum — copied from data/schema.json so a drift is a test failure.
AUTHOR_CONFIDENCE = (
    "attributed",
    "compiled",
    "legendary",
    "none_extant",
    "disputed",
    "consensus",
    "anachronism_risk",
    "layered",  # multi-hand authorship/reception (e.g. scripture layers) — see religion records
)

# Ordered strength for min-over-chain confidence propagation (lower == weaker).
# anachronism_risk is treated as a hard low cap (a claim with anachronism risk can
# never be laundered into a confident downstream assertion).
CONFIDENCE_RANK = {
    "none_extant": 0,
    "anachronism_risk": 0,
    "legendary": 1,
    "disputed": 1,
    "compiled": 2,
    "layered": 2,
    "attributed": 3,
    "consensus": 4,
}

# Required frontmatter keys for any OKF page.
REQUIRED_KEYS = ("id", "pageType")

# Typed edge keys carried in frontmatter (in addition to inline [[wikilinks]]).
EDGE_KEYS = ("links", "contradicts", "supersedes", "supersededBy", "doNotMergeWith")

# Provenance keys that an attribution-bearing page should carry.
PROVENANCE_KEYS = ("attributedAuthor", "authorConfidence", "tradition", "sources")


def confidence_rank(value) -> int:
    """Map an authorConfidence string to an ordered rank (unknown -> weakest)."""
    if not value:
        return 0
    return CONFIDENCE_RANK.get(str(value), 0)


def validate_meta(meta: dict) -> "list[str]":
    """Return a list of schema errors for one page's frontmatter (empty == valid)."""
    errors: list[str] = []
    if not isinstance(meta, dict):
        return ["frontmatter is not a mapping"]

    for key in REQUIRED_KEYS:
        if not meta.get(key):
            errors.append(f"missing required key: {key}")

    page_type = meta.get("pageType")
    if page_type is not None and page_type not in PAGE_TYPES:
        errors.append(f"invalid pageType '{page_type}' (allowed: {', '.join(PAGE_TYPES)})")

    rid = meta.get("id")
    if rid is not None and not _is_slug(str(rid)):
        errors.append(f"id '{rid}' is not a valid slug ([a-z0-9_-])")

    domain = meta.get("domain")
    if domain is not None and domain not in DOMAINS:
        errors.append(f"invalid domain '{domain}'")

    confidence = meta.get("authorConfidence")
    if confidence is not None and confidence not in AUTHOR_CONFIDENCE:
        errors.append(f"invalid authorConfidence '{confidence}'")

    for list_key in EDGE_KEYS + ("doNotAttributeTo", "sources", "aliases"):
        value = meta.get(list_key)
        if value is not None and not isinstance(value, list):
            errors.append(f"{list_key} must be a list")

    # An attribution must declare its confidence — the core source-discipline rule.
    if meta.get("attributedAuthor") and not meta.get("authorConfidence"):
        errors.append("attributedAuthor present but authorConfidence missing")

    return errors


def _is_slug(value: str) -> bool:
    if not value:
        return False
    return all(ch.isalnum() or ch in "_-" for ch in value) and value[0].isalnum()


def as_list(value) -> "list":
    """Coerce a frontmatter value to a list (None -> [], scalar -> [scalar])."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]
