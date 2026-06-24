"""Skill: wiki_grounded_answer — answer only from the OKF wiki; abstain when out-of-wiki."""
from __future__ import annotations

from skills.core import sophia_skill
from skills.mcp_bridge import call


import re


def _page_id(row: dict) -> str | None:
    for k in ("id", "pageId", "page_id", "slug"):
        if row.get(k):
            return str(row[k])
    return None


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(t) > 2}


def _relevant(query: str, row: dict) -> bool:
    """wiki_search returns fuzzy matches with no score, so we apply our own honest
    relevance gate: the matched page slug/path must share a content token with the
    query. Zero overlap = genuinely out-of-wiki (abstain), not a weak match."""
    target = f"{_page_id(row) or ''} {row.get('path', '')}"
    return bool(_tokens(query) & _tokens(target))


@sophia_skill(
    "wiki_grounded_answer",
    summary="Answer strictly from the OKF provenance wiki; abstains (held) when the question is out-of-wiki instead of fabricating.",
    uses=("wiki_search", "wiki_read"),
)
def wiki_grounded_answer(*, query: str, top_k: int = 5) -> dict:
    hits = call("wiki_search", query=query, top_k=top_k)
    results = [r for r in (hits.get("results") or []) if isinstance(r, dict)]
    relevant = [r for r in results if _relevant(query, r)]
    if not relevant:
        return {
            "verdict": "held",
            "grounded": False,
            "reason": "out-of-wiki: no OKF page is relevant to the query; abstaining instead of fabricating",
            "detail": hits,
        }
    top_id = _page_id(relevant[0])
    page = call("wiki_read", page_id=top_id) if top_id else {}
    return {
        "verdict": "grounded",
        "grounded": True,
        "topPageId": top_id,
        "sources": [_page_id(r) for r in relevant][:top_k],
        "page": page,
        "detail": hits,
    }
