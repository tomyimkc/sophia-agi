"""The OKF wiki librarian: ingest a raw source into a gated wiki page proposal.

The LLM-Wiki loop, with Sophia's source discipline wired in: read a raw source
(fenced as untrusted), extract one structured page, and write it ONLY if it passes
the provenance gate (schema-valid + no forbidden attribution + no lineage merge).
The model call and the gating/writing are separated so the gate is unit-testable
offline without depending on model output.
"""

from __future__ import annotations

import json
import re

from agent import untrusted, wiki_store
from agent.model import ModelClient, default_client

LIBRARIAN_SYSTEM = (
    "You are Sophia's OKF wiki librarian — wisdom before intelligence, strict source "
    "discipline. Read the UNTRUSTED source and output ONLY one JSON object for a single "
    "wiki page. Never invent provenance: only set attributedAuthor/authorConfidence if "
    "the source supports it. Keys: id (snake_case), pageType (text|concept|event|figure), "
    "domain, title, titleZh, tradition, attributedAuthor, authorConfidence "
    "(attributed|compiled|legendary|disputed|consensus|none_extant|anachronism_risk|layered), "
    "doNotAttributeTo (list), links (list of page ids), summary (2-4 sentence prose)."
)


def _extract_json(text: str) -> "dict | None":
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        candidate = text[start : end + 1] if start >= 0 and end > start else None
    if not candidate:
        return None
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def build_page(proposal: dict, source_id: str) -> "tuple":
    """Pure transform: a proposal dict -> (meta, body). No model, no I/O."""
    pid = str(proposal.get("id") or re.sub(r"[^a-z0-9]+", "_", str(proposal.get("title", "")).lower())).strip("_")
    meta = {
        "id": pid,
        "pageType": proposal.get("pageType") or "concept",
        "domain": proposal.get("domain"),
        "tradition": proposal.get("tradition"),
        "attributedAuthor": proposal.get("attributedAuthor"),
        "authorConfidence": proposal.get("authorConfidence"),
        "canonicalTitleEn": proposal.get("title"),
        "canonicalTitleZh": proposal.get("titleZh"),
        "doNotAttributeTo": list(proposal.get("doNotAttributeTo") or []),
        "links": list(proposal.get("links") or []),
        "sources": [f"raw/{source_id}"],
    }
    meta = {k: v for k, v in meta.items() if v is not None or k in ("doNotAttributeTo", "links", "sources")}

    title = proposal.get("title") or pid
    summary = str(proposal.get("summary") or "").strip()
    lines = [f"# {title}", "", summary]
    dna = meta.get("doNotAttributeTo") or []
    if dna:
        lines += ["", f"> **Do not attribute to:** {', '.join(dna)}."]
    lines += ["", f"_Librarian draft ingested from `raw/{source_id}` — provenance unverified until reviewed._"]
    return meta, "\n".join(lines) + "\n"


def ingest_proposal(proposal: dict, source_id: str, *, tier: str = "draft") -> dict:
    """Gate + write a proposal. Returns the wiki_store.upsert result (never raises)."""
    if not proposal or not (proposal.get("id") or proposal.get("title")):
        return {"ok": False, "rejected": True, "reasons": ["proposal missing id/title"]}
    meta, body = build_page(proposal, source_id)
    result = wiki_store.upsert(meta["id"], meta=meta, body=body, tier=tier)
    result["sourceId"] = source_id
    return result


def ingest_text(source_text: str, source_id: str, *, client: "ModelClient | None" = None, tier: str = "draft") -> dict:
    """Full loop: model extracts a page from the (untrusted) source, then gate+write."""
    client = client or default_client()
    fenced = untrusted.wrap_untrusted(source_text, f"raw:{source_id}")
    user = (
        f"Source id: {source_id}\n\n{fenced}\n\n"
        "Output ONLY the JSON page object. Apply source discipline: do not assert any "
        "attribution the source does not support; populate doNotAttributeTo for known traps."
    )
    result = client.generate(LIBRARIAN_SYSTEM, user)
    if not result.ok:
        return {"ok": False, "rejected": True, "reasons": [result.error or "model error"], "sourceId": source_id}
    proposal = _extract_json(result.text)
    if proposal is None:
        return {"ok": False, "rejected": True, "reasons": ["no JSON page object in model output"], "sourceId": source_id}
    return ingest_proposal(proposal, source_id, tier=tier)
