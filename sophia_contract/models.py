"""Wire data models for the contract: Claim and Verdict builders + input
normalisation. Field names here ARE the contract — never rename in a MINOR.

Sources accept either a bare string (an id/url) or an object
``{id|url, status?, date?}`` where ``status`` ∈ {ok, stale, refuted, invalid}.
Normalising both forms keeps the wire ergonomic while verification stays
deterministic (staleness/refutation are explicit fields, not wall-clock guesses —
so golden vectors are reproducible).
"""

from __future__ import annotations

import hashlib
import json

from sophia_contract import blp
from sophia_contract.errors import ContractError

VERDICTS = ("accepted", "rejected", "superseded", "held")
HELD_REASONS = ("no_source", "stale_source", "needs_human", "over_budget", "blp_violation")
SOURCE_STATUSES = ("ok", "stale", "refuted", "invalid")


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def claim_id_for(idempotency_key: str) -> str:
    """Deterministic claim id from the idempotency key — the same key always yields
    the same id, even across restarts, with no shared state required."""
    digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
    return f"clm_{digest[:24]}"


def normalize_source(src) -> dict:
    """Coerce a source to ``{id, status, date?}``; raise BAD_REQUEST if malformed."""
    if isinstance(src, str):
        if not src.strip():
            raise ContractError("BAD_REQUEST", "source string must be non-empty")
        return {"id": src.strip(), "status": "ok"}
    if isinstance(src, dict):
        sid = str(src.get("id") or src.get("url") or "").strip()
        if not sid:
            raise ContractError("BAD_REQUEST", "source object needs an 'id' or 'url'")
        status = str(src.get("status", "ok"))
        if status not in SOURCE_STATUSES:
            raise ContractError("BAD_REQUEST", f"source.status must be one of {SOURCE_STATUSES}")
        out = {"id": sid, "status": status}
        if src.get("date"):
            out["date"] = str(src["date"])
        return out
    raise ContractError("BAD_REQUEST", "each source must be a string or object")


def validate_record_request(req: dict) -> dict:
    """Validate + normalise a record_claim request. Returns the cleaned fields."""
    if not isinstance(req, dict):
        raise ContractError("BAD_REQUEST", "request must be an object")
    key = req.get("idempotency_key")
    if not isinstance(key, str) or not key.strip():
        raise ContractError("BAD_REQUEST", "idempotency_key is required and must be a non-empty string")
    content = req.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ContractError("BAD_REQUEST", "content is required and must be a non-empty string")
    blp_level = req.get("blp_level", "UNCLASSIFIED")
    if not blp.is_level(blp_level):
        raise ContractError("BAD_REQUEST", f"blp_level must be one of {blp.BLP_LEVELS}")
    raw_sources = req.get("sources", []) or []
    raw_parents = req.get("parents", []) or []
    if not isinstance(raw_sources, list) or not isinstance(raw_parents, list):
        raise ContractError("BAD_REQUEST", "sources and parents must be arrays")
    sources = [normalize_source(s) for s in raw_sources]
    parents = [str(p) for p in raw_parents]
    return {
        "idempotency_key": key.strip(),
        "content": content,
        "sources": sources,
        "parents": parents,
        "blp_level": blp_level,
    }


def build_claim(fields: dict, *, created_at: str, signing_key: "str | None" = None) -> dict:
    """Assemble the Claim wire object (deterministic id + optional signature)."""
    claim = {
        "claim_id": claim_id_for(fields["idempotency_key"]),
        "content": fields["content"],
        "sources": fields["sources"],
        "parents": fields["parents"],
        "blp_level": fields["blp_level"],
        "created_at": created_at,
    }
    if signing_key:
        body = _canonical({k: claim[k] for k in ("claim_id", "content", "sources", "parents", "blp_level")})
        sig = hashlib.sha256((signing_key + body).encode("utf-8")).hexdigest()
        claim["signature"] = f"sig_{sig[:32]}"
    return claim


def content_fingerprint(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def build_verdict(
    verdict: str,
    *,
    confidence: float,
    reasons: "list[str]",
    cited_evidence: "list[dict]",
    suggested_fix: "str | None" = None,
    supersedes: "str | None" = None,
    held_reason: "str | None" = None,
) -> dict:
    """Assemble the Verdict wire object; optional fields appear only when set."""
    if verdict not in VERDICTS:
        raise ContractError("INTERNAL", f"bad verdict {verdict!r}")
    if held_reason is not None and held_reason not in HELD_REASONS:
        raise ContractError("INTERNAL", f"bad held_reason {held_reason!r}")
    out = {
        "verdict": verdict,
        "confidence": round(float(confidence), 4),
        "reasons": list(reasons),
        "cited_evidence": list(cited_evidence),
    }
    if suggested_fix is not None:
        out["suggested_fix"] = suggested_fix
    if supersedes is not None:
        out["supersedes"] = supersedes
    if held_reason is not None:
        out["held_reason"] = held_reason
    return out
