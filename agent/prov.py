# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""W3C PROV lineage export (Stage B).

Sophia already records provenance: every gated claim is a ``Claim`` with
``claim_id``, ``content``, ``sources``, ``parents``, ``blp_level``, ``created_at``
and an optional ``signature`` (see ``sophia_contract/models.py:build_claim``).
What was missing was a way to serialize that lineage to an **interoperable,
queryable standard** so a third party can audit it without learning Sophia's
internal shapes.

This module maps Sophia claims to the **W3C PROV** data model (PROV-DM):

  - **Entity**   — a claim (the thing produced) and each external source.
  - **Activity** — the verification/recording act that produced the claim.
  - **Agent**    — Sophia itself (the recorder) and, where known, source authors.

Relations emitted:
  - ``wasGeneratedBy``  : claim  -> recording activity
  - ``used``            : activity -> source entities
  - ``wasDerivedFrom``  : claim  -> parent claims  (and claim -> its sources)
  - ``wasAttributedTo`` : claim  -> Sophia agent
  - ``wasAssociatedWith``: activity -> Sophia agent

Two serializations are produced, both standard and dependency-free:
  - **PROV-JSON** (the W3C PROV-JSON representation), and
  - **PROV-N** (the human-readable notation).

No network, no third-party packages. Deterministic given the input claims.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

#: Default namespace prefix for Sophia-minted identifiers.
PROV_NS = "sophia"
PROV_NS_URI = "https://tomyimkc.github.io/sophia-agi/prov#"


def _qid(local: str) -> str:
    """Qualify a local name into the Sophia PROV namespace."""
    safe = "".join(c if (c.isalnum() or c in "_.-") else "_" for c in str(local))
    return f"{PROV_NS}:{safe}"


def _source_entity_id(source: str) -> str:
    return _qid(f"source.{source}")


def _source_name(source: "Any") -> str:
    """Sources may be plain strings or contract dicts ``{"id":..,"status":..}``.
    Normalise both to a stable name."""
    if isinstance(source, dict):
        return str(source.get("id") or source.get("name") or source.get("source") or source)
    return str(source)


def _activity_id(claim_id: str) -> str:
    return _qid(f"verify.{claim_id}")


def claims_to_prov(claims: "Iterable[dict]", *, recorder: str = "sophia-gateway") -> dict:
    """Map an iterable of Sophia claims to a PROV-JSON document (a plain dict).

    Unknown/missing fields degrade gracefully: a claim with no ``parents`` simply
    emits no ``wasDerivedFrom`` to a parent; a claim with no ``sources`` emits no
    ``used``. The recorder agent is always present.
    """
    entities: dict = {}
    activities: dict = {}
    agents: dict = {}
    used: dict = {}
    generated: dict = {}
    derived: dict = {}
    attributed: dict = {}
    associated: dict = {}

    recorder_id = _qid(f"agent.{recorder}")
    agents[recorder_id] = {
        "prov:type": "prov:SoftwareAgent",
        "sophia:role": "recorder",
        "sophia:name": recorder,
    }

    n = 0
    for claim in claims:
        cid = claim.get("claim_id")
        if not cid:
            continue  # fail-closed: never mint a lineage node for an unidentified claim
        n += 1
        ent_id = _qid(f"claim.{cid}")
        act_id = _activity_id(cid)

        entities[ent_id] = {
            "prov:type": "sophia:Claim",
            "sophia:blp_level": claim.get("blp_level"),
            "sophia:content_sha": _content_sha(claim.get("content", "")),
            "sophia:signature": claim.get("signature"),
        }
        activities[act_id] = {
            "prov:type": "sophia:Verification",
            "prov:startTime": claim.get("created_at"),
            "prov:endTime": claim.get("created_at"),
        }
        generated[f"_gen{n}"] = {"prov:entity": ent_id, "prov:activity": act_id}
        attributed[f"_att{n}"] = {"prov:entity": ent_id, "prov:agent": recorder_id}
        associated[f"_assoc{n}"] = {"prov:activity": act_id, "prov:agent": recorder_id}

        for j, src in enumerate(claim.get("sources", []) or []):
            name = _source_name(src)
            sid = _source_entity_id(name)
            entities.setdefault(sid, {"prov:type": "sophia:Source", "sophia:name": name})
            used[f"_use{n}_{j}"] = {"prov:activity": act_id, "prov:entity": sid}
            derived[f"_derS{n}_{j}"] = {"prov:generatedEntity": ent_id, "prov:usedEntity": sid}

        for k, pid in enumerate(claim.get("parents", []) or []):
            pent = _qid(f"claim.{pid}")
            entities.setdefault(pent, {"prov:type": "sophia:Claim"})
            derived[f"_derP{n}_{k}"] = {"prov:generatedEntity": ent_id, "prov:usedEntity": pent}

    doc: dict = {
        "prefix": {PROV_NS: PROV_NS_URI, "prov": "http://www.w3.org/ns/prov#"},
        "entity": entities,
        "activity": activities,
        "agent": agents,
    }
    if used:
        doc["used"] = used
    if generated:
        doc["wasGeneratedBy"] = generated
    if derived:
        doc["wasDerivedFrom"] = derived
    if attributed:
        doc["wasAttributedTo"] = attributed
    if associated:
        doc["wasAssociatedWith"] = associated
    return doc


def to_prov_json(claims: "Iterable[dict]", *, recorder: str = "sophia-gateway", indent: int = 2) -> str:
    """Serialize claims to a PROV-JSON string."""
    return json.dumps(claims_to_prov(claims, recorder=recorder), indent=indent, sort_keys=True)


def to_prov_n(claims: "Iterable[dict]", *, recorder: str = "sophia-gateway") -> str:
    """Serialize claims to PROV-N (the human-readable W3C notation)."""
    doc = claims_to_prov(claims, recorder=recorder)
    lines: list = ["document"]
    for pfx, uri in doc["prefix"].items():
        lines.append(f"  prefix {pfx} <{uri}>")
    for eid, attrs in doc["entity"].items():
        lines.append(f"  entity({eid}{_attrs(attrs)})")
    for aid, attrs in doc["activity"].items():
        lines.append(f"  activity({aid}{_attrs(attrs)})")
    for gid, attrs in doc["agent"].items():
        lines.append(f"  agent({gid}{_attrs(attrs)})")
    for rec in doc.get("wasGeneratedBy", {}).values():
        lines.append(f"  wasGeneratedBy({rec['prov:entity']}, {rec['prov:activity']}, -)")
    for rec in doc.get("used", {}).values():
        lines.append(f"  used({rec['prov:activity']}, {rec['prov:entity']}, -)")
    for rec in doc.get("wasDerivedFrom", {}).values():
        lines.append(f"  wasDerivedFrom({rec['prov:generatedEntity']}, {rec['prov:usedEntity']})")
    for rec in doc.get("wasAttributedTo", {}).values():
        lines.append(f"  wasAttributedTo({rec['prov:entity']}, {rec['prov:agent']})")
    for rec in doc.get("wasAssociatedWith", {}).values():
        lines.append(f"  wasAssociatedWith({rec['prov:activity']}, {rec['prov:agent']}, -)")
    lines.append("endDocument")
    return "\n".join(lines)


def _attrs(attrs: dict) -> str:
    pairs = [(k, v) for k, v in attrs.items() if v is not None]
    if not pairs:
        return ""
    body = ", ".join(f'{k}="{v}"' for k, v in pairs)
    return f", [{body}]"


def _content_sha(content: str) -> str:
    import hashlib
    return hashlib.sha256(str(content).encode("utf-8")).hexdigest()[:16]


def validate_prov(doc: dict) -> "list[str]":
    """Lightweight structural check: every relation references a declared node.
    Returns a list of error strings (empty == valid). Used by tests/CI to keep
    the export honest without a PROV library."""
    errors: list = []
    entities = set(doc.get("entity", {}))
    activities = set(doc.get("activity", {}))
    agents = set(doc.get("agent", {}))
    for rec in doc.get("wasGeneratedBy", {}).values():
        if rec["prov:entity"] not in entities:
            errors.append(f"wasGeneratedBy references unknown entity {rec['prov:entity']}")
        if rec["prov:activity"] not in activities:
            errors.append(f"wasGeneratedBy references unknown activity {rec['prov:activity']}")
    for rec in doc.get("used", {}).values():
        if rec["prov:activity"] not in activities:
            errors.append(f"used references unknown activity {rec['prov:activity']}")
        if rec["prov:entity"] not in entities:
            errors.append(f"used references unknown entity {rec['prov:entity']}")
    for rec in doc.get("wasDerivedFrom", {}).values():
        for key in ("prov:generatedEntity", "prov:usedEntity"):
            if rec[key] not in entities:
                errors.append(f"wasDerivedFrom references unknown entity {rec[key]}")
    for rec in doc.get("wasAttributedTo", {}).values():
        if rec["prov:entity"] not in entities:
            errors.append(f"wasAttributedTo references unknown entity {rec['prov:entity']}")
        if rec["prov:agent"] not in agents:
            errors.append(f"wasAttributedTo references unknown agent {rec['prov:agent']}")
    return errors


__all__ = [
    "PROV_NS",
    "PROV_NS_URI",
    "claims_to_prov",
    "to_prov_json",
    "to_prov_n",
    "validate_prov",
]
