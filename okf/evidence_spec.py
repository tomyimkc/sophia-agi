# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Evidence-Spec contract: derive confidence from typed evidence, detect illusory corroboration.

This module turns ``authorConfidence`` from a HAND-SET label into a value DERIVED from the
mix of typed evidence backing a record, capped by the per-type ceilings declared in
``evidence_spec.json``. Three ideas, all fail-closed:

1. **min-over-chain** — a chain is only as strong as its weakest link. A single legendary /
   disputed / anachronism-risk link caps the whole derived confidence, so a strong citation
   cannot launder a weak provenance chain (reuses ``okf.schema.confidence_rank``).

2. **per-type ceilings + recency** — each evidence TYPE (citation, formal-proof, experimental,
   verifier-pass, witness-testimony, consensus) declares the strongest confidence it can
   license, a minimum independent-source count, a required-corroboration floor, and a max age.
   A staling source with no / too-old ``observedDate`` does NOT count toward its type's
   recency-bounded corroboration.

3. **source-independence graph** — N nominally distinct sources that share one ``origin``
   collapse to a single effective source (illusory corroboration). Corroboration floors are
   checked against the EFFECTIVE independent count, so ten retellings of one rumour cannot
   clear a floor of three.

The contract bounds STRUCTURE, not truth: clearing it is necessary, not sufficient.

Public API:
    load_spec(path=None) -> dict
    independence_report(sources) -> {effectiveIndependentCount, collapsed, groups}
    derive_confidence(evidence_objs, spec, *, claimed=None, as_of=None) -> DerivedConfidence
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

from okf import schema

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC_PATH = ROOT / "evidence_spec.json"

# The rank inverse of okf.schema.CONFIDENCE_RANK: pick the STRONGEST label at or below a rank
# so a numeric ceiling maps back to an authorConfidence word deterministically.
_RANK_TO_LABEL = {
    0: "none_extant",
    1: "legendary",
    2: "compiled",
    3: "attributed",
    4: "consensus",
}


def load_spec(path: Optional[Path | str] = None) -> dict:
    """Load and lightly validate ``evidence_spec.json``.

    Raises ``FileNotFoundError`` if missing, ``ValueError`` if malformed or if a declared
    ``confidenceCeiling`` is not a known authorConfidence value.
    """
    p = Path(path) if path is not None else DEFAULT_SPEC_PATH
    if not p.exists():
        raise FileNotFoundError(f"evidence spec not found: {p}")
    spec = json.loads(p.read_text(encoding="utf-8"))
    types = spec.get("evidenceTypes")
    if not isinstance(types, dict) or not types:
        raise ValueError("evidence_spec.json: missing or empty 'evidenceTypes'")
    for name, t in types.items():
        ceil = t.get("confidenceCeiling")
        if ceil not in schema.AUTHOR_CONFIDENCE:
            raise ValueError(f"evidence type '{name}': invalid confidenceCeiling '{ceil}'")
        for k in ("minIndependentSources", "requiredCorroboration"):
            if not isinstance(t.get(k), int) or t.get(k) < 0:
                raise ValueError(f"evidence type '{name}': '{k}' must be a non-negative int")
    return spec


# --------------------------------------------------------------------------------------------
# Source-independence graph (illusory-corroboration detection)
# --------------------------------------------------------------------------------------------

def _source_origin(src) -> str:
    """The independence key for a source.

    A source is a dict; its ``origin`` is the true provenance root. Sources sharing an origin
    are NOT independent. When ``origin`` is absent we fall back to the source's own id/url so a
    source with no declared shared root counts as its own origin (does not silently collapse).
    """
    if isinstance(src, dict):
        for key in ("origin", "provenanceRoot", "sourceOrigin"):
            v = src.get(key)
            if v:
                return str(v).strip().lower()
        for key in ("id", "url", "ref", "sourceId"):
            v = src.get(key)
            if v:
                return f"__self__:{str(v).strip().lower()}"
        return f"__anon__:{id(src)}"
    return f"__self__:{str(src).strip().lower()}"


def independence_report(sources: Sequence) -> dict:
    """Collapse nominally-distinct sources by shared ``origin``.

    Returns::

        {
          "rawSourceCount": int,
          "effectiveIndependentCount": int,   # number of DISTINCT origins
          "groups": {origin: [source-labels...]},
          "collapsed": [ {origin, count, members} ]  # origins backing >1 nominal source
        }

    ``effectiveIndependentCount`` is the number the corroboration floors are checked against.
    """
    groups: dict[str, list] = {}
    order: list[str] = []
    for src in sources or []:
        origin = _source_origin(src)
        label = src.get("id") or src.get("url") or src.get("ref") if isinstance(src, dict) else str(src)
        label = label if label else origin
        if origin not in groups:
            groups[origin] = []
            order.append(origin)
        groups[origin].append(label)
    collapsed = [
        {"origin": o, "count": len(groups[o]), "members": groups[o]}
        for o in order
        if len(groups[o]) > 1
    ]
    return {
        "rawSourceCount": len(list(sources or [])),
        "effectiveIndependentCount": len(groups),
        "groups": {o: groups[o] for o in order},
        "collapsed": collapsed,
    }


# --------------------------------------------------------------------------------------------
# Recency
# --------------------------------------------------------------------------------------------

def _parse_date(value) -> Optional[date]:
    if not value:
        return None
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except Exception:
        try:
            return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
        except Exception:
            return None


def _within_recency(observed, as_of: date, max_age_days: Optional[int]) -> Optional[bool]:
    """None -> type is timeless (recency N/A). True/False -> fresh / stale.

    A source with no parseable ``observedDate`` on a STALING type returns False (fail-closed:
    unknown age does not count as fresh)."""
    if max_age_days is None:
        return None
    d = _parse_date(observed)
    if d is None:
        return False
    return (as_of - d).days <= int(max_age_days)


# --------------------------------------------------------------------------------------------
# Typed evidence model + derivation
# --------------------------------------------------------------------------------------------

@dataclass
class Evidence:
    """One typed piece of evidence backing a record.

    ``etype``    one of evidence_spec evidenceTypes (citation, formal-proof, ...).
    ``confidence`` the intrinsic authorConfidence this evidence itself carries (a chain link).
    ``sources``  list of source dicts, each ideally carrying an ``origin`` and ``observedDate``.
    """
    etype: str
    confidence: Optional[str] = None
    sources: list = field(default_factory=list)

    @staticmethod
    def from_obj(obj) -> "Evidence":
        if isinstance(obj, Evidence):
            return obj
        if not isinstance(obj, dict):
            raise TypeError("evidence must be a dict or Evidence")
        return Evidence(
            etype=str(obj.get("type") or obj.get("etype") or ""),
            confidence=obj.get("confidence") or obj.get("authorConfidence"),
            sources=list(obj.get("sources") or []),
        )


@dataclass
class DerivedConfidence:
    """Result of deriving an admissible confidence from a typed evidence chain."""
    derivedRank: int
    derivedLabel: str
    reasons: list = field(default_factory=list)
    perEvidence: list = field(default_factory=list)
    effectiveIndependentCount: int = 0
    collapsed: list = field(default_factory=list)
    claimedLabel: Optional[str] = None
    claimedRank: Optional[int] = None
    inflated: Optional[bool] = None

    def to_dict(self) -> dict:
        return {
            "derivedRank": self.derivedRank,
            "derivedLabel": self.derivedLabel,
            "reasons": self.reasons,
            "perEvidence": self.perEvidence,
            "effectiveIndependentCount": self.effectiveIndependentCount,
            "collapsed": self.collapsed,
            "claimedLabel": self.claimedLabel,
            "claimedRank": self.claimedRank,
            "inflated": self.inflated,
        }


def _rank_to_label(rank: int) -> str:
    return _RANK_TO_LABEL.get(max(0, min(4, int(rank))), "none_extant")


def derive_confidence(
    evidence_objs: Iterable,
    spec: dict,
    *,
    claimed: Optional[str] = None,
    as_of: Optional[date | str] = None,
) -> DerivedConfidence:
    """Derive the MAX admissible authorConfidence from a typed evidence chain (min-over-chain).

    Algorithm (fail-closed at every step):
      1. Parse each evidence into (type, intrinsic confidence, sources).
      2. Per evidence: cap its intrinsic confidence at the TYPE ceiling; require the type's
         minIndependentSources (by EFFECTIVE independent origin count); require the type's
         requiredCorroboration among RECENCY-FRESH sources on a staling type. Failing any of
         these degrades that evidence's licensed rank (toward 0).
      3. The chain's licensed rank is the MIN over evidence (weakest link) — because a claim
         resting on multiple pieces is only as sound as its worst-supported piece. (Aggregating
         evidence of DIFFERENT types is corroboration ACROSS the chain, captured separately by
         the global corroboration floor below, not by taking a max — a max would let one strong
         link launder a weak one, exactly what min-over-chain forbids.)
      4. Merge sources across ALL evidence into ONE independence graph; the global
         corroboration floor (corroborationFloorByRank) caps the derived rank at what the
         EFFECTIVE independent count supports (e.g. 'consensus' needs >=3 distinct origins).
      5. If ``claimed`` is given, flag ``inflated`` when claimedRank > derivedRank.

    Returns a ``DerivedConfidence``. An empty evidence chain derives rank 0 (none_extant).
    """
    as_of_d = _parse_date(as_of) or date.today()
    types = spec.get("evidenceTypes", {})
    floors = spec.get("corroborationFloorByRank", {})

    evs = [Evidence.from_obj(o) for o in (evidence_objs or [])]
    per_ev: list = []
    reasons: list = []

    all_sources: list = []
    chain_ranks: list[int] = []

    for ev in evs:
        tdef = types.get(ev.etype)
        rep = independence_report(ev.sources)
        eff = rep["effectiveIndependentCount"]
        all_sources.extend(ev.sources)

        if tdef is None:
            per_ev.append({"type": ev.etype, "licensedRank": 0,
                           "reason": f"unknown evidence type '{ev.etype}' -> licenses nothing"})
            chain_ranks.append(0)
            reasons.append(f"unknown evidence type '{ev.etype}' (fail-closed to rank 0)")
            continue

        ceil_rank = schema.confidence_rank(tdef.get("confidenceCeiling"))
        intrinsic_rank = schema.confidence_rank(ev.confidence) if ev.confidence else ceil_rank
        licensed = min(ceil_rank, intrinsic_rank)
        ev_reasons: list[str] = []

        # (2a) minimum INDEPENDENT sources for this type
        min_ind = int(tdef.get("minIndependentSources", 0))
        if eff < min_ind:
            ev_reasons.append(
                f"only {eff} independent source(s) < minIndependentSources={min_ind}")
            licensed = 0

        # (2b) recency-bounded corroboration for staling types
        max_age = tdef.get("maxAgeDays")
        req_corr = int(tdef.get("requiredCorroboration", 0))
        if max_age is not None:
            fresh_origins = set()
            for s in ev.sources:
                observed = s.get("observedDate") if isinstance(s, dict) else None
                if _within_recency(observed, as_of_d, max_age):
                    fresh_origins.add(_source_origin(s))
            if len(fresh_origins) < req_corr:
                ev_reasons.append(
                    f"only {len(fresh_origins)} recency-fresh independent source(s) "
                    f"(<= {max_age}d) < requiredCorroboration={req_corr}")
                licensed = min(licensed, schema.confidence_rank("legendary"))
        else:
            # timeless type: corroboration floor still uses effective independent count
            if eff < req_corr:
                ev_reasons.append(
                    f"only {eff} independent source(s) < requiredCorroboration={req_corr}")
                licensed = min(licensed, schema.confidence_rank("legendary"))

        if rep["collapsed"]:
            ev_reasons.append(
                f"illusory corroboration: {rep['rawSourceCount']} sources collapse to "
                f"{eff} origin(s)")

        chain_ranks.append(licensed)
        per_ev.append({
            "type": ev.etype,
            "ceiling": tdef.get("confidenceCeiling"),
            "intrinsic": ev.confidence,
            "licensedRank": licensed,
            "licensedLabel": _rank_to_label(licensed),
            "effectiveIndependentCount": eff,
            "collapsed": rep["collapsed"],
            "reason": "; ".join(ev_reasons) if ev_reasons else "ok",
        })

    # (3) weakest link
    derived_rank = min(chain_ranks) if chain_ranks else 0

    # (4) global corroboration floor across the whole chain
    global_rep = independence_report(all_sources)
    eff_global = global_rep["effectiveIndependentCount"]
    # Find the strongest rank whose floor the effective independent count can support.
    for r in range(derived_rank, -1, -1):
        need = int(floors.get(_rank_to_label(r), 0))
        if eff_global >= need:
            capped = r
            break
    else:
        capped = 0
    if capped < derived_rank:
        reasons.append(
            f"global corroboration floor: {eff_global} independent origin(s) support at most "
            f"rank {capped} ({_rank_to_label(capped)}), not {derived_rank}")
    derived_rank = capped

    if global_rep["collapsed"]:
        reasons.append(
            f"chain-wide illusory corroboration: {global_rep['rawSourceCount']} sources "
            f"collapse to {eff_global} origin(s): "
            + ", ".join(c["origin"] for c in global_rep["collapsed"]))

    derived_label = _rank_to_label(derived_rank)
    claimed_rank = schema.confidence_rank(claimed) if claimed else None
    inflated = None
    if claimed is not None:
        inflated = claimed_rank > derived_rank
        if inflated:
            reasons.append(
                f"CLAIMED '{claimed}' (rank {claimed_rank}) EXCEEDS derived "
                f"'{derived_label}' (rank {derived_rank}) — confidence inflation")

    if not evs:
        reasons.append("no typed evidence supplied -> derives none_extant (rank 0)")

    return DerivedConfidence(
        derivedRank=derived_rank,
        derivedLabel=derived_label,
        reasons=reasons,
        perEvidence=per_ev,
        effectiveIndependentCount=eff_global,
        collapsed=global_rep["collapsed"],
        claimedLabel=claimed,
        claimedRank=claimed_rank,
        inflated=inflated,
    )
