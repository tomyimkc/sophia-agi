# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Verifier-gated real-time grounding — the fast, reversible ingestion loop.

This is the "close the loop" layer for linking a text world model to live web data
and fact-checking it instantly. It reuses the repo's existing truth filter rather
than adding a new one: each live claim runs through the layered fact-check gate
(:mod:`agent.fact_check_gate` with injected retrieval/entailment backends from
:mod:`agent.live_sources`), the fact-check confidence becomes a conformal
nonconformity score (:mod:`agent.conformal_gate`), and admission is fail-closed
behind the streaming/temporal decontamination + valid-time gate
(:mod:`agent.streaming_decontam`).

The unified verifier does four jobs with one provenance trail: (a) the same gate
that scores RLVR rollouts here (b) admits or quarantines a live fact, while the
belief store carries the audit fields a later (c) output check and (d) background
re-verification need.

Adaptation depth here is the SAFE one: writes go to an external belief store
(retrieval/memory), never to weights. The slow, weight-changing loop lives in
:mod:`agent.realtime_consolidation` and is gated on these verifier-certified rows.

Honest scope: nothing here trains or claims capability. Every record carries
``candidateOnly=True`` / ``level3Evidence=False``; ingestion is admission control,
not knowledge creation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from agent import streaming_decontam as sd
from agent.conformal_gate import ConformalPolicy
from agent.fact_check_gate import GateDecision, decision_to_dict, fact_check_text

SCHEMA = "sophia.realtime_grounding.v1"
BELIEF_SCHEMA = "sophia.belief.v1"

# Default conformal threshold used only when no fitted policy is supplied. A claim
# whose nonconformity exceeds this abstains (is not ingested). Kept conservative.
DEFAULT_NONCONFORMITY_THRESHOLD = 0.3


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _today() -> str:
    return date.today().isoformat()


def claim_id(text: str) -> str:
    return "rt_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def nonconformity(decision: GateDecision) -> float:
    """Map a fact-check verdict to a conformal nonconformity score in [0, 1].

    Lower = more confidently supported. ``accepted`` uses the weakest atomic-claim
    confidence (the chain is only as strong as its weakest link); ``rejected`` is
    maximally nonconforming; ``held`` sits near the abstain band.
    """
    if decision.verdict == "accepted":
        confs = [c.confidence for c in decision.claims] or [0.0]
        return round(max(0.0, 1.0 - min(confs)), 4)
    if decision.verdict == "rejected":
        return 1.0
    return 0.6  # held / abstain band


@dataclass
class BeliefRow:
    """One admission decision for a live claim. schema ``sophia.belief.v1``."""

    claimId: str
    claim: str
    ingestState: str  # ingested | quarantined | rejected | stale
    verdict: str  # fact-check verdict: accepted | held | rejected
    confidence: float
    nonconformity: float
    risk: str
    conformal: dict[str, Any]
    contentDecontam: dict[str, Any]
    temporalDecontam: dict[str, Any]
    validTime: dict[str, Any]
    sources: list[dict[str, Any]]
    validFrom: str = ""
    validUntil: str = ""
    sourceTimestamp: str = ""
    reason: str = ""
    ingestedAt: str = ""
    needsReverify: bool = False
    factCheck: dict[str, Any] = field(default_factory=dict)
    candidateOnly: bool = True
    level3Evidence: bool = False
    schema: str = BELIEF_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _Backend:
    """Structural type for an injected source backend (duck-typed).

    Any object exposing ``retriever``/``entailment``/``doi_resolver``/``url_resolver``
    works — e.g. ``agent.live_sources.FixtureFactBackend`` (offline) or
    ``LiveFactBackend`` (keyless online).
    """

    retriever: Callable
    entailment: Callable
    doi_resolver: Callable
    url_resolver: Callable


def _backend_kwargs(backend: Any) -> dict[str, Any]:
    if backend is None:
        return {}
    return {
        "retriever": getattr(backend, "retriever", None),
        "entailment": getattr(backend, "entailment", None),
        "doi_resolver": getattr(backend, "doi_resolver", None),
        "url_resolver": getattr(backend, "url_resolver", None),
    }


def _conformal_decide(policy: ConformalPolicy | None, nc: float) -> dict[str, Any]:
    if policy is not None:
        return policy.decide(nc)
    answer = nc <= DEFAULT_NONCONFORMITY_THRESHOLD
    return {
        "schema": "sophia.conformal_decision.v1",
        "verdict": "answer" if answer else "abstain",
        "nonconformity": round(nc, 4),
        "threshold": DEFAULT_NONCONFORMITY_THRESHOLD,
        "riskBucket": "default",
        "coverageGuarantee": None,
        "candidateOnly": True,
        "level3Evidence": False,
    }


def ingest_one(
    claim: str,
    *,
    backend: Any = None,
    policy: ConformalPolicy | None = None,
    as_of: str,
    eval_cutoff: str | None,
    eval_prompts: set[str] | None,
    source_timestamp: str = "",
    valid_from: str = "",
    valid_until: str = "",
    risk: str = "normal",
    ingested_at: str | None = None,
) -> BeliefRow:
    """Run one live claim through the full fail-closed admission pipeline.

    A claim is INGESTED only when every gate agrees: the fact-check verdict is
    ``accepted`` AND the conformal decision is ``answer`` AND content-decontam,
    temporal-decontam, and valid-time all pass. Any other outcome quarantines or
    rejects; the default posture is not to ingest.
    """
    decision = fact_check_text(claim, **_backend_kwargs(backend))
    dec_dict = decision_to_dict(decision)
    nc = nonconformity(decision)
    conformal = _conformal_decide(policy, nc)

    content = sd.content_decontam(claim, eval_prompts)
    temporal = sd.temporal_decontam(source_timestamp, eval_cutoff)
    vt = sd.valid_time(valid_from, valid_until, as_of)

    confs = [c.confidence for c in decision.claims] or [0.0]
    confidence = round(min(confs), 4)
    sources: list[dict[str, Any]] = []
    for cd in dec_dict.get("claims", []):
        for layer in cd.get("layers", []):
            for ev in layer.get("evidence", []):
                sources.append(ev)

    gates_ok = content["ok"] and temporal["ok"] and vt["ok"]
    admitted = decision.verdict == "accepted" and conformal["verdict"] == "answer" and gates_ok

    if admitted:
        state, reason = "ingested", "verifier-accepted, conformal-answer, decontam+valid-time clean"
    elif decision.verdict == "rejected":
        state, reason = "rejected", "fact-check contradicted the claim"
    elif not gates_ok:
        # A gate veto is a hard quarantine even if the fact-check passed.
        failed = next(g for g in ((content, "content-decontam"), (temporal, "temporal-decontam"), (vt, "valid-time")) if not g[0]["ok"])
        state, reason = "quarantined", f"{failed[1]} veto: {failed[0]['reason']}"
    else:
        state, reason = "quarantined", f"held/abstained: verdict={decision.verdict}, conformal={conformal['verdict']}"

    return BeliefRow(
        claimId=claim_id(claim),
        claim=claim,
        ingestState=state,
        verdict=decision.verdict,
        confidence=confidence,
        nonconformity=nc,
        risk=risk,
        conformal=conformal,
        contentDecontam=content,
        temporalDecontam=temporal,
        validTime=vt,
        sources=sources,
        validFrom=valid_from,
        validUntil=valid_until,
        sourceTimestamp=source_timestamp,
        reason=reason,
        ingestedAt=(ingested_at or _now_iso()) if state == "ingested" else "",
        needsReverify=False,
        factCheck={"verdict": dec_dict["verdict"], "reason": dec_dict["reason"], "nClaims": len(dec_dict.get("claims", []))},
    )


def append_belief_rows(path: str | Path, rows: Iterable[BeliefRow | dict[str, Any]]) -> int:
    """Append belief rows to the store, de-duplicating by ``claimId``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                seen.add(str(json.loads(line).get("claimId", line)))
            except json.JSONDecodeError:
                continue
    written = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            d = row.to_dict() if isinstance(row, BeliefRow) else dict(row)
            key = str(d.get("claimId", ""))
            if key in seen:
                continue
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            seen.add(key)
            written += 1
    return written


def run_grounding(
    claims: list[dict[str, Any]],
    *,
    backend: Any = None,
    policy: ConformalPolicy | None = None,
    as_of: str | None = None,
    eval_cutoff: str | None,
    root: Path | None = None,
    store_path: str | Path | None = None,
    ingested_at: str | None = None,
) -> dict[str, Any]:
    """Ground a batch of live claims and (optionally) persist the ingested ones.

    ``claims`` items: ``{"claim": str, "sourceTimestamp"?, "validFrom"?,
    "validUntil"?, "risk"?}``. Returns a report; only ``ingested`` rows are written
    to ``store_path``.
    """
    as_of = as_of or _today()
    eval_prompts = sd.eval_surface(root)
    rows = [
        ingest_one(
            str(item.get("claim", "")),
            backend=backend,
            policy=policy,
            as_of=as_of,
            eval_cutoff=eval_cutoff,
            eval_prompts=eval_prompts,
            source_timestamp=str(item.get("sourceTimestamp", "")),
            valid_from=str(item.get("validFrom", "")),
            valid_until=str(item.get("validUntil", "")),
            risk=str(item.get("risk", "normal")),
            ingested_at=ingested_at,
        )
        for item in claims
        if str(item.get("claim", "")).strip()
    ]
    ingested = [r for r in rows if r.ingestState == "ingested"]
    n_written = append_belief_rows(store_path, ingested) if store_path else 0
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.ingestState] = counts.get(r.ingestState, 0) + 1
    return {
        "schema": SCHEMA,
        "candidateOnly": True,
        "level3Evidence": False,
        "asOf": as_of,
        "evalCutoff": eval_cutoff,
        "evalSurfaceSize": len(eval_prompts),
        "nClaims": len(rows),
        "counts": counts,
        "nWrittenToStore": n_written,
        "storePath": str(store_path) if store_path else None,
        "rows": [r.to_dict() for r in rows],
    }


def mark_stale(store_path: str | Path, as_of: str) -> dict[str, Any]:
    """Re-verify daemon primitive: flag ingested beliefs whose validity has lapsed.

    Reads the belief store, marks any ``ingested`` row whose ``validUntil`` is before
    ``as_of`` as ``stale`` / ``needsReverify`` (so a background pass re-runs the
    retrieve→verify loop), and rewrites the store. Reversible and non-destructive:
    the row and its provenance are kept, only the state changes.
    """
    path = Path(store_path)
    if not path.exists():
        return {"schema": "sophia.realtime_reverify.v1", "nStale": 0, "nRows": 0, "asOf": as_of, "candidateOnly": True}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    n_stale = 0
    for r in rows:
        if r.get("ingestState") != "ingested":
            continue
        vt = sd.valid_time(r.get("validFrom", ""), r.get("validUntil", ""), as_of)
        if not vt["ok"]:
            r["ingestState"] = "stale"
            r["needsReverify"] = True
            r["reason"] = f"marked stale at {as_of}: {vt['reason']}"
            n_stale += 1
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""), encoding="utf-8")
    return {"schema": "sophia.realtime_reverify.v1", "nStale": n_stale, "nRows": len(rows), "asOf": as_of, "candidateOnly": True}


__all__ = [
    "SCHEMA",
    "BELIEF_SCHEMA",
    "BeliefRow",
    "claim_id",
    "nonconformity",
    "ingest_one",
    "append_belief_rows",
    "run_grounding",
    "mark_stale",
]
