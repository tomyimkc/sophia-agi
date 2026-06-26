# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Verified reasoning-trace logger — a dual (fact + logic) stamp per reasoning step.

This is the unification layer the 2025-26 process-supervision / TRiSM audit-logging
literature asks for, built on Sophia's *existing* primitives so it adds no new
verdict vocabulary and no new gate:

  * the **fact** stamp reuses ``ConscienceDecision.verdict`` (allow|revise|retrieve|
    clarify|escalate|abstain|block) and the OKF provenance fields
    (``authorConfidence`` rank, ``effectiveConfidenceRank`` weakest-link, ``sources``);
  * the **logic** stamp reuses ``reasoning_compiler.CompileResult`` fields
    (``emittable``, ``contradictions``, ``laundered``, ``semanticsPreserved``) — the
    compiler's fail-closed type-check is the validity/consistency verdict;
  * a step is ``verified`` ONLY when ``fact.verdict in {allow, retrieve}`` AND
    ``logic.emittable``. It is never asserted; it is derived.

The record is a strict superset of the Langfuse-style span emitted by
``sophia_contract.trace.Tracer.span`` (same ``id`` scheme, same append-only JSONL
discipline), so it drops into the existing observability feeds without a new
transport. It is additionally chained into a tamper-evident Merkle/hash log:
every line carries ``prevHash`` (the previous line's ``_selfHash``) and its own
``_selfHash``, so any mutation of a prior line breaks the chain — the EU AI Act
Art. 12-style evidence bar at near-zero extra code.

No-overclaim discipline: every record carries ``candidateOnly=True``,
``level3Evidence=False`` and the ``boundary`` string. A "verified" step is a gate
outcome at the verifier's (recall, fpr) — bounded by ``deliberation_roofline`` —
NEVER a claim that the model "truly reasoned this way" (CoT faithfulness, see METR
2025 / Anthropic intervention work, is a separate causal question).

Observers only: a logger fault must never break a compile or a conscience check —
callers wrap ``record`` in the provided :func:`emit` helper which swallows errors
(the same "loud but non-fatal" audit convention the repo uses elsewhere). The
gates' own fail-closed behaviour is untouched.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.config import ROOT
from sophia_contract.stores import _append_jsonl, _read_jsonl

TRACE_LOG = ROOT / "agent" / "memory" / "contract" / "verified_traces.jsonl"
SCHEMA = "sophia.verified_trace.v1"
BOUNDARY = (
    "Sophia is an AGI-candidate verifier-gated epistemic framework; "
    "this trace is not proof of AGI."
)

#: Verdicts that count as fact-passed (mirrors the conscience kernel's "safe to
#: proceed" outcomes: allow = go, retrieve = go but verify first). Everything else
#: (revise/clarify/escalate/abstain/block) is a fact failure for the trace's purpose.
_FACT_OK = frozenset({"allow", "retrieve"})

#: Phases this logger is in scope for. Pretraining is DELIBERATELY excluded: per-step
#: logging at 1e12-token scale is infeasible, and the discipline targets the stages
#: where a single bad step can poison downstream belief (SFT/RL/curriculum/benchmark/
#: conscience). This is stated, not assumed.
PHASES = frozenset({"rlvr", "sft", "curriculum", "benchmark", "conscience"})


def _trace_id(seed: str) -> str:
    """Deterministic content-hashed id, matching ``sophia_contract.trace._trace_id``."""
    return f"vtrace_{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:24]}"


@dataclass
class VerifiedTrace:
    """One fact+logic-stamped reasoning step. schema ``sophia.verified_trace.v1``.

    ``fact``  shape: ``{verdict, source, authorConfidence, effectiveConfidenceRank, sources}``
    ``logic`` shape: ``{emittable, contradictions, laundered, semanticsPreserved}``

    Both are plain dicts so a trace can be recorded from any caller (compiler,
    conscience, hook bus) without coupling to those modules' dataclasses at record
    time. ``verified`` is a *derived* property — never set it by hand.
    """

    traceId: str
    runId: str
    phase: str
    stepIdx: int
    claimText: str
    claimKind: str  # source|derived|goal — mirrors reasoning_compiler.Claim.kind
    fact: dict[str, Any] = field(default_factory=dict)
    logic: dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0  # bounded [-1, 1]; MUST come from a verifier/gate, never self-score
    rewardProvenance: str = ""  # which gate/verifier produced the reward (for audit)
    candidateOnly: bool = True
    level3Evidence: bool = False
    boundary: str = BOUNDARY
    prevHash: str = ""  # "" for the first line; otherwise the prior line's _selfHash
    ts: str = ""
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise ValueError(
                f"phase must be one of {sorted(PHASES)} (pretraining is out of scope), "
                f"got {self.phase!r}"
            )
        # fail-closed clamp: a runaway reward can never escape [-1, 1].
        self.reward = max(-1.0, min(1.0, float(self.reward)))
        if not self.ts:
            self.ts = datetime.now(timezone.utc).isoformat()

    @property
    def verified(self) -> bool:
        """A step is verified iff the fact gate lets it proceed AND the logic gate
        can emit it. Both must hold — one good stamp does not rescue the other."""
        return (self.fact.get("verdict") in _FACT_OK) and bool(self.logic.get("emittable"))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["verified"] = self.verified
        return d


def _self_hash(record_with_prev: dict) -> str:
    """Stable hash of a record (excluding its own _selfHash field) for chaining."""
    payload = {k: v for k, v in record_with_prev.items() if k != "_selfHash"}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _last_self_hash(path: Path) -> str:
    events = _read_jsonl(path)
    return events[-1].get("_selfHash", "") if events else ""


def record(t: VerifiedTrace, *, path: Path | None = None) -> dict[str, Any]:
    """Append one verified trace. Returns ``{traceId, verified}``. Never mutates
    prior lines (append-only). Chains the record onto the prior line's hash."""
    path = path or TRACE_LOG
    t.prevHash = _last_self_hash(path)  # "" if the log is empty
    d = t.to_dict()
    d["_selfHash"] = _self_hash(d)
    _append_jsonl(path, d)
    return {"traceId": t.traceId, "verified": t.verified}


def emit(t: VerifiedTrace | None, *, path: Path | None = None) -> None:
    """Observer-safe record: swallow any logger fault so it can never break a
    compile or conscience check (the repo's "loud but non-fatal" audit convention).

    Set ``SOPHIA_TRACE_DEBUG=1`` to print the traceback instead — for diagnosing
    the logger itself, never for production gates.
    """
    if t is None:
        return
    try:
        record(t, path=path)
    except Exception:  # noqa: BLE001 - intentional: a logger fault must not break the caller
        if os.environ.get("SOPHIA_TRACE_DEBUG") == "1":
            traceback.print_exc(file=sys.stderr)


def verify_chain(path: Path | None = None) -> dict[str, Any]:
    """Re-derive the hash chain and report the first break (if any).

    A clean chain proves no prior line was mutated or inserted since it was
    written. Used by the ``sophia_trace_verify`` MCP tool and by the replay test
    — this is the tamper-evidence guarantee (EU AI Act Art. 12-style evidence).
    """
    path = path or TRACE_LOG
    events = _read_jsonl(path)
    prev = ""
    for i, ev in enumerate(events):
        if ev.get("prevHash", "") != prev:
            return {
                "chainIntact": False,
                "brokenAt": i,
                "reason": "prevHash mismatch",
                "nEvents": len(events),
            }
        expected = _self_hash(ev)
        if ev.get("_selfHash", "") != expected:
            return {
                "chainIntact": False,
                "brokenAt": i,
                "reason": "selfHash mismatch (line mutated)",
                "nEvents": len(events),
            }
        prev = ev.get("_selfHash", "")
    return {"chainIntact": True, "brokenAt": None, "nEvents": len(events)}


__all__ = [
    "VerifiedTrace",
    "PHASES",
    "BOUNDARY",
    "TRACE_LOG",
    "record",
    "emit",
    "verify_chain",
    "_trace_id",
]
