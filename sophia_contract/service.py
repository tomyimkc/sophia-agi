# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""SophiaContract — the governance service aihk-os consumes.

Deterministic and fail-closed by construction: no model is in the verify path, so a
fixed claim always yields the same verdict (that is what makes golden vectors a real
conformance gate). The decision pipeline, in strict order:

  lookup -> BLP no-read-up -> budget -> superseded -> human preference (feedback)
  -> evidence (no_source / refuted->rejected / stale) -> confidence -> risk-tiered
  auto-approve -> else escalate (held: needs_human).

Only ``accepted`` may be published. Every verdict is written to the decision log.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from sophia_contract import blp
from sophia_contract.errors import ContractError
from sophia_contract.models import (
    build_claim,
    build_verdict,
    content_fingerprint,
    validate_record_request,
)
from sophia_contract.queue import TaskQueue
from sophia_contract.scopes import ScopeRegistry
from sophia_contract.stores import ClaimStore, DecisionLog, PreferenceStore, Supersessions
from sophia_contract.trace import Tracer

CONTRACT_VERSION = "1.1.0"
SCHEMA_URL = "schema/contract-1.1.0.json"

AUTO_ACCEPT_CONFIDENCE = 0.75
LOW_RISK_LEVELS = ("UNCLASSIFIED",)  # only low-risk claims are eligible for auto-accept

# Founder-minutes a given outcome saves vs. doing it by hand — the per-task ROI basis.
REVIEW_MINUTES_SAVED = 5.0   # an answer that did not need human review
TRIAGE_MINUTES_SAVED = 3.0   # an auto-resolved reject/supersede that skipped manual triage

# Optional capabilities advertised in describe() — all implemented.
OPTIONAL_CAPABILITIES = ("explain_verdict", "batch_verify", "health",
                         "enqueue_task", "next_task", "trace")  # trace/queue added in 1.1.0
REQUIRED_METHODS = ("describe", "record_claim", "verify_claim")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _wire(fn):
    """Wrap a public method so a ContractError becomes the wire error shape."""

    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ContractError as exc:
            return exc.to_wire()

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


class SophiaContract:
    def __init__(
        self,
        *,
        store_dir: "Path | None" = None,
        clock: Callable[[], str] = _utc_now,
        signing_key: "str | None" = None,
        verify_budget: "int | None" = None,
        scopes: "ScopeRegistry | None" = None,
        tracing: bool = True,
    ):
        d = Path(store_dir) if store_dir else None
        self.clock = clock
        self.signing_key = signing_key
        self.verify_budget = verify_budget
        self._verify_count = 0
        self.scopes = scopes or ScopeRegistry()  # empty == unrestricted (opt-in)
        self.claims = ClaimStore(d / "claims.jsonl" if d else None)
        self.decisions = DecisionLog(d / "decisions.jsonl" if d else None)
        self.preferences = PreferenceStore(d / "preferences.jsonl" if d else None)
        self.supersessions = Supersessions(d / "supersessions.jsonl" if d else None)
        self.tasks = TaskQueue(d / "tasks.jsonl" if d else None, clock=clock)
        self.tracer = Tracer(d / "traces.jsonl" if d else None, clock=clock, enabled=tracing)
        self._killswitch_path = (d / "killswitch.json") if d else None
        self._killswitch = self._load_killswitch()

    # ----------------------------------------------------------------- handshake
    @_wire
    def describe(self) -> dict:
        """Handshake: the version, advertised capabilities, schema location, and any
        active deprecations. aihk-os pins against ``version`` (semver)."""
        return {
            "version": CONTRACT_VERSION,
            "capabilities": [*REQUIRED_METHODS, *OPTIONAL_CAPABILITIES],
            "schema_url": SCHEMA_URL,
            "deprecations": [],  # none in 1.0.0; a field lives one full MAJOR before removal
        }

    # ------------------------------------------------------- kill switch (admin)
    def _load_killswitch(self) -> dict:
        if self._killswitch_path and self._killswitch_path.exists():
            import json
            try:
                return json.loads(self._killswitch_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {"engaged": False}
        return {"engaged": False}

    def _persist_killswitch(self) -> None:
        if self._killswitch_path:
            import json
            self._killswitch_path.parent.mkdir(parents=True, exist_ok=True)
            self._killswitch_path.write_text(json.dumps(self._killswitch), encoding="utf-8")

    def engage_kill_switch(self, reason: str = "manual", by: str = "founder") -> dict:
        """Halt all record/verify; calls return UNAVAILABLE (retryable). Durable."""
        self._killswitch = {"engaged": True, "reason": reason, "by": by, "at": self.clock()}
        self._persist_killswitch()
        return dict(self._killswitch)

    def release_kill_switch(self, by: str = "founder") -> dict:
        self._killswitch = {"engaged": False, "by": by, "at": self.clock()}
        self._persist_killswitch()
        return dict(self._killswitch)

    def _guard_kill_switch(self) -> None:
        if self._killswitch.get("engaged"):
            raise ContractError("UNAVAILABLE",
                                f"kill switch engaged: {self._killswitch.get('reason', 'halted')}",
                                retryable=True)

    # -------------------------------------------------------------- record_claim
    @_wire
    def record_claim(self, request: dict) -> dict:
        """Idempotently record a claim. Enforces BLP no-write-down at record time:
        a derived claim must be at least as classified as its parents, else
        BLP_VIOLATION (never silently downgraded).

        Optional (1.1.0, additive): ``role`` (capability-scope enforcement) and
        ``dry_run`` (validate + compute claim_id WITHOUT persisting)."""
        self._guard_kill_switch()
        fields = validate_record_request(request)
        role = request.get("role")
        dry_run = bool(request.get("dry_run"))
        self.scopes.check(role, "record_claim", blp_level=fields["blp_level"], dry_run=dry_run)

        parent_levels = []
        for pid in fields["parents"]:
            parent = self.claims.get_by_id(pid)
            if parent is not None:
                parent_levels.append(parent["blp_level"])
        wd = blp.write_down_violation(fields["blp_level"], parent_levels)
        if wd:
            raise ContractError("BLP_VIOLATION", wd)

        claim = build_claim(fields, created_at=self.clock(), signing_key=self.signing_key)
        if dry_run:
            self.tracer.span("record_claim", input={"idempotency_key": fields["idempotency_key"],
                             "dry_run": True}, output={"claim_id": claim["claim_id"]},
                             metadata={"role": role})
            return {**claim, "dry_run": True}
        stored, _created = self.claims.record(claim, fields["idempotency_key"])
        self.tracer.span("record_claim", input={"idempotency_key": fields["idempotency_key"]},
                         output={"claim_id": stored["claim_id"]}, metadata={"role": role})
        return stored

    # -------------------------------------------------------------- verify_claim
    @_wire
    def verify_claim(self, request: dict, *, clearance: str = "UNCLASSIFIED") -> dict:
        """Verify a recorded claim and return an explainable Verdict. Only
        ``accepted`` may be published; everything else fails closed.

        Optional (1.1.0, additive): ``role`` in the request enforces capability
        scopes; every Verdict carries a ``roi_estimate`` of founder-minutes saved."""
        self._guard_kill_switch()
        if not isinstance(request, dict) or not request.get("claim_id"):
            raise ContractError("BAD_REQUEST", "claim_id is required")
        if not blp.is_level(clearance):
            raise ContractError("BAD_REQUEST", f"clearance must be one of {blp.BLP_LEVELS}")
        self.scopes.check(request.get("role"), "verify_claim", blp_level=clearance)
        claim_id = request["claim_id"]
        claim = self.claims.get_by_id(claim_id)
        if claim is None:
            raise ContractError("BAD_REQUEST", f"unknown claim_id {claim_id!r}")

        verdict = self._decide(claim, clearance)
        verdict["roi_estimate"] = self._roi(verdict)
        self.decisions.append({
            "at": self.clock(), "claim_id": claim_id, "clearance": clearance,
            "verdict": verdict["verdict"], "confidence": verdict["confidence"],
            "held_reason": verdict.get("held_reason"),
            "founder_minutes_saved": verdict["roi_estimate"]["founder_minutes_saved"],
        })
        self.tracer.span("verify_claim", input={"claim_id": claim_id, "clearance": clearance},
                         output={"verdict": verdict["verdict"], "confidence": verdict["confidence"],
                                 "held_reason": verdict.get("held_reason")},
                         level=("WARNING" if verdict["verdict"] != "accepted" else "DEFAULT"),
                         metadata={"role": request.get("role")})
        return verdict

    @staticmethod
    def _roi(verdict: dict) -> dict:
        """Founder-minutes saved by this outcome vs. handling it by hand."""
        v = verdict["verdict"]
        human = any("human-reviewed" in r for r in verdict["reasons"])
        if v == "accepted":
            return {"founder_minutes_saved": REVIEW_MINUTES_SAVED,
                    "basis": "reused prior human ruling" if human else "auto-accepted without review"}
        if v in ("rejected", "superseded"):
            return {"founder_minutes_saved": TRIAGE_MINUTES_SAVED,
                    "basis": f"auto-{v}; skipped manual triage"}
        return {"founder_minutes_saved": 0.0, "basis": "held for human review"}

    def _decide(self, claim: dict, clearance: str) -> dict:
        claim_id = claim["claim_id"]
        fp = content_fingerprint(claim["content"])

        # 1) BLP no-read-up — fail closed, never downgraded.
        ru = blp.read_up_violation(clearance, claim["blp_level"])
        if ru:
            return build_verdict("held", confidence=0.0, reasons=[ru],
                                 cited_evidence=[], held_reason="blp_violation",
                                 suggested_fix="request review at the claim's clearance level")

        # 2) Budget cap — stop-and-report.
        self._verify_count += 1
        if self.verify_budget is not None and self._verify_count > self.verify_budget:
            return build_verdict("held", confidence=0.0,
                                 reasons=[f"verify budget of {self.verify_budget} exhausted"],
                                 cited_evidence=[], held_reason="over_budget",
                                 suggested_fix="raise the budget cap or process in the next window")

        # 3) Superseded.
        successor = self.supersessions.successor(claim_id)
        if successor:
            return build_verdict("superseded", confidence=1.0,
                                 reasons=[f"superseded by {successor}"],
                                 cited_evidence=[], supersedes=successor,
                                 suggested_fix=f"verify {successor} instead")

        # 4) Human preference (the feedback loop): a prior human ruling wins and
        #    short-circuits review next time.
        pref = self.preferences.lookup(claim_id=claim_id, fingerprint=fp)
        if pref:
            return build_verdict(pref["verdict"], confidence=1.0,
                                 reasons=[f"human-reviewed by {pref.get('reviewer', 'founder')}"
                                          + (f": {pref['note']}" if pref.get("note") else "")],
                                 cited_evidence=self._evidence(claim),
                                 held_reason=("needs_human" if pref["verdict"] == "held" else None))

        # 5) Evidence gates.
        sources = claim["sources"]
        if not sources:
            return build_verdict("held", confidence=0.2, reasons=["claim has no sources"],
                                 cited_evidence=[], held_reason="no_source",
                                 suggested_fix="attach at least one verifiable source")
        refuted = [s for s in sources if s["status"] in ("refuted", "invalid")]
        if refuted:
            return build_verdict("rejected", confidence=0.9,
                                 reasons=[f"source {s['id']} is {s['status']}" for s in refuted],
                                 cited_evidence=self._evidence(claim),
                                 suggested_fix="replace refuted/invalid sources with valid ones")
        ok_sources = [s for s in sources if s["status"] == "ok"]
        if not ok_sources:  # remaining sources are all stale
            return build_verdict("held", confidence=0.3,
                                 reasons=["all sources are stale"],
                                 cited_evidence=self._evidence(claim), held_reason="stale_source",
                                 suggested_fix="refresh the source to a current version")

        # 6) Deterministic confidence + 7) risk-tiered auto-approve.
        confidence, reasons = self._confidence(claim, ok_sources)
        cited = self._evidence(claim)
        low_risk = claim["blp_level"] in LOW_RISK_LEVELS
        if low_risk and confidence >= AUTO_ACCEPT_CONFIDENCE and cited:
            return build_verdict("accepted", confidence=confidence,
                                 reasons=reasons + ["low-risk, high-confidence, cited -> auto-accepted"],
                                 cited_evidence=cited)
        # 8) Escalate the ambiguous / higher-risk to a human.
        why = "higher-risk classification requires human review" if not low_risk else \
              "confidence below auto-accept threshold"
        return build_verdict("held", confidence=confidence,
                             reasons=reasons + [why], cited_evidence=cited,
                             held_reason="needs_human",
                             suggested_fix="add a corroborating source or request founder review")

    @staticmethod
    def _evidence(claim: dict) -> "list[dict]":
        return [{"id": s["id"], "status": s["status"], **({"date": s["date"]} if "date" in s else {})}
                for s in claim["sources"]]

    @staticmethod
    def _confidence(claim: dict, ok_sources: list) -> "tuple[float, list[str]]":
        conf = 0.5
        reasons = []
        conf += 0.25
        reasons.append(f"{len(ok_sources)} valid source(s)")
        if len(ok_sources) >= 2:
            conf += 0.15
            reasons.append("corroborated by multiple sources")
        if claim["parents"]:
            conf += 0.05
            reasons.append("has provenance lineage (parents)")
        return (max(0.0, min(1.0, round(conf, 4))), reasons)

    # ------------------------------------------------------- optional capabilities
    @_wire
    def explain_verdict(self, request: dict, *, clearance: str = "UNCLASSIFIED") -> dict:
        """Capability: the verdict plus a one-line trace of the rule path taken."""
        verdict = self.verify_claim(request, clearance=clearance)
        if "error" in verdict:
            return verdict
        trace = " | ".join(verdict["reasons"]) or "(no reasons)"
        return {**verdict, "explanation": f"verdict={verdict['verdict']} :: {trace}"}

    @_wire
    def batch_verify(self, request: dict, *, clearance: str = "UNCLASSIFIED") -> dict:
        """Capability: verify many claim_ids in one call. Each result is an independent
        Verdict or error; one bad id never fails the batch."""
        ids = request.get("claim_ids")
        if not isinstance(ids, list) or not ids:
            raise ContractError("BAD_REQUEST", "claim_ids must be a non-empty array")
        return {"results": [
            {"claim_id": cid, "result": self.verify_claim({"claim_id": cid}, clearance=clearance)}
            for cid in ids
        ]}

    @_wire
    def enqueue_task(self, request: dict) -> dict:
        """Capability (1.1.0): durably + idempotently enqueue work. Same
        idempotency_key returns the same task_id. Blocked by the kill switch."""
        self._guard_kill_switch()
        self.scopes.check(request.get("role"), "enqueue_task")
        return self.tasks.enqueue(request)

    @_wire
    def next_task(self, request: "dict | None" = None) -> dict:
        """Capability (1.1.0): lease the oldest pending task, or {task: null}."""
        self._guard_kill_switch()
        lease_by = (request or {}).get("lease_by", "worker")
        task = self.tasks.next_task(lease_by=lease_by)
        return {"task": task}

    def complete_task(self, task_id: str, *, result=None, state: str = "done") -> dict:
        return self.tasks.complete(task_id, result=result, state=state)

    def task_status(self, task_id: str) -> dict:
        return self.tasks.status(task_id)

    @_wire
    def trace(self, request: "dict | None" = None) -> dict:
        """Capability (1.1.0): the Langfuse-compatible trace events recorded so far."""
        return {"events": self.tracer.events()}

    @_wire
    def health(self) -> dict:
        """Capability: liveness + self-diagnostics for unattended operation."""
        checks = {
            "claims_store": True,
            "decision_log": True,
            "preference_store": True,
            "kill_switch_engaged": bool(self._killswitch.get("engaged")),
            "pending_tasks": self.tasks.pending_count(),
            "tracing_enabled": self.tracer.enabled,
            "budget_remaining": (None if self.verify_budget is None
                                 else max(0, self.verify_budget - self._verify_count)),
        }
        status = "degraded" if self._killswitch.get("engaged") else "ok"
        return {"status": status, "version": CONTRACT_VERSION, "checks": checks}

    # ----------------------------------------------------- feedback loop (admin)
    def record_human_verdict(self, *, claim_id: str, verdict: str, note: str = "",
                             reviewer: str = "founder") -> dict:
        """Record a human ruling so future verifies short-circuit to it. Inspectable
        and hand-editable in preferences.jsonl."""
        claim = self.claims.get_by_id(claim_id)
        fp = content_fingerprint(claim["content"]) if claim else ""
        return self.preferences.record_human_verdict(
            claim_id=claim_id, fingerprint=fp, verdict=verdict, note=note, reviewer=reviewer)

    def mark_superseded(self, old_claim_id: str, new_claim_id: str) -> None:
        self.supersessions.mark(old_claim_id, new_claim_id)
