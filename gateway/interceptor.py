"""The gateway interceptor — the fail-closed pipeline every tool call passes through.

Order (P0): lookup → role/scope authz → kill-switch + budget → BLP no-read-up →
dispatch → verify output → provenance-stamp (no-write-down) → audit/ROI → competence.
A result is returned ONLY when the verdict is ``accepted``; otherwise the verdict (with
held_reason/suggested_fix) is returned and the RAW OUTPUT IS WITHHELD.
"""

from __future__ import annotations

import hashlib

from gateway import firewall
from gateway.registry import Registry, ToolEntry
from gateway.verify_router import verify_output
from selfextend.competence_map import CompetenceMap
from sophia_contract import blp
from sophia_contract.errors import error
from sophia_contract.models import build_verdict
from sophia_contract.roles import ROLES_9
from sophia_contract.service import SophiaContract


class Gateway:
    def __init__(self, *, contract: "SophiaContract | None" = None,
                 registry: "Registry | None" = None, scopes=ROLES_9,
                 call_budget: "int | None" = None):
        self.contract = contract or SophiaContract()
        self.registry = registry or Registry()
        self.scopes = scopes
        self.call_budget = call_budget
        self._calls = 0
        self.competence = CompetenceMap()
        self._synth: dict = {}   # tool_id -> synthesized Rule (P4)

    def register(self, entry: "ToolEntry") -> "ToolEntry":
        # P1 firewall: a tool DESCRIPTION is untrusted; quarantine on injection.
        scan = firewall.scan_tool_description(entry.description)
        if not scan["clean"]:
            entry.risk_tier = "high"
            entry.injection_flags = scan["hits"]
        return self.registry.register(entry)

    # ---- P2 super-skills + universal verify -------------------------------------
    def register_skill(self, skill) -> "ToolEntry":
        return self.register(skill.to_tool())

    def eval_skill(self, skill, *, role=None, clearance="UNCLASSIFIED") -> dict:
        from gateway.skills import eval_skill as _eval
        return _eval(self, skill, role=role, clearance=clearance)

    def verify(self, content, *, verifier_ref: str = "grounding", sources=None,
               blp_level: str = "UNCLASSIFIED", clearance: str = "UNCLASSIFIED") -> dict:
        """Universal Verify API (#8): verify any content/output directly."""
        output = content if isinstance(content, dict) else {"answer": content, "sources": sources or []}
        key = f"verify:{hashlib.sha256(str(content).encode()).hexdigest()[:12]}"
        verdict, cid = verify_output(self.contract, verifier_ref=verifier_ref, output=output,
                                     tool_id="verify", args={}, blp_level=blp_level, role=None,
                                     clearance=clearance, idempotency_key=key)
        return {**verdict, "provenance_id": cid}

    # ---- P3 reliability registry ------------------------------------------------
    def rank_tools(self, *, role: "str | None" = None) -> list:
        return sorted(self.list_tools(role=role), key=lambda t: -t["reliability"])

    def best_tool(self, *, role: "str | None" = None) -> "str | None":
        ranked = self.rank_tools(role=role)
        return ranked[0]["tool_id"] if ranked else None

    # ---- P4 self-improving skills ----------------------------------------------
    def attach_synthesized_verifier(self, tool_id: str, rule) -> None:
        self._synth[tool_id] = rule

    def list_tools(self, *, role: "str | None" = None) -> list:
        return [e.public(self.competence.reliability(e.id)) for e in self.registry.list(role=role)]

    def describe(self) -> dict:
        return {
            "version": "1.2.0", "gateway": "sophia-gateway-p5",
            "tools": [e.id for e in self.registry.list()],
            "capabilities": ["gateway_describe", "list_tools", "call_tool", "verify",
                             "register_skill", "rank_tools", "federate_mcp",
                             "knowledge_tools", "synthesize_skill", "verified_consensus"],
            "schema_url": "docs/11-Platform/Sophia-Gateway.md",
        }

    def call_tool(self, tool_id: str, args: "dict | None" = None, *, role: "str | None" = None,
                  clearance: str = "UNCLASSIFIED", dry_run: bool = False,
                  idempotency_key: "str | None" = None) -> dict:
        args = args or {}
        tool = self.registry.get(tool_id)
        if tool is None:
            return error("BAD_REQUEST", f"unknown tool_id {tool_id!r}")
        if not blp.is_level(clearance):
            return error("BAD_REQUEST", f"clearance must be one of {blp.BLP_LEVELS}")

        # 1) Authorization: per-tool allow-list + role scope ceiling.
        if tool.allowed_roles is not None and (role is None or role not in tool.allowed_roles):
            return error("UNAUTHENTICATED", f"role {role!r} not permitted for {tool_id}")
        if role and self.scopes and role in self.scopes.roles:
            cap = self.scopes.roles[role].max_blp
            if blp.rank(tool.blp_level) > blp.rank(cap):
                return error("UNAUTHENTICATED", f"role {role!r} (cap {cap}) may not call a {tool.blp_level} tool")

        # 2) Firewall: scan call args for injection (untrusted input).
        hits = firewall.scan_args(args)
        if hits:
            self.competence.update(tool_id, False)
            return error("BAD_REQUEST", f"injection blocked by firewall: {hits}")

        # 3) Kill switch + budget (stop-and-report, before any side effect).
        if self.contract.health()["checks"].get("kill_switch_engaged"):
            return error("UNAVAILABLE", "gateway kill switch engaged", retryable=True)
        self._calls += 1
        if self.call_budget is not None and self._calls > self.call_budget:
            return {"tool_id": tool_id, "verdict": "held", "held_reason": "over_budget",
                    "suggested_fix": "raise the gateway call budget", "result": None}

        # 3) BLP no-read-up: the caller's clearance must dominate the tool's level.
        ru = blp.read_up_violation(clearance, tool.blp_level)
        if ru:
            self.competence.update(tool_id, False)
            return {"tool_id": tool_id, "verdict": "held", "held_reason": "blp_violation",
                    "reasons": [ru], "result": None,
                    "suggested_fix": "request access at the tool's clearance level"}

        # 4) Dispatch (honor dry-run for side-effecting tools).
        if dry_run and tool.side_effects in ("write", "external"):
            return {"tool_id": tool_id, "verdict": "held", "held_reason": "needs_human",
                    "dry_run": True, "result": None,
                    "suggested_fix": "approve to execute a side-effecting tool"}
        try:
            output = tool.handler(args)
        except Exception as exc:  # transport / tool failure
            self.competence.update(tool_id, False)
            return error("UNAVAILABLE", f"tool {tool_id} failed: {exc!r}", retryable=True)

        # 5) Verify the OUTPUT (the heart: only 'accepted' may surface).
        key = idempotency_key or f"gw:{tool_id}:{hashlib.sha256(str(args).encode()).hexdigest()[:12]}"
        synth = self._synth.get(tool_id)
        if synth is not None:                                  # P4: a synthesized verifier
            content = output.get("answer") if isinstance(output, dict) else str(output)
            ok = bool(synth.predict(str(content)))
            claim = self.contract.record_claim({"idempotency_key": key, "content": str(output),
                                                "sources": [tool_id], "blp_level": tool.blp_level})
            claim_id = claim.get("claim_id") if "error" not in claim else None
            verdict = build_verdict("accepted" if ok else "rejected",
                                    confidence=1.0 if ok else 0.9,
                                    reasons=[f"synthesized verifier {'accepted' if ok else 'rejected'}"],
                                    cited_evidence=[{"id": tool_id, "status": "ok"}])
        else:
            verdict, claim_id = verify_output(
                self.contract, verifier_ref=tool.verifier_ref, output=output, tool_id=tool_id,
                args=args, blp_level=tool.blp_level, role=role, clearance=clearance, idempotency_key=key)
        accepted = verdict.get("verdict") == "accepted"

        # 6) Audit trace + 7) competence update.
        self.contract.tracer.span(
            "gateway.call_tool", input={"tool_id": tool_id, "role": role, "clearance": clearance},
            output={"verdict": verdict.get("verdict"), "held_reason": verdict.get("held_reason")},
            level=("DEFAULT" if accepted else "WARNING"), metadata={"provenance_id": claim_id})
        self.competence.update(tool_id, accepted)

        resp = {
            "tool_id": tool_id, "verdict": verdict.get("verdict"),
            "provenance_id": claim_id, "confidence": verdict.get("confidence"),
            "reasons": verdict.get("reasons", []), "roi_estimate": verdict.get("roi_estimate"),
            "reliability": self.competence.reliability(tool_id),
        }
        if accepted:
            resp["result"] = output                       # surface ONLY when accepted
            resp["integrity"] = firewall.taint_label(tool.verifier_ref, tool.side_effects)
        else:
            resp["result"] = None                         # raw output withheld (fail-closed)
            if verdict.get("held_reason"):
                resp["held_reason"] = verdict["held_reason"]
            if verdict.get("suggested_fix"):
                resp["suggested_fix"] = verdict["suggested_fix"]
        return resp
