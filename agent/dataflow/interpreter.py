"""Constrained interpreter — the dual-LLM / CaMeL execution model (M2.2).

M2 gave a firewall but left two holes the adversarial review flagged: taint did not
propagate through ordinary Python (f-strings launder), and untrusted data could in
principle steer control flow. Those holes exist because the *model* manipulates raw
data. This interpreter removes the model from the data path:

  - A **privileged planner** (P-LLM) turns the trusted user request into a PLAN —
    a fixed sequence of steps over *symbolic variables*. It never sees untrusted
    data values, so injected text in a document can never become a new step.
  - A **quarantined extractor** (Q-LLM) may read untrusted content, but its output
    is treated as *data* (Labeled untrusted), never as control flow, and it has no
    tools.
  - This **interpreter** executes the plan deterministically. Variables hold
    `Labeled` values; every operation propagates taint via `combine`, so a value
    derived from untrusted input stays untrusted no matter how it is transformed —
    soundly, because the interpreter (not the model) does the transformation. Every
    tool call is checked by the M2 firewall, so a tainted value cannot reach a
    write/egress sink without approval.

Security properties (falsifiable, see tests/test_interpreter.py):
  1. Taint propagates through every step (no laundering).
  2. Control-flow integrity: untrusted data cannot cause a tool call not in the plan.
  3. Sink enforcement: a tainted value into a write/egress sink is blocked.

The planner and extractor are injectable (mocked in tests / CI) — the security
lives in this deterministic interpreter, not in trusting either LLM. Honest scope:
the instruction set is small (the planner's expressiveness, not the safety, is the
limit), and the planner/user request is the trust root.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agent.dataflow.capabilities import Effect
from agent.dataflow.firewall import guard_call
from agent.dataflow.manifest import cap_for
from agent.dataflow.taint import Labeled, TRUSTED, UNTRUSTED, combine, trusted, unwrap


# --- Plan steps (the only control flow; produced by the trusted planner) ------ #
@dataclass(frozen=True)
class Const:
    """Bind a trusted literal from the planner."""
    var: str
    value: Any


@dataclass(frozen=True)
class Retrieve:
    """Call a READ tool; its result is external content → tainted (untrusted)."""
    var: str
    tool: str
    query: Any            # a literal or a var name (resolved trusted/var)


@dataclass(frozen=True)
class Extract:
    """Run the quarantined extractor (Q-LLM) over an untrusted source. Output is
    DATA (untrusted), never control flow."""
    var: str
    src: str              # var holding the untrusted source
    instruction: str


@dataclass(frozen=True)
class Concat:
    """Join parts (var names or literals); taint is the union of all parts."""
    var: str
    parts: list


@dataclass(frozen=True)
class Call:
    """Call a tool with args (var names or literals); firewall-enforced."""
    var: str
    tool: str
    args: list


@dataclass
class InterpreterResult:
    env: dict = field(default_factory=dict)
    calls: list = field(default_factory=list)        # tools actually executed
    blocked: list = field(default_factory=list)      # (tool, reason) the firewall stopped

    def value(self, var: str) -> Any:
        return unwrap(self.env.get(var))

    def taint(self, var: str):
        v = self.env.get(var)
        return v.taint if isinstance(v, Labeled) else TRUSTED


class Interpreter:
    """Execute a plan with taint-tracked variables and firewall-gated tool calls."""

    def __init__(self, *, tools: dict, extractor: "Callable | None" = None,
                 approver=None, profile=None, approve_sinks: bool = False):
        self.tools = tools                 # name -> callable(*plain_args) -> result
        self.extractor = extractor         # (instruction, src_text) -> str  (Q-LLM)
        self.approver = approver
        self.profile = profile
        # Defense in depth for attacker-influenceable requests: require explicit
        # approval for EVERY write/egress call, even with trusted args (so a
        # malicious planner alone cannot trigger an unapproved side effect).
        self.approve_sinks = approve_sinks

    def _resolve(self, ref: Any, env: dict, declared: set) -> Labeled:
        """A ref is a var name (use its Labeled value) or a literal (trusted). A var
        that the plan DECLARES but never bound (its producing step was blocked) fails
        CLOSED — returned as untrusted, never as a trusted literal of its own name."""
        if isinstance(ref, str):
            if ref in env:
                return env[ref]
            if ref in declared:
                return Labeled(None, taint=UNTRUSTED, origin=f"unbound:{ref}")
        return trusted(ref)

    def run(self, plan: list) -> InterpreterResult:
        env: dict = {}
        res = InterpreterResult(env=env)
        declared = {step.var for step in plan if hasattr(step, "var")}
        for step in plan:
            if isinstance(step, Const):
                env[step.var] = trusted(step.value, origin="planner")

            elif isinstance(step, Retrieve):
                # Defense in depth: a Retrieve must name a READ tool, and the
                # firewall sees the (Labeled) arg and its decision is honoured.
                if cap_for(step.tool).effect != Effect.READ:
                    res.blocked.append((step.tool, "Retrieve must name a READ tool"))
                    continue
                qarg = self._resolve(step.query, env, declared)
                decision = guard_call(step.tool, (qarg,), profile=self.profile)
                if not decision.allowed:
                    res.blocked.append((step.tool, decision.reason))
                    continue
                raw = self.tools[step.tool](unwrap(qarg))
                env[step.var] = Labeled(raw, taint=UNTRUSTED, origin=step.tool)

            elif isinstance(step, Extract):
                src = self._resolve(step.src, env, declared)
                out = self.extractor(step.instruction, unwrap(src)) if self.extractor else ""
                # Q-LLM output is DATA, always untrusted (it read untrusted content).
                env[step.var] = Labeled(out, taint=combine(src) | UNTRUSTED, origin="extractor")

            elif isinstance(step, Concat):
                parts = [self._resolve(p, env, declared) for p in step.parts]
                text = "".join(str(unwrap(p)) for p in parts)
                env[step.var] = Labeled(text, taint=combine(*parts), origin="concat")  # taint propagates

            elif isinstance(step, Call):
                args = [self._resolve(a, env, declared) for a in step.args]
                # Defense in depth: when enabled, every write/egress sink needs
                # explicit approval even if its args are trusted.
                if self.approve_sinks and cap_for(step.tool).effect != Effect.READ:
                    granted = False
                    if self.approver is not None:
                        try:
                            granted = self.approver(step.tool, tuple(args), {}, None) is True
                        except Exception:
                            granted = False
                    if not granted:
                        res.blocked.append((step.tool, "approve_sinks: write/egress requires approval"))
                        continue
                decision = guard_call(step.tool, tuple(args), approver=self.approver, profile=self.profile)
                if decision.action == "require_hitl":
                    granted = False
                    if self.approver is not None:
                        try:
                            granted = self.approver(step.tool, tuple(args), {}, decision) is True
                        except Exception:
                            granted = False
                    if not granted:
                        res.blocked.append((step.tool, "HITL not approved: " + decision.reason))
                        continue
                elif not decision.allowed:
                    res.blocked.append((step.tool, decision.reason))
                    continue
                result = self.tools[step.tool](*[unwrap(a) for a in args])
                # Fail-safe: a tool's OUTPUT reflects external/world state, so it is
                # untrusted regardless of effect (an EGRESS result is attacker-
                # controlled web content; a WRITE may echo merged world state). This
                # closes the egress-fetch-then-store laundering path.
                env[step.var] = Labeled(result, taint=combine(*args) | UNTRUSTED, origin=step.tool)
                res.calls.append(step.tool)

            else:  # unknown step type — refuse rather than guess (fail closed)
                res.blocked.append((getattr(step, "tool", type(step).__name__), "unknown plan step"))
        return res
