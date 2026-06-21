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
                 approver=None, profile=None):
        self.tools = tools                 # name -> callable(*plain_args) -> result
        self.extractor = extractor         # (instruction, src_text) -> str  (Q-LLM)
        self.approver = approver
        self.profile = profile

    def _resolve(self, ref: Any, env: dict) -> Labeled:
        """A ref is a var name (use its Labeled value) or a literal (trusted)."""
        if isinstance(ref, str) and ref in env:
            return env[ref]
        return trusted(ref)

    def run(self, plan: list) -> InterpreterResult:
        env: dict = {}
        res = InterpreterResult(env=env)
        for step in plan:
            if isinstance(step, Const):
                env[step.var] = trusted(step.value, origin="planner")

            elif isinstance(step, Retrieve):
                query = unwrap(self._resolve(step.query, env))
                # READ tool — firewall allows reads; its OUTPUT is untrusted.
                guard_call(step.tool, (query,), profile=self.profile)
                raw = self.tools[step.tool](query)
                env[step.var] = Labeled(raw, taint=UNTRUSTED, origin=step.tool)

            elif isinstance(step, Extract):
                src = self._resolve(step.src, env)
                out = self.extractor(step.instruction, unwrap(src)) if self.extractor else ""
                # Q-LLM output is DATA, inheriting the source taint (≥ untrusted).
                env[step.var] = Labeled(out, taint=combine(src, Labeled(out, UNTRUSTED)), origin="extractor")

            elif isinstance(step, Concat):
                parts = [self._resolve(p, env) for p in step.parts]
                text = "".join(str(unwrap(p)) for p in parts)
                env[step.var] = Labeled(text, taint=combine(*parts), origin="concat")  # taint propagates

            elif isinstance(step, Call):
                args = [self._resolve(a, env) for a in step.args]
                decision = guard_call(step.tool, tuple(args), approver=self.approver, profile=self.profile)
                if not decision.allowed:
                    res.blocked.append((step.tool, decision.reason))
                    continue
                result = self.tools[step.tool](*[unwrap(a) for a in args])
                effect = cap_for(step.tool).effect
                # A sink's own output is trusted (we produced it); a read's is untrusted.
                taint = UNTRUSTED if effect == Effect.READ else combine(*args)
                env[step.var] = Labeled(result, taint=taint, origin=step.tool)
                res.calls.append(step.tool)

            else:  # unknown step type — refuse rather than guess (fail closed)
                res.blocked.append((getattr(step, "tool", type(step).__name__), "unknown plan step"))
        return res
