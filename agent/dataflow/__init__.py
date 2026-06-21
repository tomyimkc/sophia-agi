"""Out-of-prompt data-flow firewall (M2) — capability + taint enforcement.

The security boundary lives in deterministic CODE, not in a prompt label the model
could be talked out of (CaMeL principle). Untrusted data (retrieved docs, tool
output) carries a *taint*; every tool/sink call is checked against a per-tool
*capability* by a deterministic policy. The lethal trifecta — tainted data + a
write/egress sink — is blocked (or routed to human approval), regardless of what a
compromised model "decides".

This package is the enforcement core (taint, capabilities, manifest, firewall). It
is dependency-free and fully deterministic so it is unit-testable and scored by the
injection red-team (`eval/security/`). The dual-LLM privileged-planner /
quarantined-extractor split (M2.2) builds *on top* of this boundary.
"""

from agent.dataflow.capabilities import Decision, Effect, ToolCap, decide  # noqa: F401
from agent.dataflow.firewall import (  # noqa: F401
    FirewallBlocked,
    egress_blocked,
    firewalled,
    guard_call,
)
from agent.dataflow.interpreter import (  # noqa: F401
    Call,
    Concat,
    Const,
    Extract,
    Interpreter,
    InterpreterResult,
    Retrieve,
)
from agent.dataflow.manifest import TOOL_CAPS, cap_for  # noqa: F401
from agent.dataflow.extractor import (  # noqa: F401
    deterministic_extractor,
    quarantined_extractor,
)
from agent.dataflow.planner import (  # noqa: F401
    PlanError,
    model_planner,
    parse_plan,
    template_planner,
)
from agent.dataflow.taint import (  # noqa: F401
    TRUSTED,
    UNTRUSTED,
    Labeled,
    combine,
    taint_of,
    trusted,
    untrusted,
)
