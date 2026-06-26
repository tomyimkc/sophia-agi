# Defense-in-Depth Governance: two gates, two boundaries

> Sophia gates **claims** (does this assertion have honest provenance?).
> A repo-boundary gate gates **changes** (is this agent allowed to touch this file?).
> They are orthogonal. Stack them and an autonomous agent is fenced on both axes.

Sophia's contract gate (`record_claim → verify_claim`, fail-closed, only `accepted`
publishes) answers *"is this output true to its sources?"* It says nothing about
*"was this agent even allowed to write here?"* — file paths, secret leakage,
cross-module dependency violations. That is a different boundary, enforced by a
different tool.

A clean reference for that second boundary is
[`agentic-workflow-governance-tools`](https://github.com/chrisipanaque/agentic-workflow-governance-tools)
(C++20, config-driven, single binary, zero runtime deps): it intercepts an agent's
`git diff` at the **repository boundary**, validates the changed files against
JSON policy (forbidden paths, dependency-isolation rules), and returns a decision
as an exit code (`0` = proceed, `1` = needs a human). No recompilation to change
policy; deterministic; audit-logged.

## The two boundaries

| | Repo-boundary gate | Sophia claim gate |
|---|---|---|
| **Question** | May this agent change this file? | Is this claim faithful to its sources? |
| **Unit** | A `git diff` (paths, `#include`s) | A claim (`content` + `sources` + `parents`) |
| **Blocks on** | forbidden paths, secret files, cross-module deps | unsourced / stale / lineage-merged / BLP violations |
| **Verdict** | exit code `0` / `1` | `accepted · held · rejected · superseded` |
| **When** | pre-commit / CI, before code lands | pre-publish, before output ships |
| **Reference** | `chrisipanaque/agentic-workflow-governance-tools` | `sophia_contract` + [CONTRACT.md](../../CONTRACT.md) |

Neither subsumes the other. A repo gate happily commits a perfectly-scoped diff
whose *content* fabricates a citation; Sophia happily accepts a well-sourced claim
that an agent had no business writing into `config/forbidden-paths.json`. You want
both.

## The composed pipeline

```text
agent proposes change
        │
        ▼
┌───────────────────────┐   exit 1   ┌──────────────────┐
│  repo-boundary gate   │──────────▶ │  human / reject  │
│  (scan-diff vs policy)│            └──────────────────┘
└───────────┬───────────┘
        exit 0 │ (paths & deps OK)
        ▼
┌───────────────────────┐   held     ┌──────────────────┐
│  Sophia claim gate    │──────────▶ │  human review    │
│  record → verify_claim│  rejected  │  (route_after_   │
└───────────┬───────────┘──────────▶ │   verify)        │
   accepted │                        └──────────────────┘
        ▼
     publish / merge
```

Fail-closed at **both** stages: a block at either gate stops the flow. The repo
gate runs first (cheap, deterministic, no model), so a scope violation never even
reaches claim verification.

### Wiring sketch (CI step)

```bash
# Stage 1 — repo boundary (chrisipanaque/agentic-workflow-governance-tools)
govtools scan-diff || { echo "scope violation — human review"; exit 1; }

# Stage 2 — Sophia claim boundary (this repo)
python - <<'PY'
from sophia_contract import SophiaContract
from sophia_contract.langgraph_nodes import run_contract_flow
final = run_contract_flow(SophiaContract(), {
    "idempotency_key": "ci-<sha>", "content": OUTPUT_TEXT, "sources": SOURCES})
raise SystemExit(0 if final["route"] == "publish" else 1)
PY
```

Both stages emit audit records. With the OTLP exporter
([`sophia_contract/otel_export.py`](../../sophia_contract/otel_export.py)) the
Sophia verdicts land in the same OpenTelemetry collector as the rest of the agent
trace, so the two gates' decisions read as one timeline — see
[Observability](#see-also).

## Why this pairing, specifically

The repo-boundary tool's design choices mirror Sophia's contract on purpose, which
makes them easy to run side by side:

- **Config-driven, no recompile** — its JSON policy ↔ Sophia's `ROLES_9` scopes.
- **Exit-code decisions** — composable in any CI ↔ Sophia's `route` (`publish | review | reject`).
- **Deterministic + audit-logged** — both are inspectable after the fact, not black boxes.

## See also

- [CONTRACT.md](../../CONTRACT.md) — the stable claim-gate interface.
- [Observability-OTel.md](../09-Agent/Observability-OTel.md) — export verdicts as OpenTelemetry spans.
- [`sophia_contract/langgraph_nodes.py`](../../sophia_contract/langgraph_nodes.py) — the claim gate as drop-in LangGraph nodes.
