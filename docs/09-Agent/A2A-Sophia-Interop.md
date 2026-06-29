# A2A (Agent2Agent) interoperability for Sophia

> How Sophia agents talk to **each other** and to **external LLM agents**, across machines,
> for long-running work — without giving up the fail-closed / provenance discipline.

## 1. The two-layer model (the key idea)

There are **two complementary protocols**, not competitors. Sophia already speaks the first;
this work adds the second.

| Layer | Protocol | "X talks to Y" | In this repo |
|---|---|---|---|
| Tools / context | **MCP** (Anthropic; de-facto standard, ~18k servers) | agent → **tool** | `gateway/federation.py` (`HttpMcpTransport`, `tools/call`) + `sophia_mcp/server.py` |
| Coordination | **A2A** (Google → Linux Foundation, v1 in 2026) | agent → **agent** | `agent/a2a.py` (this work) |

A2A is JSON-RPC 2.0 over HTTP(S), **Agent Card** discovery, a standard **task lifecycle**
(`submitted → working → completed/failed/canceled`), and enterprise auth (apiKey / OAuth2 /
mTLS). IBM's ACP merged into A2A under the Linux Foundation (Sept 2025); the field is
consolidating on **A2A for agents + MCP for tools**. ANP / AGNTCY ("Internet of Agents")
target decentralized discovery and are worth tracking, not betting on yet.

## 2. Where Sophia is today

Sophia's multi-agent system is real but **intra-process**:

```
task → SwarmRouter.decide(task) → SwarmPlan{teams, k, budgets} → subagent.delegate() → fail-closed reduce
        (agent/swarm_router.py)                                    (agent/subagent.py)
```

`team_agents.py` / `council_deliberate.py` do *deliberation* (one model, many personas);
`reasoning/belief_allreduce.py` shares belief state. A2A is what turns those in-process
children into **networked peers** other agents (yours or third-party) can discover and
delegate to over a standard wire.

## 3. Architecture (what `agent/a2a.py` adds)

```
                       ┌──────────────────────── Spark-Sophia (A2A peer) ───────────────────────┐
  caller agent         │  A2AServer.handle(JSON-RPC)                                            │
  (Sophia / Claude /   │    agent/getCard → AgentCard (discovery, least-privilege skills)       │
   GPT / Hermes…)      │    message/send  → A2ATask(submitted→working)                          │
        │  A2AClient    │                     → runner = run_swarm() → fail-closed synthesis      │
        ├──────────────▶│                     → GATE our own answer → artifacts → completed       │
        │  transport    │    tasks/get     → poll task state (async-shaped)                      │
        │ (Stub|Http)   │    tasks/cancel  → cancel non-terminal task                            │
        ◀──────────────┤  every result carries cost_usd + est_cost_steps + gate verdict          │
  delegate_to_peer()   └────────────────────────────────────────────────────────────────────────┘
   → run PEER answer through OUR gate (untrusted external data) → GateVerdict {accept|abstain|block}
```

Core types (all in `agent/a2a.py`):

- **`AgentCard` / `AgentSkill`** — discovery. `sophia_agent_card(url)` advertises only scoped
  skills (`swarm.delegate`, `provenance.validate`, `epistemic.gate_check`) plus the
  `x-sophia.provenanceDiscipline` flag so peers know answers are verifier-gated and may abstain.
  Validated by `schema/agent-card-1.0.0.json`.
- **`A2AServer`** — wraps the existing swarm behind the A2A contract. `handle(request)` is a
  pure JSON-RPC dispatch (drop behind any HTTP handler). It **gates its own answer** before
  returning and **fails closed** on an empty task (no swarm spawned on noise).
- **`A2AClient` + transports** — `StubA2ATransport` (offline, in-process; mirrors
  `federation.StubTransport`) and `HttpA2ATransport` (live, urllib; mirrors `HttpMcpTransport`).
  `delegate_to_peer()` is the **trust boundary**: a peer's output is untrusted external data, so
  it is run through *our* gate (and a peer's own abstain/block is propagated, never upgraded).
- **MCP-as-agent bridge** — `A2AMcpTransport` + `agent_mcp_tool_specs()` register a remote
  Sophia peer as an ordinary **gated MCP tool** (`sophia.delegate`) via
  `gateway.federation.register_mcp_server` — the lowest-friction way to get agents delegating
  today, reusing the path the repo already has.

Task lifecycle ↔ RunStore: each `message/send` creates an `A2ATask`; the swarm's
`subagent.delegate` already writes a parent + per-child JSONL trace under `RUNS_DIR`, so the
A2A task id maps onto an auditable run tree. v1 executes inline but is **async-shaped** (tasks
are pollable via `tasks/get`); see §5 to make it genuinely async.

## 4. The three ways to connect (mapped to the chosen requirements)

Requirements: **multi-machine (all yours)**, **Sophia + external LLM agents**, **long-running/async**.

1. **MCP-as-agent (ship first, lowest friction).** Expose a peer as the `sophia.delegate` MCP
   tool; call it through the federation transport you already have. Great for same-org, quick
   delegation; not full A2A (no Agent Card discovery / task lifecycle). → `A2AMcpTransport`.
2. **A2A server (the real interop layer).** Stand up `A2AServer` behind an HTTP endpoint on each
   box (Spark / Mac / RunPod / web). Publish the Agent Card at
   `/.well-known/agent-card.json`. Now any A2A-speaking agent — including third-party Claude /
   GPT / Hermes-style agents — can discover Sophia and delegate. → this module + a thin HTTP shim.
3. **Async message bus (for scale + long-running).** Put a queue/stream (NATS, Redis Streams,
   or an offline-testable SQLite/file outbox) between `message/send` and execution: the server
   enqueues, a worker runs the swarm (possibly a GPU job), the task flips to `completed` when
   done and the caller polls `tasks/get` or receives an A2A push notification. Pairs with
   `agent/long_horizon.py`'s durable task tree.

## 5. Multi-machine + async specifics

- **Topology.** One Agent Card per box; a small **registry** (a JSON of peer URLs, or ANP/AGNTCY
  discovery later) lets the swarm-router pick a *remote* team the same way it picks a local one.
  `SwarmPlan` already carries per-team budgets and least-privilege scopes — a remote team is just
  an assignment whose execution is an `A2AClient.delegate_to_peer` instead of a local subagent.
- **Long-running.** Keep `require_auth=True` + per-peer API keys; return `working` immediately and
  let callers poll. For GPU work, the worker is a GitHub-Actions/RunPod job (per `AGENTS.md`:
  RunPod via Actions, never local SSH) that writes the artifact back to the task store.
- **External LLM agents.** Because the wire is standard A2A + MCP, a Claude/GPT/Hermes agent can
  be *either* a peer Sophia calls (`A2AClient` to their endpoint) *or* a caller that delegates to
  Sophia. Either way Sophia's gate sits at the boundary.

## 6. Auto-spawn / auto-trigger (the "automatically" part)

`swarm_router` is already the *policy brain*. Give it triggers:

- **Events** — webhooks / PR events / file watchers → enqueue a task → router decides → spawn
  (local subagent or remote A2A peer). Same shape as the `.claude/` hooks, one altitude up.
- **A2A push notifications / streaming** — a parent submits a task and reacts to state changes.
- **`long_horizon.py`** — a plan node needing a sub-result spawns a child (local or A2A) and
  resumes on completion: a durable auto-spawn engine.
- **Scheduled** — cron / `/loop` for recurring sweeps (e.g. route open failure-ledger items to a
  verifier peer hourly).

## 7. Security (do not skip)

Connecting agents over a wire opens **cross-agent prompt injection / capability escalation**
(an active research area: 2602.11327 threat-models MCP/A2A/Agora/ANP). Sophia's posture maps
directly and is a strength:

- **Untrusted peers.** A remote agent's claims are external data — run through the gate before
  acting (`delegate_to_peer`); never upgrade a peer's abstain/block to trust.
- **Least-privilege Agent Card.** Advertise only scoped skills; an unadvertised capability can't
  be invoked. Inbound tasks lower to least-privilege `SubagentSpec`s (never widen tool scope).
- **Auth.** `require_auth` + per-peer API keys for v1; OAuth2 / mTLS for production multi-org.
- **Fail-closed everywhere.** Empty/garbage task → no swarm; our own answer is gated before we
  hand it out; errors return a JSON-RPC error, never a stack trace.

## 8. What is built now vs next

**Shipped (this change, offline + CI-testable):**
- `agent/a2a.py` — Agent Card, task lifecycle, server, client + stub/http transports, fail-closed
  gate hook, MCP-as-agent bridge.
- `schema/agent-card-1.0.0.json` — the discovery contract.
- `tools/a2a_demo.py` — runnable offline round trip; `tests/test_a2a.py` — 16 deterministic tests.

**Next (in order):**
1. Thin HTTP shim around `A2AServer` (one endpoint + `/.well-known/agent-card.json`) and deploy
   on Spark/Mac/web; smoke-test two real boxes with `HttpA2ATransport`.
2. Wire the **real epistemic gate** (`sophia_gate_check` / `agent.grounded_*`) into the `gate=`
   hook, replacing `default_gate`.
3. Peer **registry** + a `remote` team in `swarm_router` so the router can fan out across boxes.
4. **Bus + worker** for long-running/GPU tasks (Actions/RunPod), with A2A push notifications.
5. Track ANP / AGNTCY for cross-org discovery.

## 9. References

- A2A: <https://github.com/a2aproject/A2A> · Linux Foundation A2A project · IBM "What is A2A".
- Interop survey (MCP/ACP/A2A/ANP): arXiv 2505.02279. Security threat model: arXiv 2602.11327.
- MCP server wiring here: `docs/09-Agent/MCP-Server.md`, `gateway/federation.py`.
- Swarm design: `docs/11-Platform/Agentic-MoE-Swarm.md`, `agent/swarm_router.py`, `agent/subagent.py`.
