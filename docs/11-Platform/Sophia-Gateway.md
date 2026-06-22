# Sophia Gateway — the super-MCP / super-skills spec

> **Thesis.** Sophia should not be *a* tool server competing on capability. It should be
> the **trust layer the whole tool/skill ecosystem runs through** — the gateway that
> makes *any* MCP server or AI skill **safe, verifiable, provenance-tracked, and
> self-improving**. Other MCPs give an agent more power; Sophia makes that power
> trustworthy. That is a category, not a tool.
>
> **One-line positioning:** *"The governance gateway that makes any AI skill or MCP
> server safe, verifiable, and self-improving."*

Status: **design spec + P0 MVP shipped.** The `gateway/` package implements P0 (federate
in-process tools + the fail-closed intercept pipeline); `tools/run_gateway_demo.py` +
`tests/test_gateway.py` (12 acceptance checks) run it in CI. P1–P5 below remain the plan.

---

## 1. Why this, why now

The repo already contains every primitive the gateway needs — they just aren't composed
into one front door:

| Need | Existing asset (reuse, don't rebuild) |
|---|---|
| Fail-closed accept/reject/hold of any artifact | `sophia_contract` (`record_claim` / `verify_claim`, golden vectors) |
| Classification + no-read-up / no-write-down | `sophia_contract.blp`, `agent/security/labels.py` (BLP + Biba) |
| Per-role least privilege | `sophia_contract.roles.ROLES_9`, `scopes.py` |
| Budget caps, kill switch, durable queue, ROI | `sophia_contract` v1.1.0 |
| Audit / observability | decision log + Langfuse traces (`sophia_contract.trace`, `langfuse_export`) |
| Injection / untrusted-input defense | `agent/dataflow/*` firewall + `tools/run_security_redteam.py` + AgentDojo |
| Provenance / grounding of outputs | `okf` belief graph, grounding gate, `check_claim` |
| Verify by executing | `selfextend.env_verifier`, interpreter-as-verifier (code-uplift) |
| Measured reliability + calibration | `selfextend.competence_map`, `calibration_metrics` |
| Synthesize new verifiers + self-improve | `selfextend` flywheel + `close_loop` |
| Multi-agent adjudication | sector councils, `council_deliberate` |
| Tool surface to agents | `sophia_mcp` (32 tools today) |

The gateway is the **composition layer** over these, exposed as one MCP endpoint.

---

## 2. The full idea catalog (everything in scope, phased)

Ten ideas from the brainstorm, grouped and mapped to phases (§7):

**A. Super-MCP — governance gateway**
1. **Sophia Gateway** — an MCP proxy / MCP-of-MCPs that federates downstream servers and gates every call. *(P0–P1, flagship)*
2. **MCP firewall** — scan tool descriptions for injection, taint untrusted output, quarantine. *(P1)*
3. **Provenance-stamped tool use** — every tool result carries a provenance record + confidence; ungrounded output cannot be published. *(P2)*

**B. Super-skills — verifiable, self-improving skills**
4. **Verifiable Skill format** — a skill = `{program, verifier, blp_level, eval_suite}`; output ships only if its verifier accepts. *(P2)*
5. **Skill registry with measured reliability** — competence + calibration + ROI per skill; agents pick by measured trust. *(P3)*
6. **Self-improving skills** — each skill runs the self-extend flywheel; gets better with use. *(P4)*
7. **Skill synthesis on demand** — synthesize a new skill verifier-first when a task can't be done. *(P4)*

**C. Cross-cutting super-services**
8. **Universal Verify API** — `verify(anything) → {accepted|rejected|held}` routing to the right verifier with calibrated abstention. *(P2)*
9. **Knowledge MCP** — the OKF belief graph as a shared, self-healing, provenance-tracked KB over MCP. *(P3, mostly exists)*
10. **Verified-consensus MCP** — councils as a service, adjudicated by a verifier not a vote. *(P5)*

---

## 3. Architecture

```
                         ┌──────────────────────────────────────────────┐
   agent / LangGraph /   │                SOPHIA GATEWAY                  │
   Claude Agent SDK /    │                                              │
   n8n  ───MCP──────────▶│  describe()  list_tools()  call_tool()       │
                         │                                              │
                         │   ┌── intercept pipeline (per call) ──────┐  │
                         │   │ 1 authn + role scope   (scopes)        │  │
                         │   │ 2 firewall: injection scan + taint     │  │
                         │   │ 3 budget + kill switch check           │  │
                         │   │ 4 BLP no-read-up on inputs             │  │
                         │   │ 5 DISPATCH to downstream tool/skill    │──┼──▶ downstream MCP servers
                         │   │ 6 verify_claim on the OUTPUT           │  │   (filesystem, web, db, …)
                         │   │ 7 provenance-stamp + BLP no-write-down │  │   + local skills (env-verifier)
                         │   │ 8 audit log + Langfuse trace + ROI     │  │
                         │   │ 9 competence-map update (reliability)  │  │
                         │   └────────────────────────────────────────┘  │
                         │   registry: tools, skills, downstream servers │
                         │   (risk tier · BLP · scope · verifier · stats)│
                         └──────────────────────────────────────────────┘
```

**Default = deny.** A call returns a result only if every stage passes; otherwise a typed
error or a `held` verdict. Nothing leaves the gateway un-audited.

### 3.1 Components

- **Registry** — `gateway/registry.py`: the catalog of downstream MCP servers, native
  tools, and skills. Each entry carries: `id`, `kind` (`mcp|native|skill`), `risk_tier`,
  `blp_level`, `allowed_roles`, `verifier_ref` (how to verify its output), `side_effects`
  (read/write/external), `dry_run_supported`, and live `reliability`/`calibration`/`roi`.
- **Firewall** — `gateway/firewall.py` (wraps `agent/dataflow`): injection scan of tool
  *descriptions* at registration and of tool *outputs* at return; taints untrusted output
  (Biba low-integrity) so it can't flow into a higher-integrity sink.
- **Dispatcher** — `gateway/dispatch.py`: transport to a downstream MCP (stdio/HTTP),
  a native `sophia_mcp` tool, or a local skill. Honors `dry_run`, timeout, retries.
- **Interceptor** — `gateway/interceptor.py`: the 9-stage pipeline, orchestrating
  `sophia_contract` + firewall + dispatcher + competence map. The heart of the gateway.
- **Verify router** — `gateway/verify_router.py` (idea #8): routes an output to the
  right verifier — deterministic check, env-execution (`env_verifier`), belief-graph
  grounding (`check_claim`), or LLM-judge council — and returns a calibrated verdict.
- **Server** — `gateway/server.py`: the MCP endpoint exposing the gateway tools (§4).

### 3.2 The intercept pipeline (exact order, fail-closed)

1. **Role scope** — `scopes.check(role, "call_tool", blp_level=tool.blp)`. Unknown role / out-of-scope → `UNAUTHENTICATED`.
2. **Firewall (input)** — scan args + the tool description for injection markers; refuse on hit (`BAD_REQUEST` / quarantine), else tag input integrity.
3. **Budget + kill switch** — `_guard_kill_switch()`; budget cap → `held(over_budget)` / `OVER_BUDGET`.
4. **BLP no-read-up** — caller clearance must dominate `tool.blp_level`, else `held(blp_violation)`.
5. **Dispatch** — call the downstream tool/skill (honoring `dry_run`). Transport failure → `UNAVAILABLE` (retryable).
6. **Verify output** — `verify_router.verify(output, tool.verifier_ref)`. Only `accepted` may be returned to the agent; `rejected`/`held` returns the verdict + `suggested_fix`, never the raw unverified output.
7. **Provenance-stamp + no-write-down** — wrap the accepted output as a `Claim` (sources = tool id + inputs; `blp_level` = max(inputs, tool)); taint untrusted output (Biba) so downstream can't launder it.
8. **Audit + trace + ROI** — append to the decision log, emit a Langfuse span, attach `roi_estimate`.
9. **Competence update** — record the outcome against the tool's reliability; future routing/escalation uses it.

---

## 4. Wire contract (extends governance contract v1.1.0, additive → v1.2.0)

New MCP tools / methods (field names are contract; semver rules from `CONTRACT.md` apply):

- `gateway_describe() → {version, gateway_version, downstream[], capabilities[], schema_url, deprecations[]}`
- `register_server({id, transport, url|cmd, default_blp, allowed_roles, verifier_ref?}) → ServerEntry` *(admin scope)*
- `list_tools({role?}) → [{tool_id, server, risk_tier, blp_level, allowed_roles, reliability, dry_run_supported, description}]`
- `call_tool({tool_id, args, role?, clearance?, dry_run?, idempotency_key?}) → {result?, verdict, provenance_id, roi_estimate, error?}`
  - **Invariant:** `result` is present only when `verdict == "accepted"`. Otherwise `verdict ∈ {rejected, held}` with `held_reason`/`suggested_fix` and **no raw output**.
- `register_skill({skill_id, kind, program_ref, verifier_ref, blp_level, eval_suite_ref}) → SkillEntry`
- `verify({content, kind?, sources?, blp_level?}) → Verdict` *(idea #8, universal verify)*

Reuses unchanged: `record_claim`, `verify_claim`, `health`, `enqueue_task`, `next_task`,
`engage/release_kill_switch`, `record_human_verdict`. Error model + held_reasons unchanged.

### 4.1 Registry entry (data model)

```json
{
  "id": "fs.read_file",
  "kind": "mcp",                         // mcp | native | skill
  "server": "filesystem",
  "risk_tier": "low",                    // low | medium | high (drives auto vs escalate)
  "blp_level": "CONFIDENTIAL",
  "allowed_roles": ["role_02_coding", "role_09_agents"],
  "side_effects": "read",                // none | read | write | external
  "dry_run_supported": true,
  "verifier_ref": "grounding",           // deterministic | env:arithmetic | grounding | judge:council | none
  "reliability": 0.94, "calibration_ece": 0.06, "roi_minutes": 3.0,
  "injection_scan": {"clean": true, "scanned_at": "..."}
}
```

### 4.2 Verifier routing (idea #8)

`verifier_ref` selects the verification strategy for a tool's output:
- `deterministic` — schema / regex / rule check.
- `env:<kind>` — execute and check (`selfextend.env_verifier`: arithmetic, regex; extensible to code via interpreter-as-verifier).
- `grounding` — belief-graph / retrieval grounding (`check_claim`, grounding gate).
- `judge:council` — multi-judge council with κ-reported consensus (idea #10).
- `none` — pass through but still provenance-stamp + taint (lowest trust; flagged).

---

## 5. Security model (the differentiator — idea #2)

- **Default deny / fail-closed** at every stage; ambiguity → `held`, never silent pass.
- **Injection firewall:** tool *descriptions* are untrusted input (documented MCP attack
  surface). Scan at registration **and** re-scan outputs; a hit quarantines the tool
  (`risk_tier=high`, requires human promotion via `record_human_verdict`).
- **Taint / Biba no-write-up:** downstream output is low-integrity until verified; it
  cannot flow into a high-integrity sink (e.g., a `record_claim` to canonical memory)
  without passing the verify router.
- **BLP both directions:** no-read-up at dispatch, no-write-down at provenance-stamp.
- **Capability scopes:** every call carries a `role`; the 9-role registry is the default.
- **Budget + kill switch:** global stop-and-report; per-role budgets optional.
- **Full audit:** decision log + Langfuse trace per call; every result traceable to its
  tool, inputs, verifier, and verdict.

---

## 6. Super-skills layer (ideas #4–#7)

- **Verifiable Skill** = `{skill_id, program_ref, verifier_ref, blp_level, eval_suite_ref}`.
  A skill's output is run through its verifier before return — **skills that self-test**.
  Packaged as Claude Code skills / Agent SDK plugins; registered like any tool.
- **Reliability registry (#5):** each skill accrues `competence_map` reliability +
  `calibration` + `roi`; `list_tools` returns these so an agent (or a router) picks the
  most-trusted skill, not the best-marketed one.
- **Self-improving skills (#6):** a skill periodically runs `selfextend.close_loop` over
  its abstention ledger — synthesizes/validates new verifiers, improves by verified-reward
  selection (live RL when a GPU exists). Skills compound with use.
- **Skill synthesis (#7):** on an unservable task, synthesize a skill **verifier-first**
  (write the check, then the program, validate on held-out, register only if it clears the
  bar). The flywheel applied to skill *creation*.

---

## 7. Phased roadmap (each phase ships with falsifiable acceptance tests)

| Phase | Scope | Falsifiable acceptance (offline, deterministic) |
|---|---|---|
| **P0 — Gateway MVP** ✅ **DONE** | registry + interceptor + dispatcher + verify-router over in-process tools; stages 1,3,4,5,6,7,8,9 | ✅ low-risk grounded read → `accepted` + provenance_id; ungrounded → `held(no_source)`, **raw withheld**; env-verify pass/fail → accepted/rejected; SECRET tool → `held(blp_violation)`; out-of-scope role → `UNAUTHENTICATED`; kill switch → `UNAVAILABLE`; budget → `held(over_budget)`; dry-run write → `held(needs_human)` (not executed) — all in `tests/test_gateway.py` |
| **P1 — Firewall + risk tiers** | stage 2 + `risk_tier` auto/escalate | a tool description with an injection marker → quarantined (`high`, blocked); a clean call passes; high-risk tool → `held(needs_human)` |
| **P2 — Provenance-stamp + Universal Verify + Verifiable Skill** | stages 7/9 full; `verify()`; skill format | tool output re-entering context carries `provenance_id`; `verify()` routes arithmetic→env, claim→grounding; a skill whose verifier rejects does not ship |
| **P3 — Reliability registry + Knowledge MCP** | competence/calibration/ROI per tool; OKF KB exposed | `list_tools` ranks by measured reliability; a flaky tool's reliability drops and routing avoids it |
| **P4 — Self-improving + synthesized skills** | flywheel per skill; synth-on-demand | a skill's coverage rises run-over-run (false-accept ≤ bound); an unservable task yields a validated new skill or stays abstained |
| **P5 — Verified-consensus MCP** | councils as a service | a contested output is adjudicated by verifier with κ reported, not majority vote |

---

## 8. MVP (P0) — concrete build plan

New package `gateway/` (mirrors `sophia_contract/` conventions; dependency-free; offline):

```
gateway/
├─ __init__.py
├─ registry.py        # ServerEntry/ToolEntry + in-memory + JSONL store
├─ firewall.py        # (P1) injection scan + taint; P0 stub passes through
├─ dispatch.py        # call mock|native|mcp downstream (P0: native + mock)
├─ verify_router.py   # route output -> deterministic|env|grounding|none
├─ interceptor.py     # the 9-stage pipeline (P0: 1,3,4,6,7,8)
└─ server.py          # MCP tools: gateway_describe/list_tools/call_tool
tools/run_gateway_demo.py     # narrated offline demo (accept / held / denied / killed)
tests/test_gateway.py         # one falsifiable check per acceptance row + CI wiring
```

Reuse: `sophia_contract` (gate/scopes/blp/budget/killswitch/trace), `selfextend.competence_map`,
`selfextend.env_verifier`. No new third-party deps. Mock downstream = an in-process tool
table so the whole pipeline runs in CI with no network.

**P0 acceptance test (the gate that proves it):** a registered low-risk tool returns
`accepted` with a `provenance_id` and an audit entry; the same tool's ungrounded output
returns `held` with the **raw output withheld**; an out-of-scope role returns
`UNAUTHENTICATED`; the kill switch returns `UNAVAILABLE`. All deterministic.

---

## 9. Risks & open questions

- **Latency/overhead** of gating every call — mitigate with risk-tiered fast-path for
  `low` + `verifier_ref=deterministic`.
- **Verifier coverage** — many tools have no obvious output verifier; `none` is allowed
  but flagged (still provenance-stamped + tainted). The flywheel grows coverage over time.
- **Downstream MCP transport** — P0 is in-process/mock; real stdio/HTTP federation is P1+
  (needs the `mcp` client lib; keep optional like the server's `fastmcp` guard).
- **Wire stability** — gateway additions are a MINOR bump (v1.2.0); the existing golden
  vectors must keep passing (semver discipline already enforced in CI).

---

## 10. What this unlocks (positioning recap)

A single endpoint that turns the entire tool/skill ecosystem **safe (firewall + BLP +
fail-closed), verifiable (verify router + provenance), accountable (audit + ROI), and
self-improving (flywheel + synthesis)** — the "super-MCP / super-skills" category, built
entirely on assets Sophia already has. Commodity MCPs add capability; Sophia adds trust
and learning to all of them.
