# Verifier-Gated Trust Boundary (multi-agent)

> Status: **candidate control machinery** — deterministic, CI-tested invariants, not a
> capability or safety guarantee. The gate is a filter, not a proof of truth.

## The problem

The Swarm-Router (`agent/swarm_router.py`) decides *which* sub-agents to dispatch for a task.
But in any blackboard / AutoGen / LangGraph-style swarm there is a second, unguarded question:
**what is a sub-agent allowed to tell its siblings?**

In the default design, a sub-agent that hallucinates an attribution ("Confucius wrote the Dao
De Jing"), fabricates a citation, or asserts unsound arithmetic writes that straight into the
shared state. Every sibling then reads it as context and reasons on top of the error. This is
*exactly* the failure mode Sophia's single-agent provenance gate exists to stop — silently
re-introduced at the multi-agent layer, where it is harder to see and compounds across agents.

## The rule

Make **verification the inter-agent trust boundary**:

```
sub-agent output --> agent.gate.check_response --> accepted? --> readable by siblings
                                               \-> held      --> quarantined (audit only)
```

A sub-agent's output may enter the swarm's shared state — and so become readable context for
sibling agents — **only if it clears the machine gate**. Output carrying a hard verifier
violation is `held`: recorded for audit, never readable by a sibling, never folded into the
reduce step.

## Implementation

`agent/swarm_trust_boundary.py` — dependency-free, pure machine gate, no model, no network.

| Type | Role |
|---|---|
| `AgentMessage(agent_id, content, question, mode)` | a sub-agent's candidate contribution |
| `GatedEntry(... admitted, verdict, violations)` | the audited result of one submission |
| `GatedSharedState` | the verifier-gated blackboard |

`GatedSharedState`:
- `submit(msg)` — runs `agent.gate.check_response`; admits iff there is **no hard violation**
  (attribution / legal / numeric / routed). It keys on `violations`, **not** the gate's style
  `warnings` (a missing 中文 summary is not a contamination risk), so the boundary is unhackable.
- `readable()` — accepted entries only (the boundary).
- `held()` — quarantined entries, retained for audit, never sibling-readable.
- `context_for(agent_id)` — the shared context a given agent sees: accepted contributions from
  *other* agents only (no held output, no self-read).
- `audit()` — totals + per-entry verdicts, for the failure-ledger trail.

## Falsifiable invariants (CI)

`offline_invariants()` and `tests/test_swarm_trust_boundary.py` assert, deterministically:

1. a gate-clean contribution is admitted;
2. a gate-failing contribution is held;
3. held output carries the verifier reasons (audit provenance);
4. a sibling's readable context contains the clean claim and **not** the poison;
5. an agent never reads its own contribution back as "shared" context;
6. audit totals reconcile (`total == accepted + held`).

## How it composes

- **With the RLVR reward** (`provenance_bench/swarm_rl.py`): that reward already penalises
  `over_reliance` — dispatching a team whose output fails the gate. The trust boundary is the
  *runtime* enforcement of the same signal the *reward* trains toward: a failed-gate
  contribution earns negative reward **and** is barred from contaminating siblings.
- **With the single-agent gate** (`agent/gate.py`): same verifier farm, lifted from "one
  answer" to "one agent's message into a shared context." It is the multi-agent generalisation
  of the provenance gate.
- **With the Preference Engine** (`tools/gen_verifier_dpo.py`): the same accepted/held verdict
  that gates inter-agent reads can mint `(chosen, rejected)` preference pairs — the verifier is
  reused as runtime guard *and* as training-data labeller.

## Honest limits (pre-registered)

- The gate is a **filter, not a guarantee**. A *false* claim that asserts no forbidden
  attribution, no bad citation, and no unsound arithmetic can still be admitted. The boundary
  bounds inter-agent contamination **to what the verifiers cover** — it does not certify truth.
- Coverage is bounded by the verifier set; a claim type no verifier covers is admitted by
  default (the single-agent gate has the same coverage bound; see `RESULTS.md`).
- This is control machinery with falsifiable invariants. It makes **no** AGI or safety claim.
  Whether routing sibling reads through the boundary raises verified-success or lowers cost on
  a third-party agentic benchmark is an **OPEN** measurement (see the failure ledger).
