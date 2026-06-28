# Swarm Variants V3 & V4 — full specs

> Companion to [`Agentic-MoE-Swarm.md`](Agentic-MoE-Swarm.md). Expands the two most novel
> variants into buildable specs. **Status:** design (PROPOSED). The v1 deterministic
> router (`agent/swarm_router.py`), schema (`schema/swarm-plan-1.0.0.json`), benchmark
> (`provenance_bench/swarm_benchmark.py`) and RLVR reward (`provenance_bench/swarm_rl.py`)
> are the foothold these build on.

---

## V3 — Branch-Train-MiX-Swarm: one adapter is both an FFN-expert *and* an agent seat

**Thesis source:** Sukhbaatar et al. 2024, *Branch-Train-MiX (BTX)* (train expert LLMs in
parallel, then fold their FFNs into a single MoE); Li et al. 2022, *Branch-Train-Merge*;
DeepSeek-AI 2024, *DeepSeekMoE* (shared + routed experts). The novelty here is the
**dual-use artifact**: each specialist is trained once and used at *two altitudes*.

### The core idea

Today the repo has two disjoint notions of "specialist":
- a **council seat** (`agent/sector_council.py`) — a persona realised by a *prompt*;
- an **MoE expert** (`moe/router.py`) — a sub-network realised by *weights*.

V3 collapses them. For each domain `d ∈ {search, research, math, legal, ontology, …}` you
train **one LoRA adapter** `θ_d`. That adapter is used two ways:

| Altitude | How `θ_d` is used | Cost | When the router picks it |
|---|---|---|---|
| **In-weights (fast)** | mixed into the backbone as an FFN-MoE routed expert (BTX merge) | one forward pass | low-difficulty, single-aspect tasks |
| **As-agent (deep)** | loaded into a child harness as a full sub-agent persona/seat | a whole `run_subagent` loop | high-difficulty tasks the Swarm-Router fans out |

So the *same* `θ_d` that the token-router activates for a cheap query is the *same*
specialist the Swarm-Router spawns as an agent for a hard query. One training run, two
deployment modes — and `moe/adapt.py`'s per-adapter bit-allocation already governs the
in-weights side.

### Why it's elegant for Sophia specifically

- **The council becomes trainable.** Seats are currently hand-authored prompts that don't
  improve. As LoRA adapters they get the full `agent/continual_plasticity.py` promotion
  gate — a seat that regresses the protected suite is rejected.
- **Two routers, one expert set.** The token-router (`moe/router.py`) and the
  Swarm-Router (`agent/swarm_router.py`) both index into the same `{θ_d}`. A natural
  curriculum: the Swarm-Router learns *when one forward pass through θ_d is enough* vs
  *when θ_d needs a full agent loop with tools*.
- **Distillation closes the loop** (`Council-Distillation.md`): distill the as-agent
  traces back into the in-weights expert, so next time the cheap path suffices more often.

### Build plan

1. **Branch.** For each domain, LoRA-tune a copy of the backbone on that domain's corpus
   (the repo already has `training/council/traces.jsonl`, sector councils, math-code
   curriculum). Promote each `θ_d` only through `continual_plasticity`.
2. **Mix.** Fold the `θ_d` FFNs into a routed-expert MoE (BTX). Add a learned token-router
   on top (standard `moe/router.py` machinery); the **shared expert** is the un-adapted
   backbone (DeepSeekMoE split) so every token gets baseline reasoning for free.
3. **Dual-bind.** Register each `θ_d` in *both* the MoE expert table and
   `agent/swarm_router.TEAMS` (a `Team` gains an `adapter_id`). `Team.spec()` loads
   `adapter_id` into the child harness.
4. **Co-train the two routers.** Token-router with the Switch aux loss; Swarm-Router with
   `provenance_bench/swarm_rl.py` (which already lifts the *same* aux to team-altitude).
5. **Distill** as-agent wins back into the in-weights experts.

### Honest failure modes

- **Adapter interference** when many `θ_d` are mixed (known BTX issue) → cap active
  experts, keep the shared expert dominant, measure per-domain regression on the
  protected suite.
- **Double-counting independence** — an in-weights expert and its as-agent twin are *not*
  independent, so a swarm of `{θ_d-as-agent, θ_d-in-weights}` is one opinion, not two.
  `team_agents.py`'s effective-N guard must treat them as correlated.

### Minimal first step
Add `adapter_id: str | None` to `agent/swarm_router.Team` and a loader hook in
`Team.spec()`; train **one** real `θ_search` adapter; show it works both as an MoE expert
(offline forward) and as a spawned agent (via `subagent`). No full BTX merge yet.

---

## V4 — Hypernetwork Spawner: mint a bespoke sub-agent instead of picking from a roster

**Thesis source:** Ha et al. 2016, *HyperNetworks* (a network that generates another
network's parameters); Toolformer (Schick 2023, learned dispatch tokens); soft-prompt /
prefix-tuning (Li & Liang 2021) as the *thing being generated*. The novelty: nobody has
used a hypernetwork to **synthesise a sub-agent's configuration on demand**.

### The core idea

The v1 router picks teams from a fixed catalogue (`TEAMS`). That's a hard ceiling: a novel
task may need a specialist that isn't in the menu. V4 replaces the discrete choice with a
**generated agent**:

```
task t → hypernet H(t) → agent embedding z ∈ R^d
                         ├─ decode → system_prompt(z)        (what the child is told to be)
                         ├─ decode → tool_mask(z)            (a least-privilege subset of the tool registry)
                         └─ decode → budget(z)               (steps / cost ceiling)
```

`z` is a continuous "agent latent." Instead of *selecting* expert `e`, the router *emits*
a point in agent-space and the child is instantiated from it. The fixed catalogue becomes
the **anchor set**: `z` is decoded as a soft mixture/perturbation of catalogue anchors, so
a generated agent is always *interpretable* as "mostly a search agent, tilted toward
legal, with the ontology tool added."

### Why it's powerful — and the safety knife-edge

- **Open-ended specialisation** without retraining the catalogue. The long tail of tasks
  gets a fitted agent.
- **But generation must stay fail-closed.** A hypernet that can mint *any* tool-mask is a
  privilege-escalation surface. So the **tool_mask decode is projected onto the allowed
  registry and intersected with the parent's scope** — the generated child can only ever
  receive a *subset* of tools the parent already holds (least privilege is a hard
  constraint on the decode, not a suggestion). The system_prompt passes the same
  `conscience_check` / `public_sanitize` every other emission does.

### How you train it honestly

`z` is trained by the **same RLVR reward** (`provenance_bench/swarm_rl.py`) — the
hypernet's gradient is "did the agent you minted produce verified success at low cost?"
Plus two regularisers:
- **anchor KL** — keep `z` decodable as a mixture of catalogue anchors (interpretability);
- **scope penalty** — punish requesting tools that go unused (drives minimal privilege).

Bootstrap the decoders from the existing catalogue: each `TEAMS[e]` is a labelled
`(z_e → system_prompt_e, tool_mask_e)` pair, so the hypernet starts by *reproducing* the
roster and only then learns to interpolate/extrapolate.

### Honest failure modes
- **Off-distribution z** → an incoherent agent. Guard: OOD detector on `z` (reuse the
  `predictive_world_model.py` OOD-hold idea) → fall back to the nearest catalogue anchor
  (V1 behaviour). Generation is an *enhancement* over a safe discrete floor, never a
  replacement for it.
- **Reward-hacking the minted agent** (mint an agent that games the verifier) → the
  NO-HACK held-out verifier (`agent/self_evolving_agent.py`) is the backstop.

### Minimal first step
Skip the neural hypernet. Implement the *decode contract* first: a deterministic
`synthesize_agent(z)` where `z` is a hand-set mixture weight over `TEAMS` anchors, proving
the tool-mask projection (`mask ⊆ allowed_registry ∩ parent_scope`) is fail-closed and the
system_prompt passes `conscience_check`. That's a CI-testable seam; the learned `H(t)` is
a later, GPU-gated drop-in — exactly the discipline the v1 router followed.

---

*Both variants reuse the same contract (`SwarmPlan`) and the same unhackable reward
(`swarm_rl.py`). V3 makes the experts trainable and dual-use; V4 makes the roster
open-ended. They compose: a BTX expert set is the natural **anchor set** for the V4
hypernetwork.*
