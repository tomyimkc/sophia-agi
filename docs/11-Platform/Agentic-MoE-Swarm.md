# Agentic MoE — a Swarm-Router that spawns verifier-gated agent teams

> **Status:** design / brainstorm (no code yet). This is a *roadmap artifact*, not a
> capability claim. Everything below is labelled by what it would take to earn, in the
> repo's no-overclaim idiom. Where it names a mechanism as "implemented elsewhere" it
> links the real module; where it proposes something new it says **PROPOSED**.

## 0. One-line thesis

> **Lift Mixture-of-Experts from `token → FFN-expert` (inside the weights) up to
> `task → agent-team` (outside the weights), and keep Sophia's verifier-gate on the
> dispatch — so the model learns *when* to fan out a swarm, *which* teams to spawn,
> and may only synthesise from sub-agents that returned machine-checkable evidence.**

The MoE router and the agent swarm are the *same idea at two altitudes*. Nobody has
unified them under a **fail-closed, provenance-carrying** gate. That gap is exactly
Sophia's lane ("innovate at the trust layer", `VISION.md`).

---

## 1. Where this sits — the three layers the repo already has

The repo already contains **three** of the four layers this design needs. The missing
one is the *learned* glue.

| Altitude | What it is | Lives in | Learned? |
|---|---|---|---|
| **Intra-network MoE** | token routed to `k` of `E` FFN experts; capacity + Switch load-balancing loss | [`moe/router.py`](../../moe/router.py) | ✅ gating is learned |
| **Council (deliberation)** | *one* model wearing many personas in one shared context; map → gate → reduce | [`agent/council_deliberate.py`](../../agent/council_deliberate.py), [`agent/sector_council.py`](../../agent/sector_council.py), [`agent/team_agents.py`](../../agent/team_agents.py) | ❌ routing is hand-coded (trigger-term match) |
| **Delegation (swarm)** | a parent spawns *real* child agents, each with isolated context, least-privilege tools, own budget, own JSONL trace; fail-closed reduce | [`agent/subagent.py`](../../agent/subagent.py), [`agent/long_horizon.py`](../../agent/long_horizon.py) | ❌ decomposition is hand-coded by the caller |
| **Swarm-Router (the gap)** | **PROPOSED** — a *learned* policy that, from the task alone, decides solo-vs-fan-out, picks teams, sizes `k`, sets budgets | — | this is what we train |

So "MoE mode that spawns a swarm of team agents" = **put a trained router on top of the
delegation layer the repo already shipped**, and reuse the Switch load-balancing math and
the fail-closed reduce that are already proven offline in CI.

---

## 2. The core mechanism — "Agentic MoE"

Classic sparse MoE (Shazeer 2017; Fedus/Zoph/Shazeer 2021, Switch; Lepikhin 2020,
GShard):

```
token x → gate g(x) = softmax(W·x) → top-k experts → weighted combine → y
        + aux load-balance loss  aux = E · Σ_e f_e · P_e   (keeps experts evenly used)
```

**Agentic MoE** keeps the *form* and swaps the *experts*:

```
task t → SwarmRouter ρ(t) → top-k AGENT TEAMS → fan-out (parallel) → fail-closed reduce → answer+sources
       + load-balance loss      (don't collapse onto one team)
       + trust-balance loss     (don't over-rely on the cheap-but-wrong team)
       + cost/latency penalty   (don't spawn a 12-agent swarm to answer "2+2")
```

The "experts" are now **spawnable teams** (research, web-search, math-verify, legal,
ontology, red-team, …), each realised by the existing `subagent.run_subagent` with its
own tool-scope and budget. The router's output is a **structured swarm plan**, not a
soft mixture — but it is trained with MoE's gradient machinery (routing logits + a
balancing auxiliary), which is the novel transplant.

### What makes it *Sophia's* and not just MetaGPT/AutoGen

- **Provenance flows through dispatch.** Every sub-agent returns `claims-with-sources`;
  the reduce step is the *same* fail-closed gate as `subagent.py` (`ABSTAIN_NO_CHILDREN`
  — zero successful, evidence-bearing children → **abstain**, never synthesise from
  failures).
- **Two balancing losses, not one.** Switch's load-balance keeps experts *used*; we add
  a **trust-balance** term so the router can't learn to always pick the cheap team that
  *looks* confident but fails the verifier (the over-reliance failure in
  `Governed-Scaling.md`).
- **Verifiable reward.** The repo already owns verifiers — `agent/lean_verifier.py`,
  `agent/math_verifier.py`, `agent/gate.py`, `agent/legal_faithfulness.py`. They give the
  swarm a **machine-checkable RLVR signal** most multi-agent papers don't have.

---

## 3. Seven architectural variants (ranked, with the creative ones flagged)

Pick one to start; they compose.

### V1 — Dispatch-token MoE *(recommended MVP)*
Add reserved vocabulary tokens — `<spawn:research k=4 budget=lo>`,
`<spawn:search k=2>`, `<reduce>`. The router *is* the LM head's distribution over those
tokens. Train it Toolformer-style (Schick 2023): self-supervise by inserting a `<spawn>`
only where it raises downstream **verified** reward, keep the ones that help, drop the
rest. **Why first:** smallest change, reuses the whole subagent stack, decoder-native, no
new architecture. The model literally *speaks* its swarm plan.

### V2 — Two-level MoE (shared + routed)
Borrow DeepSeekMoE's **shared-expert + routed-expert** split (DeepSeek-AI 2024). A
*shared* always-on backbone does cheap reasoning every token; a *macro-router* fans out
*routed* agent-teams only for the hard sub-problems. Cheap FLOPs inside, expensive
cognition only when the router pays for it.

### V3 — Branch-Train-MiX-Swarm *(unifies council + MoE)*
**The elegant one.** Train `N` specialist LoRA adapters (Branch-Train-MiX, Sukhbaatar
2024) — but each adapter is *both* an FFN-MoE expert **and** an agent persona/seat. One
artifact, two uses: as an in-weights expert it's a fast forward pass; loaded into a child
harness it's a full agent. This collapses the council and the MoE into one object and
fits [`moe/adapt.py`](../../moe/adapt.py)'s per-adapter bit-allocation already in the repo.

### V4 — Hypernetwork spawner *(ultra-creative)*
The router emits not a *choice from a fixed roster* but a **learned agent embedding** — a
vector decoded into the child's system prompt + tool-scope mask. Agents are *synthesised
on demand* for a novel task instead of picked from a menu. Closest prior art is hypernets
+ soft-prompt generation; nobody has used it to *mint a sub-agent*. High-risk/high-novelty.

### V5 — Speculative / async swarm
Skeleton-of-Thought (Ning 2023): emit a plan skeleton, fan out the independent branches
*in parallel*, reduce. Pure latency win; the swarm's wall-clock ≈ slowest single agent,
not the sum. Maps cleanly onto the repo's parallel orchestration.

### V6 — Auction / market router *(ties to active inference)*
Teams *bid* with calibrated confidence; the router allocates a compute budget to maximise
**expected verified information gain** per token, not raw accuracy. Connects directly to
[`agent/active_inference.py`](../../agent/active_inference.py) (which already turns gaps
into prioritised verification plans) and `agent/selective_risk.py`. Economic MoE.

### V7 — Fractal / recursive swarm
Each spawned agent may itself enter swarm-mode — bounded recursion over
[`agent/long_horizon.py`](../../agent/long_horizon.py)'s durable task tree (which already
does dependency ordering + recovery memory). A depth cap + per-node budget keeps it from
exploding. Society-of-Mind made literal (Minsky 1986).

---

## 4. Thesis-based spine (the sources this rests on)

**Sparse MoE / routing (the in-weights half):**
- Shazeer et al. 2017, *Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer* — the original learned gate.
- Lepikhin et al. 2020, *GShard* — capacity factor + auto-sharding.
- Fedus, Zoph, Shazeer 2021, *Switch Transformer* — top-1 routing + the load-balance aux loss the repo already reproduces (`moe/router.py`).
- DeepSeek-AI 2024, *DeepSeekMoE* — fine-grained + **shared/routed expert** split (V2/V3 above).

**Expertizing then mixing (how you *make* the experts):**
- Li et al. 2022, *Branch-Train-Merge* — embarrassingly-parallel expert LM training.
- Sukhbaatar et al. 2024 (Meta), *Branch-Train-MiX (BTX)* — train experts, then fold into one MoE. Direct basis for V3.

**Multi-agent / society (the out-of-weights half):**
- Minsky 1986, *The Society of Mind* — the conceptual ancestor.
- Du et al. 2023, *Improving Factuality and Reasoning via Multiagent Debate*.
- Wang et al. 2024, *Mixture-of-Agents (MoA)* — layered LLM agents beating a single model; the closest direct prior art to "agent-level MoE."
- Wu et al. 2023, *AutoGen*; Hong et al. 2023, *MetaGPT*; Qian et al. 2023, *ChatDev* — role-structured multi-agent systems.

**Teaching a model to act / call / decompose:**
- Schick et al. 2023, *Toolformer* — self-supervised token-level tool learning (basis for V1's dispatch tokens).
- Yao et al. 2022, *ReAct*; Yao et al. 2023, *Tree of Thoughts*; Zhou et al. 2024, *Self-Discover* — structured reasoning/planning.
- Shinn et al. 2023, *Reflexion*; Khattab et al. 2023, *DSPy* — optimise multi-stage agent pipelines.

**Reward you can trust (why Sophia can train this honestly):**
- Zelikman et al. 2022, *STaR*; Gulcehre et al. 2023, *ReST* — bootstrap from self-generated, filtered traces.
- Lightman et al. 2023, *Let's Verify Step by Step* — **process** reward models.
- DeepSeek-AI 2025, *DeepSeek-R1* — **RLVR / GRPO** on machine-verifiable rewards.

**Efficiency (so a swarm is affordable):**
- Ning et al. 2023, *Skeleton-of-Thought* — parallel decode (V5).
- The repo's own cheap-compute work — `moe/quant.py` (FP8/NVFP4), open PR #219 (AirLLM-style layer streaming). Cheap serving is what makes "spawn 8 agents" not insane.

---

## 5. How to train it (the frontier-model recipe)

Sophia's stated philosophy is *don't out-train frontier labs; orchestrate and innovate at
the trust layer* (`VISION.md`). So the recipe **starts from a strong open MoE base** and
adds the swarm behaviour + the governors.

**Stage 0 — Backbone.** Start from an open MoE (DeepSeek-V3-class, or Qwen/Llama-MoE).
Don't pretrain from scratch. Serve it cheap via the repo's quant lane (`moe/quant.py`,
NVFP4) + DGX-Spark/RunPod (`mcp__runpod__*` is wired in this repo) so a multi-agent run is
economical.

**Stage 1 — Make the experts (Branch-Train-MiX).** Continue-train / LoRA `N` specialists
on the corpora the repo already has: council traces ([`training/council/traces.jsonl`](../../training/council/traces.jsonl)),
sector councils, math/code curriculum ([`training/sophia-math-code-curriculum/`](../../training/sophia-math-code-curriculum)),
tool-use traces. Each adapter = one expert **and** one agent seat (V3). Promote each only
through [`agent/continual_plasticity.py`](../../agent/continual_plasticity.py).

**Stage 2 — Teach dispatch (SFT, Toolformer-style).** Build trajectories where the
*optimal* move is to fan out. Seed from [`training/team_agents/sft_traces.jsonl`](../../training/team_agents/sft_traces.jsonl)
and synthesise more: take a hard task, solve it solo vs with a spawned team, and keep the
`<spawn>` annotation **only when the team's verified reward beats solo**. Self-supervised,
no human labels for the routing decision.

**Stage 3 — RLVR on the router (the core).** This is where Sophia wins, because the reward
is *verifiable*:

```
R = verified_task_success            # lean_verifier / math_verifier / gate / legal_faithfulness
  − λ_cost · compute_spent           # tokens × agents × steps
  − λ_lb   · load_imbalance          # Switch aux loss, lifted to teams
  − λ_trust· over_reliance           # penalise picking a team whose output later fails the gate
  − λ_lat  · wall_clock              # discourages needless serial depth
```

Optimise the routing logits with GRPO/PPO (DeepSeek-R1 style). Crucially, **success is
machine-checked**, so there's no LLM-judge reward-hacking hole for the math/code/citation
slices.

**Stage 4 — Process reward on the reduce step.** Train a PRM (Lightman 2023) — or reuse
[`agent/faithfulness_probe.py`](../../agent/faithfulness_probe.py) — to score whether the
*synthesis faithfully used* the sub-agents' evidence, not whether it merely sounds good.
This is the anti-confabulation guard on the merge.

**Stage 5 — Distill the swarm back into the weights (close the loop).** The repo already
specs this: [`Council-Distillation.md`](Council-Distillation.md). Distill multi-agent
traces into a single forward pass so the model *internalises* "I can answer this alone" vs
"this needs a research team." Result: cheaper at inference, and the router gets *better*
at not spawning when it doesn't need to.

**Governors wrapped around every stage (already in the repo):** no-reward-hacking held-out
verifier ([`agent/self_evolving_agent.py`](../../agent/self_evolving_agent.py) NO-HACK
gate), promotion floor + retention gate (`continual_plasticity` + `continual_retention`),
SSIL capability-ceiling probes (`agent/ssil_*`). An improvement that regresses the
protected suite or hacks the checker is **rejected**, not shipped.

---

## 6. Honest failure modes (pre-register before claiming anything)

| Risk | Why it bites a swarm specifically | Guard already in repo |
|---|---|---|
| **Router collapse** | gate learns to always spawn the same team | Switch load-balance loss (`moe/router.py`) |
| **Swarm groupthink** | sub-agents from one base model make *correlated* errors → false consensus | `team_agents.py` already enforces effective-N; require `N_eff ≥ 2` independence |
| **Cost / latency blowup** | naive fan-out is N× the tokens | cost+latency penalties in R; V5 async; cheap-serving (PR #219) |
| **Verifier reward-hacking** | router games the checker, not the task | NO-HACK held-out verifier (`self_evolving_agent.py`) |
| **Confabulated reduce** | merge invents a consensus the children didn't support | fail-closed `ABSTAIN_NO_CHILDREN` + PRM on reduce (Stage 4) |
| **Over-reliance** | trusts the swarm even when it should abstain | trust-balance loss + `agent/selective_risk.py` |

Pre-registered bar (same as `RESULTS.md`): a swarm "win" counts only when verified task
success beats the **solo same-model baseline** on a held-out, decontaminated pack, ≥2
judge families (or a machine-verifiable slice), ≥3 seeds, 95% CIs excluding zero — *and*
the cost delta is reported, not hidden.

---

## 7. Minimal first step (one PR, offline-testable, fits the repo idiom)

1. `agent/swarm_router.py` — a `SwarmRouter` policy: `decide(task) → SwarmPlan` where
   `SwarmPlan ∈ {answer_solo | spawn(team, k, budget)…}`. Start as a **scored policy**
   over existing signals (query-understanding difficulty, source-count, contradiction
   risk) — deterministic, no network, CI-testable like every other `agent/*` module.
2. `schema/swarm_plan.v1.json` — the dispatch contract.
3. Wire `SwarmPlan` into `agent/subagent.py` (it already does the fan-out + fail-closed
   reduce — the router just *chooses* the `SubagentSpec`s instead of the caller).
4. `provenance_bench/swarm_benchmark.py` — solo-vs-swarm on a held-out pack, reporting
   verified success **and** cost, under the no-overclaim gate.
5. Only *then*, once the hand-scored policy shows a real delta, replace `decide()` with the
   trained Stage-2/3 head.

That sequence keeps the first deliverable **honest and mergeable** (a deterministic router
+ a measurement harness) and defers the GPU-trained head until the offline harness proves
the swarm is worth training for.

---

*See also: [`Governed-Scaling.md`](Governed-Scaling.md) (the trust-governor philosophy),
[`Council-For-Small-LLMs.md`](Council-For-Small-LLMs.md), [`Council-Distillation.md`](Council-Distillation.md),
[`AGI-Missing-Pillars.md`](AGI-Missing-Pillars.md), and [`Routed-Metacognition.md`](Routed-Metacognition.md).*
