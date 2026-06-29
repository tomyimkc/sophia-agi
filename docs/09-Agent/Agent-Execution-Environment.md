# Agent execution environment — the prerequisite for a team-agent-native model

> Status: **shipped contract + design.** The environment glue (`agent/swarm_env.py`) is shipped and
> CI-tested via an injected deterministic runner; the live model-driven runner and the GPU training
> steps are OPEN. Makes no capability/AGI claim — it is the world an RL loop steps, not a result.

## Why this is prerequisite #1

The goal is a model that *behaves* like Claude Code's team-agents mode — spawn sub-agents, route,
execute tools, aggregate — **in its own weights, runnable on any inference server**, not dependent
on a specific terminal. That is **Mixture-of-Agents (MoA) orchestration** baked into a policy, the
macro-level cousin of in-weights MoE (see `docs/06-Roadmap/` and `provenance_bench/swarm_rl.py`,
which already lifts the MoE load-balancing objective from tokens-over-experts to tasks-over-teams).

A model cannot *learn* to orchestrate without a world that **executes** the orchestration and a
verifier that **scores** it. RL needs an executable, verifiable environment. That world is what the
repo lacked: the pieces existed separately but nothing composed them into one steppable episode.

## What already existed (do not rebuild)

| Piece | Role |
|---|---|
| `agent/swarm_router.py` | the routing **policy** seam (`decide` — "the single seam the trained head overrides") |
| `agent/subagent.py` | real **delegation**: isolated run store, least-privilege tools, bounded budget, fail-closed reduce |
| `agent/swarm_trust_boundary.py` | the inter-agent **gate** (a child's output is sibling-readable only if it clears the gate) |
| `provenance_bench/swarm_rl.py` | the machine-verified **reward** (`SwarmOutcome`, `swarm_reward`, `trajectory_reward`) |

## What was missing — and is now shipped

`agent/swarm_env.py` composes them into one episode:

```
task --router.decide--> SwarmPlan --to_specs--> least-privilege subagents
     --child_runner--> child outputs --TRUST BOUNDARY--> only gate-clean enter shared state
     --fail-closed reduce over ADMITTED only--> synthesis --machine verify--> SwarmOutcome --> reward
```

The critical change from `subagent.delegate` alone: `delegate` reduces over children that are merely
harness-`ok` (succeeded + on budget). `swarm_env` makes the **trust boundary the inter-agent
contract** — a child that succeeds but hallucinates an attribution is **held**, never enters the
synthesis, and does not count toward verified success (`test_harness_ok_but_gate_failing_child...`).

Two seams a trainer overrides:
- **`router`** — the policy RL optimises (default `SwarmRouter`; the trained head overrides `decide`).
- **`child_runner`** — how subagents execute (default wraps `agent.subagent.delegate`, the real
  harness; tests inject a deterministic runner so the route→gate→reduce→reward **contract** is
  CI-checkable without a model).

Measured contract (deterministic fixtures): a clean-orchestration episode rewards **0.955**; an
all-poison episode rewards **−0.445** and abstains. That gap is the policy gradient a trainer follows.

## The five-step path to the trained model (this is step 0→1)

| Step | What | State |
|---|---|---|
| 1 | **Agent execution environment** (route → gate → reduce → reward) | **shipped** (`agent/swarm_env.py`, 7 invariants + 7 tests) |
| 2 | Multi-turn trajectory reward over episodes | shipped (`trajectory_reward`); wire `SwarmEpisode` as a turn — OPEN |
| 3 | Orchestration trace data — generate episodes, label with verifiers (the preference engine generalises) | OPEN |
| 4 | Train the policy — SFT the dispatch format at the `decide` seam, then GRPO on the reward | OPEN (GPU, RunPod Action) |
| 5 | Distill — fold orchestration into the weights so the model self-dispatches inline, no scaffold | OPEN |

## Honest limits

- CI exercises the **contract** via an injected deterministic `child_runner`; the **live** runner
  needs the harness + a model client and is exercised in real runs, not CI (avoids model/key/RUNS_DIR
  flakiness) — the standard way this repo gates live paths.
- The gate is a filter, not a truth oracle: a false claim with no detectable violation can still be
  admitted (`admittedPoisonResidual`, cf. the trust-boundary measurement). The environment bounds
  contamination to what the verifiers cover.
- This is the world, not a trained policy. Whether a model trained in it actually orchestrates well —
  and transfers off the scaffold — is the OPEN question steps 3–5 answer, pre-registered like the
  adapter-transfer risk in the failure ledger.

## On referencing prior art

Team-agents ≈ **Mixture-of-Agents** (Wang et al. 2024, arXiv 2406.04692) + an agent harness;
the broader orchestration technique is open prior art (AutoGen, LangGraph, CrewAI, MetaGPT). MoE is
Shazeer 2017 / Switch 2021 / DeepSeekMoE. All freely citeable. Sophia's own contribution is the
**verifier-gated** version (the trust boundary as the inter-agent contract + the machine-checked
reward) — that is the defensible, novel layer. Do not reverse-engineer a specific product's harness;
build from the open research, as this environment does.
