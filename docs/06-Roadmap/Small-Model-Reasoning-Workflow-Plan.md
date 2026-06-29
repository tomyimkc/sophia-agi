# A 3B that reasons, runs workflows, and orchestrates team-agents

> Status: **design + shipped offline machinery.** The environment, reward, data engines, and the
> test-time thinking controller are shipped and CI-tested. The five GPU training stages are OPEN.
> `canClaimAGI` stays false. The honest target is a **domain-specialist** 3B, not a general
> frontier model.

## The honest claim

A 3B does **not** beat frontier models in general. On a **narrow, verifiable domain**, inside a
verifier-gated harness, it can match or beat much larger general models — there is hard evidence:
s1 (1k distilled traces + budget forcing) beat o1-preview on math; DeepSeek-R1-Distill-Qwen-1.5B
beats GPT-4o on AIME; and "tool use replaces thinking" shows small models win by **offloading truth
to tools/verifiers** instead of reasoning internally — exactly Sophia's thesis.

## Workflows vs team-agents (and what "ultra mode" is)

| | Team-agents (subagent) | Workflows ("ultra mode") |
|---|---|---|
| Orchestration | **model-driven** (LLM decides when to spawn) | **code-driven** (a deterministic script: loops, fan-out, pipeline) |
| Control flow | emergent, non-deterministic | reproducible, auditable |

"Ultra mode / workflow" is a **harness feature — deterministic code — not a model capability.** So a
3B does not need to *be* the orchestration engine; it is the **worker + router + script-author** the
harness calls. The deterministic structure lives in *your* harness (`agent/swarm_env.py`,
`tools/sophia_autoresearch.py`); the local intelligence lives in the 3B. That division is *why* a
small model can do this on your domain.

## How "thinking power" is actually tuned (not mainly attention)

In impact order: (1) **SFT on long chain-of-thought** traces (s1: ~1k examples); (2) **RL with
verifiable rewards** (R1) — the model learns to produce longer *productive* reasoning; (3)
**test-time compute / token budget** with `<think></think>` and **budget forcing** (append "Wait");
(4) adaptive continue-thinking tokens. Attention's role is *indirect* — long context (RoPE/YaRN) is
needed to *hold* a long trace, not to make it good. Architectural depth (recurrent-depth, latent
reasoning, Mamba) adds compute-per-token but is optional. And more thinking is **not** monotonically
better (overthinking hurts) — which is why a **verifier** that bounds thinking is an asset.

## Shipped machinery (offline, CI-tested)

| Piece | Role | Tests |
|---|---|---|
| `agent/swarm_env.py` | the RL **environment** (route→gate→reduce→reward) + multi-turn trajectory | 13 inv + 9 |
| `provenance_bench/swarm_rl.py` | the machine-verified **reward** (single + trajectory) | 15 inv |
| `agent/swarm_trust_boundary.py` | the inter-agent **trust contract** | 7 inv + 5 |
| `agent/test_time_thinking.py` | **gate-bounded budget forcing** — think until the verifier accepts, force "Wait" while wrong + budget remains, min-thinking floor | 5 inv + 6 |
| `tools/gen_reasoning_distill.py` | **verifier-gated distillation** — only gate-clean teacher traces become `<think>` SFT rows | 5 |
| `tools/gen_verifier_dpo.py` | preference-pair data engine (verifier as labeller) | 7 |
| `tools/sophia_autoresearch.py` | gated self-improvement loop (reward-hacking firewall) | 9 |

The novel synthesis in `test_time_thinking`: replace s1's arbitrary length cap with a **verifier as
the stop criterion** — test-time compute spent in proportion to *verified difficulty*, and the model
never sees the verdict (it only feels the "Wait" nudge), so it cannot game the gate.

## The five training stages (all GPU, OPEN; RunPod via the GitHub Action)

| Stage | What | Uses (shipped) |
|---|---|---|
| 1. Reasoning distill | SFT a 3B base (Qwen2.5-3B / Llama-3.2-3B) on gate-clean `<think>` traces | `tools/gen_reasoning_distill.py` |
| 2. Tool-offload SFT | teach it to **call the verifiers** (swarm_env tool format) instead of reasoning internally | `agent/swarm_env.py` + the ~30 verifiers |
| 3. RLVR (GRPO) | reward = the trajectory reward, **in `swarm_env`** | `run_swarm_trajectory` + `trajectory_reward` |
| 4. Test-time scaling | wrap inference in gate-bounded budget forcing | `agent/test_time_thinking.py` |
| 5. Workflow authoring | SFT/RLVR on emitting orchestration scripts + acting as nodes | `swarm_env` + `sophia_autoresearch` as targets |

Then distill + serve on vLLM/SGLang — runs anywhere, no specific terminal.

## Honest limits / pre-registered risks

- Domain-specialist, not general. Claim a win only on a **third-party verifiable pack**, ≥2 judge
  families, CI excluding zero — and remember the ledger's adapter-non-transfer risk
  (`v4-adapter-externally-unvalidated`).
- The thinking controller and environment are **contracts** tested with stub policies; the real 3B
  and the GPU stages are OPEN.
- The gate is a filter, not a truth oracle; it bounds error to what the verifiers cover.

## Sources

s1 (Muennighoff et al. 2025, arXiv 2501.19393) · DeepSeek-R1 · "Replacing thinking with tool usage…"
(arXiv 2507.05065) · limits of test-time scaling (arXiv 2507.14419) · Mixture-of-Agents
(arXiv 2406.04692).
