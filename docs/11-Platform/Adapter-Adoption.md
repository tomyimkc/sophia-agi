# Adopting θ_search-style dual-use adapters — the feasible path

> **Status:** implemented. This is the production answer to "we validated θ_search — how do
> we actually *use* it?", derived from what the cross-model study proved.

## What the evidence forced

| Base | Corpus | Source-discipline Δ (3 judges) | Gate |
|---|---|---|---|
| **Qwen2.5-7B** | council traces | **+0.20** · CIs [+0.12,+0.29] · κ→1.0 | **PROMOTE** |
| **Mistral-7B** | council traces | **−0.28** · CIs [−0.40,−0.14] · κ→1.0 | **REJECT** |
| **Mistral-7B** | **format-robust** | **+0.12 / +0.23 / +0.12** (lex/stance/LLM) · CIs exclude 0 | **PROMOTE** |

The lift is **recipe-specific**, and the recipe matters more than the base: the council
corpus teaches a verbose multi-seat *format* that Qwen resists but Mistral overfits 30/30
(losing the direct refutation → −0.28). Swapping to a **format-robust** corpus (plain
refutations, no scaffolding) drops the council-format to **0/30** and **recovers** the
transfer on Mistral (+0.12 to +0.23, all CIs exclude zero, confirmed by an independent
DeepSeek judge at κ=1.0). The same multi-judge apparatus confirmed the positive, the
negative, AND the fix — so the measurement is trustworthy in all three.

**Recommended recipe going forward:** format-robust distillation
(`tools/build_discipline_sft.py`), which transfers across model families; council-trace SFT
happened to work on Qwen but is format-overfit-prone off-Qwen.

**Consequence for adoption:** you cannot "train once, bind everywhere." The safe pattern is
a **registry keyed by `(base_model, team)`** where a binding is admitted only if its
evidence clears the no-overclaim bar, and the router falls back to the plain backbone
otherwise.

## The mechanism (implemented)

```
candidate adapter
   │  run the acceptance test:  multiseed_remote.sh  (3 seeds, third-party pack)
   │                            + tools/llm_judge_score.py  (independent LLM-judge family)
   ▼
AcceptanceEvidence  ──►  AcceptanceEvidence.decide()   # >=2 judge families, every family
   │                                                   # positive with 95% CI excluding zero,
   │                                                   # inter-family kappa >= 0.40
   ▼
AdapterBinding(base_model, team, adapter_id, accepted)  ──►  data/adapters/registry.json
   ▼
SwarmRouter / SwarmPlan.to_specs(base_model=…)  ──►  registry.resolve(base_model, team)
                                                     accepted? bind adapter : plain backbone (fail-closed)
```

- **Acceptance gate** — `agent/adapter_registry.py`: `AcceptanceEvidence.decide()` is the
  bar (≥2 independent judge families, every family's mean Δ > 0 with a 95% CI excluding
  zero, κ ≥ 0.40). `decide_binding()` builds the decision straight from a run-result JSON +
  an LLM-judge report.
- **Registry** — `data/adapters/registry.json` (committed, sha-auditable). Today it records
  the **accepted** Qwen θ_search binding and the **rejected** Mistral-council binding — the
  rejection is kept on purpose (a public negative, like the failure ledger).
- **Router wiring** — `SwarmPlan.to_specs(base_model=…)` / `run_swarm(base_model=…)` resolve
  each team's adapter from the registry for the *active* base model. Fail-closed: no
  accepted binding → the team spawns on the un-adapted backbone. No regressing adapter is
  ever attached.

## How to adopt a new adapter (runbook)

1. **Train** a candidate for `(base_model, team)` — `SOPHIA_MODEL`, `SOPHIA_SFT`,
   `SOPHIA_EPOCHS` on `training/swarm_router/multiseed_remote.sh` (3 seeds).
2. **Acceptance-test** it: the run scores 2 heuristic families + saves raw generations;
   then `tools/llm_judge_score.py` adds an independent LLM-judge family offline (free,
   no GPU — the payoff of capturing generations).
3. **Decide**: `adapter_registry.decide_binding(run_result, …, llm_judge_report=…)`. If
   accepted, `registry.add(...)` + `registry.save()`. If not, record the rejected binding
   (honest ledger) — the router will keep using the backbone.
4. **Ship**: nothing else changes. `run_swarm(task, base_model=…)` now binds the adapter
   for that base only.

## The cross-model fix (validated)

The negative was diagnosed as *format* overfitting, and the fix — a **format-robust
corpus** (plain `claim → direct disciplined answer` pairs, distilled with
`tools/build_discipline_sft.py`, no council scaffolding, decontaminated from the eval) —
**worked**: on Mistral it dropped council-format outputs from 30/30 to 0/30 and flipped the
result from −0.28 to **+0.12/+0.23/+0.12** across three judge families, all CIs excluding
zero. So `data/adapters/registry.json` now carries an **accepted** `(Mistral, search)`
binding (`theta-search-mistral-robust`) alongside the Qwen one.

Crucially, the adoption mechanism did not depend on this success — had the fix failed, the
registry would simply keep Mistral fail-closed. That independence is what makes it the
feasible path: **correctness never depends on every base transferring.**

*See also: `agent/adapter_registry.py`, `agent/swarm_router.py` (`to_specs(base_model=…)`),
`Swarm-Variants-V3-V4-Spec.md` (V3 dual-use), `Governed-Scaling.md`.*
