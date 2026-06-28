# Adopting θ_search-style dual-use adapters — the feasible path

> **Status:** implemented. This is the production answer to "we validated θ_search — how do
> we actually *use* it?", derived from what the cross-model study proved.

## What the evidence forced

| Base | Corpus | Source-discipline Δ (3 judges) | Gate |
|---|---|---|---|
| **Qwen2.5-7B** | council traces | **+0.20** · CIs [+0.12,+0.29] · κ→1.0 | **PROMOTE** |
| **Mistral-7B** | council traces | **−0.28** · CIs [−0.40,−0.14] · κ→1.0 | **REJECT** |

The lift is **base-model- and recipe-specific** (root cause: the council corpus teaches a
verbose multi-seat *format* that Qwen resists but Mistral overfits 30/30, losing the direct
refutation). The same multi-judge apparatus confirmed **both** the positive and the
negative — so the measurement is trustworthy in both directions.

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

## The cross-model fix (in progress)

The negative was diagnosed as *format* overfitting, so the fix is a **format-robust
corpus** — plain `claim → direct disciplined answer` pairs, distilled with
`tools/build_discipline_sft.py` (no council scaffolding; decontaminated from the eval).
Re-running Mistral on that corpus tests whether the lift then transfers:
- **transfers** → a more general recipe; add the accepted Mistral binding.
- **still flat/negative** → transfer is genuinely base-specific; the registry already makes
  that safe (per-base bindings, backbone fallback).

Either outcome is *handled* by the mechanism above — which is exactly why this is the
feasible adoption path: correctness does not depend on every base transferring.

*See also: `agent/adapter_registry.py`, `agent/swarm_router.py` (`to_specs(base_model=…)`),
`Swarm-Variants-V3-V4-Spec.md` (V3 dual-use), `Governed-Scaling.md`.*
