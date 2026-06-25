# Math RLVR — live 3-seed held-out run, N=60 (2026-06-25)

Live GRPO on `zai-org/glm-4-9b-chat-hf`, reward = `agent.verifiers.math_equivalent`
(sympy; deterministic, **non-gameable**, **no LLM judge**), run on RunPod A100 80GB
via **vLLM colocate** (trl 0.19.1 + vllm 0.9.1, launched with `accelerate launch`).
Pack: 164 train / **60 fixed held-out**; the 3 held-out families
(`derivative_chain`, `integrate_func`, `second_derivative`) are **disjoint** from the
6 train families and **identical across seeds**, so a passing eval item is an unseen
problem *type* (generalization), not a memorized instance.

## Result

| Seed | N | Base pass@1 | Adapter pass@1 | Δ | Passed |
|---|---|---|---|---|---|
| 0 | 60 | 0.000 (0/60) | 0.1167 (7/60) | **+0.1167** | ✓ |
| 1 | 60 | 0.000 (0/60) | 0.1000 (6/60) | **+0.1000** | ✓ |
| 2 | 60 | 0.000 (0/60) | 0.0833 (5/60) | **+0.0833** | ✓ |

- **Mean Δ = +0.100**, all three seeds Δ > 0, base = **0/60 in every seed**.
- **95% across-seed CI = [0.0585, 0.1415] — excludes 0.**
- Pooled: adapter **18/180** vs base **0/180**.
- Every run: `contaminationFree`, `noPassAt1Regression`, `adapterImprovesPassAt1`.

## Verdict — the rung gate is CLEARED

This is a **reproducible, statistically-supported, judge-free generalization gain**:
the base model floors at 0/60 on these unseen harder families; after GRPO on the
simpler related families the adapter solves 5–7/60, consistently across 3 independent
seeds, with the 95% CI excluding 0. It meets the self-extension rung's no-overclaim
bar (≥3 seeds, all Δ>0, CI excludes 0, contamination-checked, non-gameable verifier).

## Honest scope — what this is NOT

A **modest, narrow** capability: ~10% absolute on a small held-out math set where the
base model scores 0%. It clears *this rung's* gate; it is **not** an AGI claim, and the
package-level `canClaimAGI` stays **False**. The value is that the self-extension loop
now produces a *measured, honest, reproducible* gain on a judge-free domain — the rung
the failure ledger had flagged as open. The 177 MB adapter checkpoints stayed on the
(auto-deleted) pods; only these JSON reports are retained.

## Reproduce

```
# one seed (RunPod, vLLM colocate fast path):
gh workflow run rlvr-runpod.yml -f confirm=RUN -f remote_mode=live -f task=math \
  -f epochs=3.0 -f seed=0 -f interruptible=false
# vary --seed across runs; the held-out split is fixed, so seeds are comparable.
```
