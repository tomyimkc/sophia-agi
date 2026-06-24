# Math RLVR — live 3-seed held-out run (2026-06-24)

Live GRPO on `zai-org/glm-4-9b-chat-hf`, reward = `agent.verifiers.math_equivalent`
(sympy; deterministic, **no LLM judge**), run on RunPod on-demand A100 80GB pods
via `.github/workflows/rlvr-runpod.yml --task math`. Held-out split is
**family-disjoint** (a passing eval problem is an unseen problem *type*).

## Result

| Seed | Held-out families | N | Base pass@1 | Adapter pass@1 | Δ |
|---|---|---|---|---|---|
| 0 | factor, simplify | 8 | 0.25 | 0.25 | 0.0 |
| 1 | factor, integrate | 8 | 0.00 | 0.125 | +0.125 |
| 2 | factor, integrate | 8 | 0.00 | 0.125 | +0.125 |

Mean Δ ≈ **+0.083**. Every run contamination-free, no regression.

## Honest verdict

- **✅ The self-extension rung closes mechanically.** A live weight update on a
  judge-free, family-disjoint, contamination-checked held-out domain, across 3
  seeds, reproducible, pods auto-deleted. This is the "remaining rung" the failure
  ledger flagged — it now runs end-to-end.
- **❌ Not a validated capability claim.** The uplift is **within noise** at N=8
  (CI includes 0); it does not clear the no-overclaim gate
  (`aggregate._is_validated`). Cause: only 16 training problems, 1 epoch, base
  model near the tiny-N ceiling.

Superseded by a larger-N pack run (held-out N≥50, fixed held-out families across
seeds, more epochs) so the CI can actually exclude 0. The 177 MB adapter
checkpoints stayed on the (now-deleted) pods; only these JSON reports are retained.
