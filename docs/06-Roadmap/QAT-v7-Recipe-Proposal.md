# QAT NVFP4 v7 recipe — closing the last 0.009 (v6 top1 0.9609 → 0.97)

> Offline design doc (no GPU spent). `canClaimAGI` false. Cert bar unchanged:
> **mean_kl ≤ 0.05 AND top1 ≥ 0.97** (quantized-vs-FP next-token agreement, n=256).
> Companion to `QAT-v6-Recipe-Proposal.md`. Every v7 arm is pre-registered below and
> gated on an artifact + sha256 before any row flips.

## 1. Where v6 actually landed (the measured baseline)

| metric | v5 (train-08) | **v6** | bar |
|---|---|---|---|
| top1 agreement | 0.8828 | **0.9609** | ≥ 0.97 |
| mean_kl | 0.0506 ✗ | **0.0341** ✓ | ≤ 0.05 |
| verdict | FAIL | **NO-GO** (top1 short by 0.009) | |

v6 = fused-expert co-adaptation (all 32/32 experts reached — the fix that unblocked this)
+ output-space KD (`α=1.0`, `T=2.0`) + top1-margin (`β=0.5`, `margin=0`), `λ=0.0005`,
`epochs=5`, uniform LoRA `r=16`. Ledger: `nvfp4-v6-coadapt-cert-2026-07-01`; cert
`agi-proof/benchmark-results/certify-lowram-olmoe-nvfp4-v6.json`.

**Diagnosis (this is the whole point):** `mean_kl` already **passes** — the KD term solved the
distribution match. The residual is purely the **argmax**: top1 0.9609 ⇒ the quantized model flips
the next-token argmax on **~3.9%** of tokens (down from v5's ~12%). The top1-margin term works but
at `β=0.5 / margin=0` it is not strong enough on the low-margin tail. So v7 does **not** touch `λ`
(v4's over-regularization lesson) — it pushes the argmax objective harder and adds capacity where the
argmax is actually decided.

## 2. v7 levers — pre-registered, cheapest first

**Lever A — strengthen the top1-margin term (first; free of new machinery).**
- `β`: 0.5 → **0.75** → 1.0 (the v6 doc's own "sweep β if top1 still short").
- `margin`: 0 → **0.1** — a positive hinge forces the FP-argmax logit a margin above the runner-up,
  so the argmax **survives quant rounding** on the tail. v6 used `margin=0` (plain top1 CE); a margin
  is the natural next move for the ~4% flips.
- Head-room check: v6 `mean_kl` is 0.034 vs the 0.05 bar, so there is room to trade a little KL for
  top1 before the KL gate binds. Watch `mean_kl` doesn't cross 0.05.

**Lever B — expert LoRA-rank allocation (the v6 doc's untried lever).**
- v6 trained **uniform `r=16`**. Enable `--lora-rank-alloc`: **experts `r≈32`, attn `r≈16`** — same
  param budget, redistributed. MoE capacity lives in the experts; `r=16` may under-fit a quant-robust
  argmax. (train_lora already exposes `lora_rank_alloc`; the v6 fire did not use it.)

**Lever C — epochs 5 → 7.**
- v6 jumped 0.883→0.961 over 5 epochs; the remaining flips are the hardest rows. A couple more epochs
  is cheap and may specialize the experts on the tail. Diminishing returns — drop if it doesn't help.

**Lever D — depth-based mixed precision (if A–C still short; the v6 doc's key insight).**
- Hold the **last 1–2 transformer blocks bf16** at serve time. The argmax is decided by the final
  layers; the session already showed holding by *projection type* (`--keep-top-experts`) did not move
  top1 much — hold by **depth**. This needs a new `keep_layers` lever alongside the existing
  `keep_suffixes` / `keep_top_experts` (extend `tools/expert_protection.py` + the cert). Measure the
  `mem_ratio` bump (~2/16 blocks bf16) against the low-RAM budget — this is a serve-time change, not a
  retrain, so it composes with any A–C checkpoint.

**Lever E — the repo-coherent ship-now path (independent of the cert).**
- v6 + conformal abstention **already** serves ~93% coverage @ answered_top1 **0.983** (measured,
  `certify-lowram-olmoe-nvfp4-v6.json` abstention_frontier). That is the honest deliverable **today**;
  v7 is the attempt to make the FULL model pass so abstention isn't needed. Ship v6+abstention as the
  baseline **now** and iterate v7 in the background.

## 3. Pre-registered gates (no-overclaim)

- **GO** iff `top1 ≥ 0.97` AND `mean_kl ≤ 0.05` AND `protected_max_kl ≤ 0.1`, on the authoritative
  **n=256** cert (comparable to v5 train-08 and v6 — never quote a small-n draw as the headline).
- Each lever's cert is logged as its own row; a lever that does not move `top1` by ≥ **+0.005** is
  **dropped**, not hopefully stacked. Report the number with its n; re-run clean rather than clear a
  candidate on a mixed metric (see `rlvr-harness-traps` §C).
- A v7 that still misses 0.97 is an **honest NO-GO** → ship v6 + abstention. `canClaimAGI` stays false.

## 4. Cost + sequence (GPU-gated — this doc is the offline prep)

Each v7 train ≈ **2.7 h** on the Spark (v6's ~11–18 s/step × ~550 opt steps) + ~10 min cert; the one
GPU is **serial** — run behind the cluster's current work, owner-gated (`--run-train` escalates).

Order (stop at the first PASS or ship v6+abstention):
1. **A**: `QAT_TOP1_WEIGHT=0.75 QAT_MARGIN=0.1` (else keep the v6 recipe) → re-cert.
2. still short → **+B**: `--lora-rank-alloc` (experts r≈32).
3. still short → **+C**: `QAT_EPOCHS=7`.
4. still short → **+D**: depth mixed precision at serve (no retrain).

**Fire (when the GPU frees, owner GO on record)** — same bridge lane as `v6-train-03`, a FRESH
adapter:
```
bridge/commands/<date>-nvfp4-v7-train-01.json
  args: "--bench-b --run-train --execute"
  env:  QAT_ADAPTER=training/lora/checkpoints/olmoe-qat-spark-v7  (NOT v6/v5)
        QAT_KD_WEIGHT=1.0  QAT_TOP1_WEIGHT=0.75  QAT_MARGIN=0.1  QAT_TEMP=2.0
        QAT_LAMBDA=0.0005  QAT_EPOCHS=7
        CERT_OUT=agi-proof/benchmark-results/certify-lowram-olmoe-nvfp4-v7.json
```
The `attach_qat` fused-expert reach fix + the `_torch_nvfp4` bucketize perf + skip-gradient-
checkpointing-under-QAT are already on `main`, so a v7 fire trains at the v6 rate and the coverage
line will read `expert=32` (verify it, not the offline invariants).

**One line:** v6 solved the distribution match (`mean_kl` passes); v7 attacks the *argmax* — push the
top1-margin (β↑, margin>0), give the experts more rank, and if needed hold the final layers bf16 —
while shipping v6+abstention now. Compute isn't the blocker; the last 0.009 is an argmax problem.
