# NVFP4 cert: from knob-turning to lever-portfolio search (methodology)

**Date:** 2026-07-01 · **canClaimAGI=false** · evidence from probes run this cycle on the Spark.

## Problem
v3→v5 is a **sequential single-knob search on λ** (the QAT push penalty): top1 0.906→0.922; the
0.97 wall is on axes λ never touches. Each iteration costs a full train+cert (hours). The
pre-registered "NOT another knob" is right; what was missing is a *methodology*.

## Reframe
**The cert is a search over a portfolio of mostly *cert-time* levers; training is the last,
most expensive lever.** On the bandwidth-poor aarch64 Spark (train = hours, cert = minutes),
attack the wall *without retraining* first. The frontier re-cert proved the pattern: v5 went
FAIL → shippable in minutes, no train, by adding a *selective-coverage* lever.

## Methodology (M1–M5)
- **M1 Measure, then target.** Build a per-expert/per-layer error map from the cert's own outputs before choosing a lever.
- **M2 Cheap (cert-time) levers before expensive (train-time) ones.** Exhaust RHT / precision-allocation / keep-list / abstention-α on the *existing* adapter before any retrain.
- **M3 Pull orthogonal levers, not one knob.** Axes: outlier handling, rounding, precision allocation, distillation objective, selective coverage.
- **M4 Optimize `fidelity × coverage`** (the abstention frontier), not raw top1.
- **M5 Turn pre-registration into a lever-ROI prior** (log predicted-vs-actual per lever).

## Empirical findings (this cycle — why the ranking was REVISED)
**Probe 1 — per-expert NVFP4 weight-quant rel-error (2048 expert-modules, base OLMoE):**
mean **0.108**, p50 0.108, p90 0.109, p99 0.110, max 0.112. **Top-5% of experts carry 5.1%,
top-10% carry 10.1% of total error — perfectly flat.** `down_proj` (0.108) ≈ `gate_up_proj`
(0.107). → The FP4 weight distortion is an **intrinsic, uniform ~10.8%**; there is **no small
set of "bad experts."** The keep-`down_proj` heuristic is *not* weight-justified.

**Probe 2 — Hadamard/orthogonal rotation on the quantized dim (96 matrices):**
rel-err 0.1083 → 0.1084 (**−0.1%, no effect**). → RHT does **not** reduce the intrinsic uniform
*weight* quant error (RHT helps *activation* outliers, not Gaussian-ish weight quant).

## Revised, evidence-based lever ranking
- **REFUTED (cheap, weight-space):** expert-mixed-precision *by weight-error*; RHT-on-weights. Neither reduces the intrinsic uniform ~10.8%.
- **STILL STANDING (untested — the next cheap probe):** **routing-FREQUENCY-weighted expert protection.** Uniform *weight* error ≠ uniform *output* error: OLMoE routes 8/64 experts per token, so high-frequency experts contribute more to top1-flips. Protecting the top-k *most-routed* experts in bf16 may lift top1 at low memory cost. Needs a calib-forward router-count + per-expert output-KL probe.
- **WORKS — ship it:** the **abstention frontier** (v5 shippable @ coverage 0.86 / answered 0.982). Given the intrinsic weight error + the co-adaptation plateau at 0.92, selective prediction is the pragmatic product.
- **TRAINING lever (uncertain ceiling):** **QAD** (full soft-target KL distillation) > plain QAT — but co-adaptation already plateaued at 0.92, so QAD's headroom on *raw* fidelity is uncertain. Run it to push the *frontier's coverage* higher, not to chase raw 0.97.

## Strategic conclusion
Intrinsic uniform ~10.8% NVFP4 weight distortion + co-adaptation plateau at 0.92 ⇒ raw NVFP4 is
unlikely to clear 0.97 via cheap levers. Evidence-based strategy: **(1)** ship v5 via the
abstention frontier (the product that works); **(2)** run ONE more cheap probe (routing-frequency
expert protection) before any train; **(3)** reserve QAD for pushing the frontier's coverage, not
chasing raw 0.97.

## Next experiments (spec)
1. **[cheap, next] Routing-frequency probe** — calib forward; per-expert routing counts + per-expert output-KL; test protecting top-k most-routed experts in bf16 (the one untested cheap lever).
2. **[cheap] Frontier lever-sweep harness on v5** — {abstention-α grid} × {routing-freq keep-k ∈ 0,4,8} → Pareto (raw top1, coverage@0.97, mem_ratio). No train.
3. **[expensive, last] QAD v6** — only if 1–2 plateau below target coverage. Full soft-target KL to the FP teacher, token-scaled. Fix the 2 harness footguns first: `--target-modules attn-mlp` AND `--lora-dropout 0`.

## Provenance
Probes: `/tmp/expert_sens.py`, `/tmp/rht_probe.py` (Spark). Reuse `training/qat.fake_quant`.
Weight-space only — the load-bearing sensitivity (activation-weighted output-KL) is experiment #1 above.

## Findings continued (same cycle): routing skew + the decisive protection test
**Probe 3 — routing concentration (calib forward, 128 seqs):** despite load-balance training,
top-8 experts/layer carry **27.8% of routing** (max 32.9%, layer 4) vs 12.5% uniform — ~2.2x skew.
→ *output* error is NOT uniform even though *weight* error is; the expert-protection lever is LIVE.

**Probe 4 — protection test (base model, n=64, guard KEPT-BF16-OK):** top1-vs-bf16 with (A) all
NVFP4 = **0.8650** vs (B) NVFP4 + top-8/64 most-routed experts/layer bf16 = **0.8765** → **+0.0115**
at **12% expert-param bf16**. (A first run returned a bogus 1.0000 — `is_served_param(suffixes=())`
matched 0 params, a quant no-op; the rlvr-harness-traps too-clean rule caught it. Fixed via the full
served-suffix set → 96 params matched.)

## Refined conclusion (fully evidenced)
Expert-protection is REAL but **MODEST** (+0.0115 for 12% bf16) — it cannot clear 0.97 alone
(needs ~+0.10 → protecting a large fraction, defeating the memory purpose). Its role is a **cheap,
no-train FRONTIER-IMPROVER**: +~1pt raw top1 buys back abstention coverage. Net strategy:
1. **Ship v5 via the abstention frontier** (the product that works; cloud building the serve-time gate).
2. **Optionally stack top-k expert-protection** (12% bf16, +~1pt) to raise the frontier's coverage — cheap, no train.
3. **QAD only if coverage still short** — not to chase raw 0.97.
4. **Do NOT chase raw NVFP4 0.97 via cheap levers** — intrinsic uniform ~10.8% weight error + modest protection gain make it infeasible.
Probes: `/tmp/{expert_sens,rht_probe,routing_probe,protect2}.py` (Spark).
