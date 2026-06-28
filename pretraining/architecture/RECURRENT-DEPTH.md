# Recurrent-Depth Transformers — the OpenMythos takeaway, and a feasible plan

> Honest, falsifiable, no-overclaim. This note records what the
> [OpenMythos](https://github.com/kyegomez/OpenMythos) reconstruction of Claude-"Mythos"
> actually offers, what is transferable to Sophia, and a phased plan that de-risks the
> mechanism on CPU **before** any GPU compute is spent. The Phase-0 study referenced here
> is `recurrent_depth.py` — pure-Python, CI-checked, real gradients.

## 1. What OpenMythos is (and is not)

OpenMythos is a **theoretical reconstruction**, not a trained model: no released weights, no
empirical benchmarks. Its performance lines ("770M loop ≈ 1.3B fixed", "2–3× throughput")
are quoted *predictions from cited papers*, not measured OpenMythos results. What it does
contribute is a clean statement of a real, current research direction — the
**Recurrent-Depth Transformer (RDT)** / looped transformer:

```
prelude  →  recurrent block looped ×N (shared weights)  →  coda
h_{t+1} = A·h_t + B·e + block(h_t, e)
```

with three load-bearing ideas:

1. **Looping = latent chain-of-thought.** Each loop is a reasoning step in continuous space
   (no token emitted). Depth becomes adjustable *at inference* — run more loops on harder
   inputs (Saunshi et al. 2025; Geiping et al. recurrent-depth).
2. **LTI-constrained injection (the "Parcae" stability device).** Constrain the recurrence so
   its spectral radius stays `< 1`, or training over many loops diverges.
3. **Fine-grained MoE + adaptive halting.** Many small experts, always-on shared experts, and
   a learned halt to avoid "overthinking".

**"Better than Claude" from scratch is not the feasible goal** — and Sophia's whole ethos
(the no-overclaim gate, the failure ledger) exists to stop that kind of claim. Claude is a
multi-billion-dollar scale/data/RLHF operation; you cannot out-scale it. You **can**
out-*specialize* it on Sophia's axis: grounded, abstaining, provenance-faithful reasoning,
measured under a pre-registered gate.

So the reframed, defensible objective:

> **A small recurrent-depth model whose extra loops are spent verifying against provenance
> before halting — beating same-size (and ideally larger) baselines on
> hallucination / calibration / provenance, measured under the ≥2-judge, CI-excludes-zero
> gate.** That is "better than Claude *at the thing Sophia measures*", which the existing
> machinery can actually validate.

## 2. The research thesis

> **Verification-Gated Recurrent Depth (VGRD):** spending test-time loop iterations on
> provenance checking before halting reduces hallucination at fixed parameter count.

It unifies two lines neither repo has alone — looped-transformer latent reasoning
(Saunshi/Parcae/Geiping) and Sophia's abstaining provenance gate — under Sophia's
no-overclaim protocol. The novel coupling: **compute = verification depth**. The halting
signal (loop count) becomes a *second* confidence signal alongside self-consistency, which
RESULTS.md already validated as the only working one.

## 3. Phase 0 — de-risk the mechanism on the nano substrate (done, $0 GPU)

Before trusting any of OpenMythos's claims, validate the *mechanism* on the
known-floor `pretraining/nano/` substrate. `recurrent_depth.py` implements a real,
hand-backpropped RDT (BPTT verified against finite differences to `< 1e-4`) and measures the
three claims against closed-form controls. **Measured (full mode, seed 0):**

| Claim | Control | Result |
|---|---|---|
| **LTI stability** | exact diagonal ρ = max\|A_i\| | constrained ρ = 0.95 (`<1` by construction); free-run ‖h‖ grows **12.9×** over depth vs **198 494×** unconstrained |
| **Depth extrapolation** | 1/V chance = 0.17 | trained on ≤5 hops → **83–100%** at 6/7/8/**10** hops (unseen, up to 2× the deepest trained depth) by running more loops |
| **Parameter efficiency** | exact param counts | shared block **1 776** params at 100% acc vs unshared **7 104** (4×) — sharing buys depth at ¼ the block params |

Crucially, the study also records the **honest caveat** that distinguishes it from
hand-waving: the diagonal constraint bounds the *state* but **not** the BPTT gradient (the
full recurrent Jacobian `diag(A)+W_rec` can still have ρ > 1), so training through many loops
*also* needs gradient clipping / a small recurrent block — the LTI constraint is necessary
for state-boundedness, not sufficient for trainability. Conflating the two would overclaim.

```bash
python -m pretraining.architecture.recurrent_depth --quick   # seconds, CI mode
python -m pretraining.architecture.recurrent_depth           # full, writes *-latest.json
python -c "from pretraining.architecture.recurrent_depth import offline_invariants as f; print(f()[0])"
```

**Decision gate met:** the looped-transformer mechanism is real and reproducible at nano
scale → a GPU-scale build is warranted. Had any sub-study failed, the plan would stop here
having spent $0.

## 4. Phases 1–3 — the GPU plan, grounded in existing Sophia infra

The repo already has everything the scale-up needs: RunPod MCP access, a pre-baked CUDA
image + cost-guard runbook (`docker/wisdom-pilot`, the `wisdom-gpu-prebaked` skill), a
validated SFT/ORPO pilot (`tools/runpod_wisdom_pilot_selfreport.py`), the scaling-law
fitter (`pretraining/scaling/fit.py`), and the no-overclaim eval harness.

**Phase 1.1 — PyTorch RDT + local CPU validation (DONE, $0).** `rdt_torch.py` is the
GPU-trainable RDT, re-derived (not imported from OpenMythos): prelude → `[LTI-inject(e) →
shared block] × n_loop` → coda, with RMSNorm + GQA + RoPE + SwiGLU/MoE blocks, a halting
head, and tied LM head. The LTI gate is the S4/Mamba zero-order-hold discretization
`a = exp(-(softplus(A_log)·softplus(log_dt)) - min_decay)`, so the state pole is **strictly
< 1 for every parameter value** (max 0.9999 over 200 hostile random inits — the stability
guarantee is structural, not a training outcome). Validated on CPU for $0 by `self_test()`:
dense+MoE forward/backward with finite gradients, depth-changes-output (the loop is
load-bearing), and a tiny-batch overfit that drives loss **4.14 → 0.007** (the recurrence
learns). `rdt_pretrain.py` runs the full data→loss→optimizer→checkpoint pipeline as a
hermetic CPU smoke (loss below uniform + checkpoint round-trip), and the *same code path*
scales to GPU by config.

```bash
python -m pretraining.architecture.rdt_torch --self-test     # CPU, seconds, $0
python -m pretraining.architecture.rdt_pretrain --smoke       # CPU pipeline smoke, $0
```

**Phase 1.2 — small from-scratch pretrain (1 RunPod node, cost-guarded).** Train `~0.5B` on
FineWeb-Edu `sample-10BT` via `rdt_pretrain.py --dataset fineweb-edu` under the cost-guard
runbook (`wisdom-gpu-prebaked` skill: stock-torch image, cheap validation first, restart-loop
abort, confirm zero leaked pods). The cost-guard rule is why Phase 1.1 is CPU-validated first:
**no GPU credit is spent on unvalidated code.** Deliverable: a base checkpoint **and** a
looped-vs-fixed scaling-law fit at matched compute (reuse `scaling/fit.py`).

*Launch (SSH-free, self-reporting, cost-guarded).* `tools/runpod_rdt_pretrain.py` + the
`rdt-pretrain-runpod` GitHub Actions workflow rent a pod that clones the branch, runs the RDT
self-test + the training run on the GPU, git-pushes the train report to
`agi-proof/benchmark-results/rdt-pretrain/`, then self-deletes (the launcher also deletes the
pod the moment a fresh report lands). **Always dispatch `mode=smoke` first** — a few hundred
GPU steps (~minutes, cheapest GPU) that validate the CUDA path before any real spend. Only
dispatch `mode=pretrain` after a green smoke. Requires the `RUNPOD_API_KEY` Actions secret
(no HF token — FineWeb-Edu and the byte tokenizer are ungated). The workflow registers for
dispatch once it lands on the default branch.

```bash
python tools/runpod_rdt_pretrain.py --dry-run --mode smoke      # inspect payload, no cost
# then, from CI with RUNPOD_API_KEY: dispatch rdt-pretrain-runpod (mode=smoke) → mode=pretrain
```

**Phase 2 — wisdom alignment + the novel coupling.** SFT + ORPO on the Sophia corpus
(`training/corpus.jsonl`, `moral_gate_sft.jsonl`) via the validated pilot path. Then train
the **halting head** so the model spends *extra loops* on claims the provenance verifier
would reject, and learns to emit the abstain token instead of fabricating — making
"compute = verification depth" a learned behavior, not a wrapper.

**Phase 3 — measure under the no-overclaim gate.** Run it through the existing harness:
attribution-hallucination delta, SimpleQA-Verified selective-prediction lift, the
legal-citation verifier. **Headline only what clears ≥2 judge families + CI-excludes-zero**;
log everything else illustrative — exactly as RESULTS.md does. Win condition: the small
gated RDT beats a same-size dense baseline (ideally Gemma-3-4B) on calibration/hallucination,
with the **loop-count halting signal validated as a second confidence signal** alongside
self-consistency.

## 5. What this is NOT (honest scope)

- **Not a trained model and not a capability claim** about Claude, Mythos, or any frontier
  system. Phase 0 is a *nano-scale methodology study of the looped-transformer mechanism*.
- **Ignore the 100B/1T OpenMythos configs.** The defensible contribution is at ≤4B with a
  *better-measured* claim, not a bigger one.
- **Treat OpenMythos code as reference pseudocode**, not a dependency to `pip install` and
  trust — it has no tests and no validated weights.
- A "better than X" claim requires the same measurement **at scale** under the no-overclaim
  gate (≥2 judge families, CI excluding zero). Phase 0 only earns the right to start.

## References (for the reader, not fetched here)

- Saunshi et al. (2025) — *Reasoning with Latent Thoughts: On the Power of Looped Transformers.*
- Geiping et al. — recurrent-depth / latent-reasoning transformers.
- "Parcae" — scaling laws for stable looped LMs (LTI-constrained injection), as cited by OpenMythos.
- Dehghani et al. (2018) — *Universal Transformers* (adaptive computation time / halting).
- Bae et al. (2024) — *Relaxed Recursive Transformers* (depth-wise LoRA).
- kyegomez/OpenMythos — the reconstruction this note responds to.
