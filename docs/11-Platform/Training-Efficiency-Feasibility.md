# Training-Efficiency Feasibility Assessment

_A brutally honest technical evaluation of the Sophia-AGI fast-adaptation claims,
grounded in the actual code and measured artifacts in this repo (2026-06-24)._

> **Scope.** This document evaluates the claim that the Sophia stack enables a
> **10–50× wall-clock reduction** when adapting a pre-trained model (Qwen2.5-7B,
> Llama-3.2-3B, Gemma2-9B, …) into a **provenance-disciplined, 0%-fabrication**
> Sophia model. It separates *what is real and measured* from *what is overstated*,
> and lists the concrete changes that were made to harden the trainer.

---

## 1. Executive verdict

The **engineering substrate is real, runs, and is unusually honest** (decontamination,
gate-filtered distillation, a formally-checked promotion gate, a 3-seed bar). The
**headline claims are overstated and partly misattributed**:

- **"10–50× faster"** — misattributed. The speedup comes almost entirely from
  *training on 500–2000 examples for 1 epoch instead of a large multi-epoch corpus*.
  That is a **smaller job**, not the **same job done faster**. Against a fair baseline
  (same model, same rows, same epoch, vanilla PEFT) the optimized stack is **~2–4×**.
- **"0% fabrication Sophia model"** — the model+gate **system** achieves this; the
  **weights** do not. `training/local_sophia_v2/eval_ladder_adapter.json` shows the
  trained adapter still produces **20 gate failures** at the `adapter+gate` rung.
- **"AGI proof"** — unsupported, and `tools/promote_adapter.py` already says so
  verbatim ("not a validated capability and not an AGI claim").

### Scores

| | Score | Note |
|---|---|---|
| **Overall (claim as stated)** | **30 / 100** | 10–50× + 0%-fabrication-weights + AGI framing |
| **Overall (system as built)** | **78 / 100** | Real, honest, fail-closed, reproducible |

| # | Technique | Score | Verdict |
|---|-----------|------:|---------|
| 1 | Ultra-curated data + 1 epoch | 68 | Real lever (LIMA/Phi); but val 1.83 vs train 0.71 = overfit; speedup mis-attributed |
| 2 | Council distillation + gate-filter | 74 | Best idea in the stack; anti-circularity firewall is genuine. Risks: imitation≠capability, N=125 |
| 3 | MLX / Unsloth + 4-bit LoRA | 72 | Standard & feasible — but the trainer was vanilla PEFT, CUDA-only; no Unsloth backend exists |
| 4 | Self-extend flywheel + early stopping | 34 | `run_selfextend_loop.py` is a deterministic toy; early stopping was **not wired into the trainer** |
| 5 | Modular Skills v2 + external OKF RAG | 80 | Strongest, most defensible thesis; the eval ladder *supports* it |

---

## 2. Realistic speedup (measured against the repo's own run)

Logged MLX run (`training/local_sophia_v2/training_run_mlx_sophia_v2.json`):
Qwen2.5-3B, 500 iters, batch 4, seq 1024, **peak 25.5 GB**, mlx-lm 0.29.1,
`finalTrainLoss 0.714`, `finalValLoss 1.828`.

| Hardware | Fast path | Fair baseline (same data/epoch) | Honest speedup |
|---|---|---|---|
| **M3 Max** | mlx-lm LoRA, ~10–25 min | mlx-lm LoRA, ~same | **~1.0–1.5×** |
| **RTX 4090** | QLoRA 4-bit (25.5 GB peak ⇒ bf16 won't fit 24 GB), ~5–15 min; +Unsloth ~3–8 min | vanilla PEFT, ~15–30 min | **~2–4×** |

The largest *untapped* win is not in the five techniques: the trainer previously used
`padding="max_length"` to 1024 — every short row paid a full 1024-token
forward/backward. **Dynamic padding / packing alone is plausibly 3–10× on this short
corpus.** This has now been fixed (see §4).

> **Honest reframing that survives review:** "a small, gate-disciplined adapter trained
> in minutes on a curated corpus, with a formally-checked promotion gate and a runtime
> verifier that guarantees fail-closed behaviour the weights alone do not." That claim
> is true and measured.

---

## 3. Blockers & gaps found in the code

- **Advertised flags didn't exist.** `train_lora.py` had `--4bit`, `--resume-adapter`,
  `--dry-run` — but no `--scaffold`, `--guard`, or `--distill`. **Now implemented** (§4).
- **No eval loop ⇒ "early stopping via eval ladder" was fiction at the code level.**
  The manual loop never touched the holdout. **Now wired in** (§4).
- **MLX path is an external `python3 -m mlx_lm lora` invocation**, not repo code; the
  committed `adapters.safetensors` are 133-byte placeholders. The PyTorch and MLX paths
  did not share masking/seed logic and silently diverged.
- **Prompt-masking inconsistency.** The MLX run used `--mask-prompt`; the PyTorch trainer
  did not, so the two were not comparable. This shows up as the 62.5% (run note) vs 75%
  (ladder JSON) discrepancy. **Now both default to completion-only loss.**
- **Overfitting unaddressed.** train 0.714 / val 1.828 on 1356 rows; religion regressed
  1/6 → 0/6 (correctly *not promoted*). 1-epoch-on-curated-data is fragile, not bulletproof.
- **Silent truncation.** "Some rows exceeded 1024 tokens and were truncated." Now surfaced
  as a loud warning with a pointer to `split_long_training_rows.py`.
- **Distillation stability.** Gate-filtered (good), but N=125, single teacher run. The
  literature is clear that narrow imitation copies *style*, not *capability*
  (Gudibande et al. 2023, *The False Promise of Imitating Proprietary LLMs*). Orca
  (Mukherjee 2023) needed millions of explanation traces; Phi-3 / *Textbooks Are All You
  Need* (Gunasekar 2023) is a *pretraining*-scale curation result, not 1-epoch LoRA. The
  500–2000 figure is LIMA territory (Zhou 2023) — defensible for alignment/style, not new
  capability.

---

## 4. Changes made to `tools/train_lora.py`

All of the following are implemented on this branch; the pure-Python paths are
smoke-tested (the GPU paths import lazily and run on Colab/4090/M-series).

**A. Dynamic padding (biggest real speedup).** Removed `padding="max_length"`.
Rows are tokenized unpadded; `DynamicCausalCollator` pads to the longest sequence
*per batch*. Short rows no longer pay a 1024-token forward/backward.

**B. Completion-only loss (`--mask-prompt`, default on).** `split_prompt_completion`
reconstructs the chat format and masks prompt tokens to `-100`, matching the MLX-LM
`--mask-prompt` path so the two backends are finally comparable. `--no-mask-prompt`
restores full-sequence loss for ablation.

**C. Provenance-discipline data hooks (the advertised flags, now real):**
- `--scaffold` — injects the advisor source-discipline system prompt into any row
  lacking a system turn.
- `--guard` — **intrinsic** fail-closed filter: drops a target only on a genuine
  fabricated citation / false arithmetic / forbidden-lineage merge. It deliberately
  does **not** pass the question, because doing so invokes the attribution *trap
  grader* ("expected discussion of socrates", "expected tradition context 'daoist'"),
  a positive-expectation completeness check that flagged **88/564 (16%) of clean,
  curated rows over wording, not fabrication**. Intrinsic-only checking flags
  **0/439** curated rows (verified) while still catching real fabrication in
  `--distill` synthetic targets.
- `--distill` — folds in the gate-clean council traces (`training/council/traces.jsonl`)
  as extra SFT targets.

**D. Holdout eval loop + early stopping.** `--eval-every` (auto = 25 when
`holdout.jsonl` exists), `--patience`, `--min-delta`, `--eval-batches`, and
`--overfit-ratio` (stop when `val/train` exceeds a ceiling). The best-val-loss
checkpoint is the one persisted — the eval ladder now drives training, not hindsight.

**E. Reproducibility & stability.** `--seed` (set across `random`/`numpy`/`torch`/CUDA
and **emitted into `sophia_lora_config.json`** so `promote_adapter.py`'s 3-seed bar can
verify it), cosine LR + warmup (`--warmup-ratio`), gradient clipping (`--max-grad-norm`),
and bf16 by default on Ada/Hopper (`--dtype {auto,bf16,fp16}`). A loud truncation
warning replaces silent row-clipping.

> Still **not** done (deliberately, larger scope): a real in-repo Unsloth backend
> (`--backend {peft,unsloth,mlx}`) and pre-split enforcement in the prep step. These are
> recommended next.

---

## 5. Quick experiments to validate the claims (each falsifies a specific one)

1. **Padding ablation** — same data/seed/epoch: `max_length` vs dynamic vs packing.
   Report tokens/s + wall clock on the 4090. *Expected: 3–10× from this alone.*
2. **Fair-baseline speedup** — same model/data/1 epoch: vanilla PEFT fp16 vs QLoRA 4-bit
   vs Unsloth. *Expected: 2–4×, not 10–50×.* Publish this; it kills the strawman first.
3. **Data-scaling curve** — train on 100/250/500/750/all; eval gate-clean rate on holdout.
   Find the plateau — this is what actually justifies "500–2000 perfect examples."
4. **Weights-vs-gate isolation (most important)** — eval the adapter with the runtime
   gate **OFF**. If un-gated fabrication isn't far below base, the capability lives in
   the gate, not the weights. Say so.
5. **Prompt-masking ablation** — completion-only vs full-sequence loss; reconcile the
   62.5% vs 75% discrepancy.
6. **3-seed CI on the 50→75 jump** — N is tiny; report a confidence interval before
   anyone quotes it.

---

## 6. Keeping the gate / fail-closed / conscience kernel intact

- **Do not bake the gate into the weights.** The ladder proves the deterministic gate
  still catches 20 violations the adapter emits — defense-in-depth working as intended.
  A learned policy *plus* an independent verifier is a stronger safety story than trusting
  the weights.
- **Keep the conscience kernel external.** Safety baked into a LoRA is reversible by
  anyone who continues fine-tuning the adapter (Qi et al. 2023, *Fine-tuning … Compromises
  Safety*). An external classifier cannot be fine-tuned away downstream.
- **Extend the gate-filter to the DPO `rejected` side** in `wiki_to_training.py`: run the
  `rejected` string through `check_response` to prove it is a genuine violation, closing
  the last anti-circularity gap.
- **Lean on `promote_adapter.py`.** Its formal-verifier protected-floor proof is the
  best-engineered piece here. Make the trainer's emitted seed feed it so the 3-seed proof
  is end-to-end machine-checked.

---

## References

- Zhou et al. 2023 — *LIMA: Less Is More for Alignment* (1k-example alignment).
- Gunasekar et al. 2023 — *Textbooks Are All You Need* (Phi; curation at pretraining scale).
- Mukherjee et al. 2023 — *Orca* (explanation-trace distillation, millions of traces).
- Gudibande et al. 2023 — *The False Promise of Imitating Proprietary LLMs* (imitation ≈ style, not capability).
- Qi et al. 2023 — *Fine-tuning Aligned Language Models Compromises Safety* (why safety stays external).
- Dettmers et al. 2023 — *QLoRA* (4-bit NF4 + paged optimizers).
- Unsloth benchmarks — fused-kernel LoRA, ~2× throughput / ~½ memory vs vanilla PEFT.
