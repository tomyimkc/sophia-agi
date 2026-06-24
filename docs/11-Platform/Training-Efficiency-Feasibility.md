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
  (same model, same rows, same epoch, vanilla PEFT) the optimized stack is **2.0×,
  measured on an RTX 4090** (§2b) — from dynamic padding; QLoRA and Unsloth did not beat
  it at this scale.
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
| **RTX 4090** | fp16 + dynamic padding | vanilla PEFT + full-length padding | **2.0× (measured, §2b)** |

The largest *untapped* win is not in the five techniques: the trainer previously used
`padding="max_length"` to 1024 — every short row paid a full 1024-token
forward/backward. **Dynamic padding / packing alone is plausibly 3–10× on this short
corpus.** This has now been fixed (see §4) and **measured at 2.0× on an RTX 4090** (§2b).

> **Honest reframing that survives review:** "a small, gate-disciplined adapter trained
> in minutes on a curated corpus, with a formally-checked promotion gate and a runtime
> verifier that guarantees fail-closed behaviour the weights alone do not." That claim
> is true and measured.

### 2b. Measured speedup — RunPod RTX 4090 (spot), 2026-06-24

The fair-baseline benchmark (`tools/bench_lora_speedup.py` via the `speedup-runpod`
workflow) on the device class the claim targets — Qwen2.5-3B-Instruct, **128 rows, 1
epoch, 32 optimizer steps, seed 0**, identical across configs. Report:
`agi-proof/benchmark-results/runpod-speedup/run-28125095723.speedup_report.json`.

| Config | Wall (s) | Speedup vs standard-LoRA ref |
|---|---|---|
| `peft-fp16-maxpad` *(standard-LoRA reference: full-length padding)* | 58.3 | 1.00× |
| **`peft-fp16-dynpad`** *(the one fix that matters)* | **29.04** | **2.01×** |
| `peft-4bit-dynpad` (QLoRA) | 36.26 | 1.61× |
| `unsloth-4bit-dynpad` | 63.15 | 0.92× |

**The measured, defensible multiplier vs standard LoRA is ~2.0×, from dynamic padding
alone.** Three honest findings, none flattering:
- **QLoRA 4-bit was *slower* than fp16 dynamic padding** (36.3 s vs 29.0 s). It only
  beats the reference because it *also* uses dynamic padding. 4-bit's quant/dequant
  overhead isn't worth it when a 3B already fits in fp16 on a 4090 — 4-bit is for
  *fitting bigger models*, not speed.
- **Unsloth was net *slower* than the reference (0.92×).** Its compile/patch overhead
  dominates a 32-step micro-run and never amortizes; the advertised ~2× shows up on long
  runs, **not** on this bench. Do **not** quote a 2× Unsloth number from this evidence.
- The 2.0× from padding is **below my own 3–10× estimate** for this subset — magnitude
  scales with how short the rows are relative to `max_seq_len` and with subset size; at
  128 rows it is a conservative, real 2.0×.

> **Bottom line, now measured on both platforms:** the apples-to-apples speedup vs
> standard LoRA is **~2×** (dynamic padding), **not 10–50×**. The 10–50× headline is
> the corpus-shrink lever (fewer examples = a smaller, narrower job), which this bench
> deliberately does not measure. Quote "~2× wall-clock at fixed data/epochs (RTX 4090,
> measured)"; do not quote QLoRA or Unsloth as speedups at this scale.

### 2a. Measured run — Mac Studio M3 Ultra (96 GB), 2026-06-24

First end-to-end run of the hardened stack on real hardware (mlx-lm, Qwen2.5-3B-Instruct,
full `--scaffold --guard --distill`, completion-only loss, batch 1, 1 epoch):

| Metric | Value |
|---|---|
| Wall clock | **177 s** (~3 min) |
| Peak unified memory | **10.2 GB** (cf. 25.5 GB for the earlier batch-4 run — memory scales with batch) |
| Throughput | 78.2 it/s · 456.6 tok/s · 75,293 trained tokens |
| Iters | 553 (1 epoch over 553 fitted rows) |
| `--guard` dropped | 0 (curated corpus is intrinsically clean) |
| Pre-split | 11 council traces dropped as unsplittable-at-1024 (surfaced, not silent) |
| **Final train loss** | **1.746** |
| **Final val loss** | **2.894** (val/train = **1.66 → overfitting confirmed**, more pronounced than the v2 run's 0.71/1.83) |

**What this DID and did NOT prove:**
- ✅ The full provenance-disciplined adapter trains in **~3 min at ~10 GB** on an M3 Ultra —
  cheap and fast in *absolute* terms, and the pipeline (scaffold/guard/distill/pre-split)
  runs clean end-to-end.
- ✅ The **speedup multiplier is now measured on CUDA (§2b): ~2.0×**, not on this Mac.
  Experiment #2 (`bench_lora_speedup.py`) needs CUDA + bitsandbytes + Unsloth and **cannot
  run on Apple Silicon**; experiment #1 (padding ablation) is **invalid on MLX** —
  `--pad-to-max` is a no-op there because mlx_lm controls its own padding (the device run
  correctly rejected the naive 159.7 s / 177.0 s = 0.90× as run-to-run variance, not a
  padding effect). The trainer now warns loudly about both rather than failing silently.
  Both ran on the RunPod RTX 4090 instead — see §2b.
- ⚠️ The **train/val gap (1.75 vs 2.89) confirms the overfitting risk** from §3. mlx-lm runs
  all iters with no Sophia early-stop, so the run did not stop early. To measure the real
  speedup multiplier and to exercise early-stopping, run on a CUDA box.

> **Net:** the absolute MLX numbers are good and now reproducible; the headline *multiplier*
> is now **measured at ~2.0× on a CUDA RTX 4090 (§2b)** — quote that, not the old estimate.

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

**F. Backends (`--backend {peft,unsloth,mlx}`).** `peft` is the vanilla CUDA path.
`unsloth` uses `FastLanguageModel` fused kernels (~2× throughput / ~½ memory) and feeds
the same manual loop. `mlx` builds the MLX chat-data dir in-repo (via `fit_rows`) and
invokes `python -m mlx_lm lora` with the seed/mask-prompt args, matching the logged v2
run — so the MLX logic no longer lives only in shell history. Optional deps are listed
in `requirements-lora.txt`. **MLX caveats (now warned at runtime):** `--pad-to-max` is a
no-op on `mlx` (mlx_lm owns padding), and Sophia early-stopping (`--patience`/
`--overfit-ratio`) is `peft`/`unsloth`-only — on `mlx`, mlx_lm reports validation every
`--steps-per-eval` but runs all iters. Use `--backend peft` (CUDA) for the padding
ablation and early-stopping.

**G. Pre-split enforcement in prep.** `tools/prepare_lora_dataset.py` now runs `fit_rows`
(offline conservative heuristic) over train and holdout by default (`--max-tokens`,
`--no-presplit`), so over-long rows are split at turn boundaries before they can be
silently truncated. The fit report is recorded in the manifest.

**H. DPO `rejected` anti-circularity guard.** `tools/wiki_to_training.py` now gate-checks
every `chosen` target (intrinsic, fail-closed) and drops self-contradictory pages
(attributed author appearing in its own `doNotAttributeTo`). The `rejected` sample's
wrongness is established by the wiki provenance graph itself — the deterministic gate has
no canonical attribution table for arbitrary titles and is *honestly not relied on* to
adjudicate it (verified: it returns no violation for a wrong free-text attribution).

**I. Seed → promotion provenance.** `tools/promote_adapter.py` reads the trainer-emitted
seed from `sophia_lora_config.json` (`--adapter-config`) and records it in the promotion
report + artifact list, so the reproducibility provenance the 3-seed bar relies on is
machine-traceable.

**J. Speedup benchmark.** `tools/bench_lora_speedup.py` runs experiment #2 directly:
same model/data/1 epoch across {fp16-maxpad reference, fp16-dynpad, bf16-pack,
QLoRA-4bit, Unsloth-4bit}, reporting wall clock + speedup vs the standard-LoRA reference.
`--dry-run` validates plumbing offline; real numbers need a CUDA GPU.

### 4a. Research-backed knobs added (2026 literature review)

A three-thread literature review (throughput / adapter-quality / distillation) drove the
following, each tied to the measured overfitting (train 1.75 / val 2.89) or the remaining
speed headroom. *Faster:*
- **`--pack`** — sequence packing via `DataCollatorWithFlattening` + Flash-Attention
  varlen (`--attn flash_attention_2`); concatenates short rows so no step wastes tokens.
  HF measured ~2× on short, high-variance data — *on top of* dynamic padding. Completion
  `-100` masks preserved (loss-masking ⟂ attention-masking).

*Better (anti-overfitting):*
- **`--lr` default lowered 2e-4 → 5e-5** — LR is the #1 small-data knob; controlled 2026
  studies find it dominates the choice of LoRA variant.
- **`--rslora`** — rank-stabilized scaling (α/√r) fixes the over-aggressive α/r = 2.
- **`--neftune-alpha`** (try 5) — embedding-noise regularizer that helps instruction
  tuning most on small data (NEFTune, arXiv 2310.05914).
- **`--weight-decay`, `--lora-dropout`, `--target-modules all-linear`** — cheap
  regularizers; all-linear targets MLP (the dominant locus of adaptation).

*Discipline transfer:* **`tools/train_orpo.py`** — ORPO preference training on the
gate-clean chosen/rejected pairs from `wiki_to_training.py`. Abstention/citation is
*contrastive*, so preference methods instil it better than SFT; ORPO is single-stage and
reference-model-free. Recommended pattern: SFT for format → ORPO for discipline; mix in
unanswerable negatives to avoid the refusal-forgetting "hallucination tax".

> Honest caveat: exotic LoRA variants (DoRA/LoRA+/PiSSA) buy only ~1–2% over a
> well-tuned vanilla LoRA per the controlled studies — so effort went to LR, scaling,
> NEFTune, packing, and data quality, not variant-chasing. Several supporting citations
> are post-knowledge-cutoff (2026 arXiv ids) and must be verified before entering the
> public proof package; the core techniques (LIMA, NEFTune, rsLoRA, packing, ORPO,
> Gudibande) are well-established.

---

## 5. Quick experiments to validate the claims (each falsifies a specific one)

1. **Padding ablation** — `tools/train_lora.py --pad-to-max` vs default dynamic padding,
   same data/seed/epoch. *Expected: 3–10× from this alone.*
2. **Fair-baseline speedup** — `python tools/bench_lora_speedup.py` (fp16-maxpad ref vs
   fp16-dynpad vs QLoRA-4bit vs Unsloth-4bit). *Expected: 2–4×, not 10–50×.* Publish this;
   it kills the strawman first.
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

## 6. Repo-development roadmap (fastest training, gate upheld)

The fastest path to a stronger Sophia adapter is *not* a bigger run — it is a staged
pipeline where each phase pays for itself and the deterministic provenance gate is never
weakened. The gate is **fail-closed**: abstention is a correct output, not a failure.

| Phase | Name | What it delivers | Gate interaction |
|---|---|---|---|
| **P0** | Infra / prebaked image | A pinned CUDA image (torch/peft/trl/bitsandbytes/unsloth/mlx baked in) so a run starts in seconds, not minutes of `pip`. Removes the cold-start tax that swamped the §2b micro-bench. | None — pure plumbing. The gate ships in the image unchanged. |
| **P1** | Config-transfer, no sweep | Apply the 2026 *LoRA Without Regret* defaults: `--target-modules all-linear`, LoRA LR ≈ 10× full-FT LR and **~rank-independent** (so no per-rank LR sweep), effective batch < 32, 1 epoch on small curated data. **(This component.)** | None — config only. Intrinsic `--guard` still runs at the data layer. |
| **P2** | Gate-filtered RFT data engine | Rejection-sampled / council-distilled targets, each passed through the **intrinsic** fail-closed gate (`check_response(text, mode="advisor")["violations"]`, *no* question) to drop fabricated citations / false arithmetic / forbidden-lineage merges before they enter SFT. | Gate as a **fail-closed data filter**. Must NOT pass a question (that invokes the attribution trap-grader, a positive-expectation completeness check that wrongly deletes clean curated rows over wording). |
| **P3** | Gate-as-reward GRPO | Online RL (GRPO/RFT) where the verifier signal is the reward. **Abstention must be reward-positive**: a correct "I can't verify that" earns reward, never a penalty. | Gate as **reward**. The single most dangerous coupling — see caveats below. Hold the fail-closed semantics fixed; never let the reward train *out* of abstention. |
| **P4** | Gate-honesty hardening | Trap red-team set (attribution traps, fabricated-source bait), explicit **false-negative / false-positive** accounting on the gate, and **multi-family** evaluation (≥3 base-model families) so results aren't a single-model artifact. | Adversarial pressure *on the gate itself* — measures, never relaxes, fail-closed behavior. |

**Honest caveats (these are the load-bearing risks, stated plainly):**

- **Abstention-collapse is the #1 risk.** Naive RLVR scores an abstention as *wrong*
  (no positive reward for "I won't fabricate"), so the policy learns to stop abstaining —
  i.e. RL *trains out* the fail-closed behavior the gate exists to protect. **Fix:**
  abstention is **reward-positive** in P3. A correct refusal-to-fabricate is a correct
  output and must be rewarded as one. This is non-negotiable and overrides any raw
  accuracy metric.
- **Verifier ceiling.** The gate certifies **absence-of-violation, not correctness.**
  Passing the gate means "no fabricated citation / false arithmetic / forbidden merge was
  detected," *not* "the answer is true and complete." Optimizing hard against the gate can
  produce gate-clean-but-vacuous answers; pair it with task-quality signals.
- **Reward-hacking.** A policy trained against a gate will learn that gate's blind spots.
  **Mitigation:** hold out a **stronger / independent gate** and a separate **trap-prompt
  set** for evaluation only (P4), so the training gate is never the same artifact that
  certifies the result.
- **Citation hygiene.** Many supporting 2026 results (incl. *LoRA Without Regret* and
  several GRPO/RFT papers) are **post-knowledge-cutoff** and **must be independently
  verified before entering the proof package.** The long-established anchors (LIMA,
  NEFTune, rsLoRA, packing, ORPO, Gudibande) are safe to cite now; the 2026 arXiv ids are
  not, until checked.

No phase weakens the gate; each either leaves it untouched (P0/P1), uses it fail-closed
(P2), keeps abstention reward-positive (P3), or adversarially measures it (P4).

---

## 7. Keeping the gate / fail-closed / conscience kernel intact

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
