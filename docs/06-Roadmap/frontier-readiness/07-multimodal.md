# 07 — Multimodal: from bench to credible training/modeling

**Author:** multimodal-model research engineer (planning doc)
**Repo:** `/home/user/sophia-agi`
**Date:** 2026-06-26
**Status:** plan only. No capability claims. Every number below is a *target*, gated by the repo's no-overclaim bar.

> One-line thesis: the repo today has an honest multimodal *evaluation* harness (`multimodal_bench/`) but **zero multimodal modeling**. The shortest credible path to "I have trained a multimodal model" — and the one that maps cleanly onto frontier-lab Research/Product Engineer bars (Anthropic Audio/multimodal, Anthropic Computer Use, DeepMind multimodal) — is to **train a LLaVA-style projector** that connects a frozen ViT/SigLIP encoder to the repo's existing local LLM (Qwen2.5-3B-Instruct, already wired for LoRA), do visual instruction tuning, then turn the existing GUI-agent harness into a *grounded-click* computer-use loop. The repo's provenance/abstention moat ports directly to vision as a **visual-grounding / abstention eval** — which is the differentiator, not raw VQA score.

---

## 1. Thesis & references

### 1.1 The strategic claim

Frontier labs hire multimodal engineers who can (a) connect a vision encoder to an LLM and train the connector/adapter, (b) do visual instruction tuning, (c) build *grounded* computer-use agents, and (d) measure hallucination/grounding honestly. The repo already over-indexes on (d) — its `multimodal_bench/` is a genuinely good measurement harness with judge-free verifiers, multi-judge consensus, CIs, and calibrated abstention. What is missing is **(a)–(c): any actual modeling**. This plan closes that gap with the *minimum* real training that is still credible, while preserving the repo's honest/measured ethos and tying multimodal into its provenance/abstention theme (visual grounding = the multimodal analog of citation-grounded abstention).

The connector-training route (LLaVA) is deliberately chosen over full native-multimodal pretraining: it is the highest capability-per-GPU-hour on-ramp, it is what most open VLMs actually do, and it is honest about the repo's scope ("Sophia consumes an open backbone and adds the trust layer" — VISION.md). It is also the route that produces a *trainable artifact* on a single GPU in days, not a frontier-scale run.

### 1.2 Architecture families (cited by name)

**Contrastive vision-language pretraining**
- **CLIP** (Radford et al., 2021, OpenAI) — dual encoder, InfoNCE contrastive loss over image–text pairs; the canonical joint embedding space. Already wired (probe-only) in `encoder_probe.py` (`clip:<id>`).
- **SigLIP** (Zhai et al., 2023, Google) — replaces softmax/InfoNCE with a **pairwise sigmoid** loss; trains stably at smaller batch, better zero-shot at a given compute. SigLIP / SigLIP2 are the de-facto encoder choice for modern open VLMs (used by PaliGemma, Idefics2/3). Already wired (probe-only) as `siglip:<id>`.
- **EVA-CLIP**, **DFN**, **MetaCLIP** — stronger CLIP-family encoders, candidate swap-ins for the encoder ablation.

**Vision encoders (ViT)**
- **ViT** (Dosovitskiy et al., 2021) — patchify image → token sequence → transformer. The encoder whose *patch grid* (e.g. 24×24 for 336px/14-patch) becomes the visual token sequence the projector maps into LLM space. AnyRes / image-tiling (LLaVA-1.6/NeXT) handles higher resolution by tiling.

**Connector / projector + LLM (the route this plan takes)**
- **LLaVA** (Liu et al., 2023) — frozen CLIP ViT → **linear/MLP projector** → LLM (Vicuna). Two-stage: (1) projector-only pretraining on caption pairs (align visual tokens to word space), (2) end-to-end visual instruction tuning. **LLaVA-1.5** uses a 2-layer MLP projector + academic-task VQA mixtures; the MLP bump is most of the gain.
- **LLaVA-1.6 / LLaVA-NeXT** — AnyRes tiling for high-res, OCR/chart gains.
- **MiniGPT-4**, **ShareGPT4V** — projector + curated high-quality caption data; data quality > data quantity for the connector.
- **BLIP-2** (Li et al., 2023) — **Q-Former**: a small set of learned query tokens cross-attend the frozen ViT to produce a fixed-length visual prefix; more parameter-efficient connector than a per-patch MLP.
- **Honeybee** (Cha et al., 2023) — locality-preserving C-Abstractor connector; connector design *matters*.

**Cross-attention / interleaved**
- **Flamingo** (Alayrac et al., 2022, DeepMind) — frozen LLM + frozen vision encoder + **gated cross-attention** layers (Perceiver Resampler → cross-attn into LM blocks); supports interleaved image-text and few-shot. **Idefics / Idefics2-3** (HF) are the open reproductions.

**Native / early-fusion multimodal**
- **Fuyu** (Adept) — no separate encoder; image patches linearly projected straight into the decoder (architecturally simplest, harder to train).
- **Chameleon** (Meta, 2024) — early-fusion token-based mixed-modal; **Gemini** (DeepMind) and **GPT-4o** — natively multimodal. Out of scope for this repo (frontier-scale); named as the end of the spectrum.

**Visual instruction tuning & data**
- **LLaVA-Instruct-150K** — GPT-4-generated multimodal instructions (the seed of visual instruction tuning). **The Cauldron** (HF, Idefics2) — 50-dataset unified VQA/OCR/chart/doc instruction mixture. **LLaVA-1.5 mixture**, **ShareGPT4V** captions, **DocVQA/ChartQA/TextVQA/AI2D** for OCR/chart/doc skills.

**Computer-use / GUI grounding**
- **Set-of-Marks prompting (SoM)** (Yang et al., 2023, Microsoft) — overlay numbered marks on segmented regions so the VLM references a *mark id* instead of raw pixels; the single biggest grounding lever.
- **SeeClick** (Cheng et al., 2024) — GUI grounding pretraining (predict click coordinates from instruction+screenshot).
- **ScreenSpot** — the standard GUI-grounding benchmark (click accuracy on real UIs).
- **Ferret-UI** (Apple), **CogAgent** (THUDM), **UGround / OS-Atlas**, **WebArena / VisualWebArena**, **OSWorld**, **Mind2Web** — GUI-agent grounding models and task-success benchmarks.
- **Anthropic Computer Use** (computer-use tool: screenshot → predict pixel coordinates for click/type) — the product this repo's `gui_agent.py` is positioned against.

**Multimodal RLHF / visual reward & hallucination**
- **RLHF-V / RLAIF-V** (Yu et al., 2024) — dense, segment-level human/AI feedback to reduce VLM hallucination. **POVID**, **HA-DPO**, **mDPO** — preference optimization against hallucinated vs grounded responses (the OPD pattern the repo already gestures at in `visual_reward.py`).
- **Hallucination / grounding evals:** **POPE** (Li et al., 2023 — polling object presence; the canonical object-hallucination probe), **CHAIR** (caption object hallucination rate), **MMHal-Bench**, **HallusionBench**, **MME**, **MMBench**, **MMMU**, **SEED-Bench**, **MathVista**. **RefCOCO/RefCOCO+/RefCOCOg** for referring-expression grounding (box IoU).

### 1.3 Where the repo's moat plugs in

The repo's distinctive asset is **calibrated, verifier-grounded abstention** (`calibration.py`, `judge.py`, `verifiers.py`, `visual_reward.py` with the `correct > abstain > wrong` ordering). No major open VLM ships a calibrated-abstention story for VQA or a fail-closed grounding gate for clicks. So the credible, *non-leaderboard* contribution is not "another VLM" — it is **"a LLaVA-style VLM that knows when it can't see the answer, with a verifier-gated grounding eval and a fail-closed computer-use loop."** That is the POPE/grounding axis crossed with the repo's risk-coverage discipline.

---

## 2. Current repo state (file-level, honest)

`multimodal_bench/` is an **evaluation + GUI-action-gating harness, not a model**. There is no trainable multimodal parameter anywhere in it. Honest file-by-file:

| File | What it actually is | Honest limitation |
|---|---|---|
| `model.py` | "Answer functions" = deterministic **mocks** (`grounded`/`credulous`/`abstainer`) + an opt-in OpenAI-vision API caller. | No model weights. The "model" is a stub or a remote API; nothing is trained. |
| `encoder_probe.py` | Image↔text **retrieval probe** with bootstrap CI. Default `hashing` backend embeds the *structured caption string*, not pixels (self-labeled). `clip:`/`siglip:` paths load real HF weights but are **not exercised in CI** (recorded as blockers if deps/weights absent). | Default measures harness plumbing, not perception. Real-encoder path is built but unrun; no projector, no LLM connection. |
| `gui_agent.py` | Fail-closed **action verifier**: given a screenshot *scene spec* + a proposed `{click, target, at:[x,y]}`, allows only if the named element exists AND the coordinate lands on it; else withholds + escalates. `DEMO_SCREEN` + grounded/hallucinated action sets. | The agent does **not produce** actions from pixels — it only *gates* externally-supplied actions against a ground-truth element list. No VLM, no screen parsing, no real screenshots. |
| `visual_reward.py` | Calibration-aware **RLVR reward** (`correct +1 / abstain -0.25 / wrong -1`), TRL-`GRPOTrainer`-compatible `make_grpo_reward`. Fail-closed via judge-free verifier. | Reward surface exists; **no training run consumes it**. `offline_invariants()` only asserts the reward math. |
| `visual_dataset.py` | Family-disjoint train/eval split over trap categories (contamination-free). | Tiny (rows = trap suite); prompts only, no images for training. |
| `judge.py` | Lexical judge (abstain/yes-no/count/text) + multi-judge consensus + Cohen's κ. | Text-only screen of free-text answers; not visual. |
| `calibration.py` | Risk-coverage / ECE / AURC / selective-risk for VQA, reusing `agent.calibration`. Synthetic calibrated-vs-overconfident demo. | Driven by synthetic confidence fns, not a real model. |
| `render.py` | Scene spec → PNG (Pillow), for the real-VLM path only. | Toy rasteriser (rectangles + drawn strings); not photographs. |
| `runner.py` | Runs the trap suite, `aggregate_runs` pools N runs → bootstrap CIs + **validated-headline checklist** (real model, ≥2 judge families, κ≥0.40, ≥3 runs, CI computed). | Excellent eval discipline — but it evaluates *answer functions*, which today are mocks/APIs. |
| `synthesize.py` | Deterministic, **verifier-checked** synthesis of chart/table/document traps (distractors are plausible misreads). | Structured scenes, not natural images; a clean source of *eval* data, not *training* images. |
| `verifiers.py` | Judge-free deterministic verifiers (presence/count/relation/ocr/chart/table/doc) over scene specs; `gold_matches_check` integrity invariant. | The crown jewel — judge-free ground truth. Operates on scene specs, not pixels. |
| `data/visual_traps*.json` | ~Hand-authored + synthetic trap rows; categories: phantom_object, miscount, spatial_relation, fabricated_ocr, chart/table/doc_qa. | Small (eval-scale). Scenes are structured specs. |

**Adjacent infra that this plan reuses (real, on disk):**
- `tools/train_lora.py` — manual SFT + PEFT/LoRA loop, completion-only masking, holdout/early-stop; `DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"`. **This is the repo's de-facto local LLM** and the LoRA loop to extend.
- `tools/run_rlvr.py` — GRPO (TRL) verifier-reward training (math/code); `requirements-rl.txt` has trl/peft/vllm.
- `tools/run_visual_rlvr.py`, `tools/run_multimodal_reward.py` — visual RLVR/reward wiring, **offline assertions only** (no live VLM-GRPO bundled).
- `agent/model.py` — `PRESETS` backend abstraction (anthropic/openai/vllm/ollama/mock); how the repo wires model backends.
- `tools/runpod_train.py`, `tools/runpod_rlvr.py` — RunPod pod lifecycle (create → SSH → stream → copy artifacts → delete), GPU fallback (4090 24GB → A100 80GB), image `runpod/pytorch:...torch291-cu1281`.
- `pretraining/nano/model.py` — a *toy* from-scratch softmax LM (vocab=16); **not** a viable VLM backbone — ignore for multimodal, named only to be honest that the only "from scratch" model is a research toy.
- `requirements-lora.txt` (torch≥2.7, transformers≥4.44, peft, bitsandbytes), `requirements-rl.txt` (trl≥0.16, vllm≥0.6).
- `agent/calibration.py`, `provenance_bench/` (consensus, family splits, contamination) — the trust layer being ported.

**Bottom line:** today the repo can *score* a VLM and *gate* its clicks honestly; it cannot *produce* a visual answer or a click from an image. The plan adds exactly that, smallest credible slice first.

---

## 3. Top-tier target end-state

A reproducible, single-GPU-trainable **`multimodal_train/` package** alongside `multimodal_bench/`, delivering:

1. **A trained LLaVA-style VLM** — frozen SigLIP/CLIP ViT + trained MLP projector + Qwen2.5-3B-Instruct (projector-only stage 1, then LoRA-on-LLM + projector stage 2 visual instruction tuning). Checkpoints + a `vlm:<ckpt>` answer-fn backend wired into `multimodal_bench.model.resolve_answer_fn` so the *existing* eval harness scores the *trained* model end-to-end.
2. **A real visual-grounding / abstention eval** — POPE-style object-hallucination + RefCOCO-style box-IoU grounding, scored by the repo's judge-free verifiers and reported with multi-judge consensus + CIs + risk-coverage curves. The headline: *grounding-conditioned abstention lowers hallucination at a measured coverage cost* (the provenance moat, in pixels).
3. **A grounded computer-use loop** — `gui_agent.py` upgraded from "gate externally-supplied actions" to "**parse screen → predict a grounded click → gate it fail-closed**," using Set-of-Marks prompting + the trained VLM, evaluated on a ScreenSpot-style click-accuracy + task-success harness, with the existing withhold/escalate gate intact.

Each ships behind the no-overclaim gate; nothing is published until a real CUDA run lands (failure-ledger "Open" until then).

---

## 4. Phased plan (milestones, files, libraries)

### Milestone 0 — make the real-encoder path *live* (1–2 days, CPU/1×GPU)
*De-risk the dependency surface before any training.*
- Add `requirements-multimodal.txt`: `torch`, `transformers`, `pillow`, `torchvision`, `accelerate`, `peft`, `bitsandbytes`, `datasets`, `sentencepiece`, `einops`.
- Exercise `encoder_probe.py`'s `siglip:google/siglip-base-patch16-224` and `clip:openai/clip-vit-base-patch32` paths on real weights (download via RunPod or local GPU); record recall@1 + CI **and** the "isRealEncoder" flag. Convert today's blockers into real numbers.
- New: `multimodal_train/encoders.py` — load a frozen ViT encoder, expose `encode_image(pil) -> [num_patches, d_vision]` (penultimate patch grid, not pooled CLS), the visual-token source for the projector.

### Milestone 1 (START) — train a LLaVA-style projector + visual instruction tuning
**Goal:** connect a frozen ViT to the local LLM and train it to answer about images.

- **Architecture** (`multimodal_train/model.py`):
  - Frozen vision encoder: SigLIP-SO400M or CLIP-ViT-L/14-336 (patch grid → `[N, d_v]`).
  - **Projector** (the trained part, stage 1): 2-layer MLP `d_v → d_llm` (LLaVA-1.5), GELU. Optional Q-Former variant (`multimodal_train/connectors.py`) for a BLIP-2 ablation (fixed 32 query tokens).
  - LLM: `Qwen/Qwen2.5-3B-Instruct` (the repo's `train_lora.py` default). Visual tokens spliced in at an `<image>` placeholder in the chat template; attention over `[visual_tokens; text_tokens]`.
- **Stage 1 — projector pretraining (alignment):** freeze ViT + LLM, train **only** the projector on caption pairs (LLaVA-558K / CC3M-595K subset, or ShareGPT4V captions). Objective: next-token LM loss on the caption given the image. ~1 epoch.
- **Stage 2 — visual instruction tuning:** unfreeze projector + **LoRA on the LLM** (reuse `tools/train_lora.py` LoRA config, target attn+MLP), keep ViT frozen. Train on LLaVA-Instruct-150K + a VQA/OCR/chart slice of **The Cauldron** (DocVQA, ChartQA, TextVQA, AI2D) + **the repo's own verifier-checked synthetic traps** (`synthesize.py`) as a grounding/abstention-flavored slice.
- **Files:** `multimodal_train/model.py` (VLM module), `multimodal_train/data.py` (image-text → tokenized batches with `<image>` splicing, completion-only masking borrowed from `train_lora.py`), `multimodal_train/train_projector.py` (stage 1), `multimodal_train/train_vit.py` (stage 2, wraps the LoRA loop), `multimodal_train/checkpoint.py`.
- **Eval wiring (closes the loop with the existing harness):** add `vlm:<ckpt>` to `multimodal_bench.model.resolve_answer_fn` so `runner.run_cases` scores the *trained* VLM on the existing trap suite with the *same* judge/consensus/CI machinery. **This is the moment the bench stops scoring mocks and starts scoring a model you trained.**
- **Libraries:** transformers, peft, accelerate, bitsandbytes (4-bit LLM), torchvision/PIL, datasets, trl (later). RunPod via `tools/runpod_train.py` (extend for the two-stage schedule).

### Milestone 2 — visual-grounding / abstention eval tying into provenance
**Goal:** the repo's moat, in pixels — *grounding-conditioned abstention*.
- **POPE-style object-hallucination eval** (`multimodal_bench/pope_eval.py`): "Is there a {object}?" over random/popular/adversarial negatives; verifier = judge-free presence check (already in `verifiers.present`). Report hallucination rate + CI by sampling regime.
- **Referring-grounding eval** (`multimodal_bench/grounding_eval.py`): model must return a **box/region** supporting its answer; verifier checks IoU against ground-truth box (RefCOCO-style; reuse `verifiers.point_in_box`/`element_at`). This is the literal multimodal analog of citation-grounded abstention: *an answer must point at the evidence region, and the gate re-checks the crop.* Ties straight into the repo's provenance theme.
- **Grounded abstention surface:** extend `calibration.py` so confidence is *derived from grounding strength* (IoU / region-presence margin), not a free scalar. Headline: **risk-coverage curve where abstaining-on-weak-grounding lowers hallucination** — measured, with selective-risk < base-risk at <100% coverage.
- **Reporting:** all through `runner.aggregate_runs` (multi-judge consensus, κ≥0.40, ≥3 runs, CIs). Add a `RESULTS.md`-style entry only after a real run clears the gate.
- **Files:** `multimodal_bench/pope_eval.py`, `multimodal_bench/grounding_eval.py`, extend `calibration.py`, extend `verifiers.py` (IoU helper). **Libraries:** none new (reuses existing harness + Milestone-1 VLM backend).

### Milestone 3 — strengthen the computer-use / GUI agent with grounded clicks
**Goal:** turn `gui_agent.py` from action-*gate* into a *grounded-click producer + gate* (maps directly to Anthropic Computer Use).
- **Screen parsing + Set-of-Marks** (`multimodal_bench/screen_parse.py`): from a screenshot, produce candidate UI elements (boxes+labels) and overlay numbered marks (SoM); the VLM picks a mark id → resolves to a box center → that becomes the proposed `{click, target, at}`. SoM converts "guess raw pixels" into "pick a labeled region," the biggest grounding lever.
- **Grounded-click policy** (`multimodal_bench/gui_policy.py`): VLM(screenshot+SoM+instruction) → action; **then** the *existing* `gui_agent.verify_action` fail-closed gate re-checks element-present + coordinate-on-target before dispatch. Hallucinated/phantom clicks are withheld + escalated (unchanged gate, now fed by a real model).
- **Eval** (`multimodal_bench/screenspot_eval.py`): ScreenSpot-style **click accuracy** (does `at` land in the gold element box?) + a small **task-success** harness on scripted multi-step flows (login form, etc., generalizing `DEMO_SCREEN`). Report success ± CI, plus the withhold/escalation rate (the safety axis).
- **Optional GUI grounding fine-tune:** a SeeClick-style LoRA pass (predict click coords from instruction+screenshot) on the Milestone-1 VLM, trained on synthetic UI scenes from `render.py` (extended with UI widgets) — verifier-labeled, so it stays in the repo's "verifier-checked synthesis" lane.
- **Files:** `multimodal_bench/screen_parse.py`, `gui_policy.py`, `screenspot_eval.py`; extend `gui_agent.py` (compose the producer with the gate), `render.py` (UI widgets), `synthesize.py` (UI-scene synth). **Libraries:** PIL (mark overlay), the Milestone-1 VLM backend; optional OCR (`rapidocr`/`tesseract`) for real screenshots.

### Milestone 4 (stretch) — multimodal RLVR / preference optimization
- Wire `visual_reward.make_grpo_reward` into a **live** TRL `GRPOTrainer` over the VLM (mirrors `run_rlvr.py`); reward = verifier-correct AND grounded AND abstains-when-uncertain.
- **OPD/DPO** preference pairs from verifier disagreement (chosen = grounded+calibrated; rejected = confident hallucination the verifier caught) — RLHF-V/mDPO pattern.
- GPU on RunPod; pre-registered "Open" until a CUDA run lands.

---

## 5. Compute / budget tiers

| Tier | Hardware | Scope | Est. wall-clock | Est. cost (RunPod spot) |
|---|---|---|---|---|
| **T0 smoke** | 1× RTX 4090 24GB (or A10) | M0 encoder probe + M1 stage-1 projector on a 50–100K caption subset; overfit a tiny instruction set to prove the pipe trains | 6–12 h | ~$5–15 |
| **T1 credible single-GPU** | 1× A100 80GB | M1 full: stage-1 (558K captions, ~1 ep) + stage-2 (LLaVA-150K + Cauldron VQA/OCR slice, LoRA on Qwen-3B, 4-bit) | 2–4 days | ~$80–250 |
| **T2 eval + computer-use** | 1× A100 80GB (inference-heavy) | M2 POPE/grounding eval (≥3 seeds for CIs) + M3 ScreenSpot eval + optional SeeClick LoRA | 1–2 days | ~$40–120 |
| **T3 RLVR/DPO (stretch)** | 1–2× A100 80GB | M4 GRPO/DPO over the VLM | 2–5 days | ~$150–500 |

Budget the **whole credible portfolio (T0–T2) at roughly $150–400 and ~1 week of GPU**. T1 alone (one trained VLM + harness wiring) is the minimum to claim "trained a multimodal model." Use `tools/runpod_train.py` GPU-fallback (4090 → A100) and spot pricing; estimate ETA before launch (existing ETA estimator).

---

## 6. Honest metrics (with CIs, via the existing gate)

All reported through `runner.aggregate_runs`: ≥3 runs, multi-judge consensus (≥2 provider families), Cohen's κ ≥ 0.40, bootstrap 95% CIs, validated-headline checklist. No single-judge or single-seed headline.

| Axis | Metric | Verifier (judge-free) | Honest target framing |
|---|---|---|---|
| VQA capability | trap grounding rate, accuracy on TextVQA/ChartQA/DocVQA slice | `verifiers.resolve_check`, exact/tolerance match | report as "trained-model vs mock baselines vs an open VLM API," with CIs; **not** vs leaderboard SOTA |
| Hallucination | POPE accuracy/F1, CHAIR (caption obj-hallucination), trap hallucination rate | `verifiers.present`/count | rate ± CI by sampling regime (random/popular/adversarial) |
| Grounding | RefCOCO-style box IoU@0.5, region-presence | `point_in_box`/IoU helper | grounded-answer rate ± CI |
| Calibration/abstention | ECE, AURC, selective-risk @ {0.5,0.8,1.0} coverage, risk-coverage curve | `agent.calibration` via `multimodal_bench/calibration.py` | **headline:** selective-risk < base-risk at <100% coverage (the moat) |
| Computer-use | ScreenSpot click accuracy; multi-step task success; withhold/escalation rate | `verifiers.element_at`/`point_hits_label` | success ± CI; report *both* success and safe-withhold rate |
| Encoder choice | image↔text recall@1 | `encoder_probe.retrieval_probe` | CI + isRealEncoder flag; ablation, not a leaderboard |

**Baselines that make numbers meaningful:** the existing `mock:grounded` (upper bound, hallucination≈0), `mock:credulous` (lower bound, hallucination≈1), `mock:abstainer`, **and** an off-the-shelf open VLM via the OpenAI-vision backend (e.g. a hosted LLaVA/Qwen-VL) — so the trained model is positioned against honest anchors, not a vacuum.

---

## 7. Risks & overclaim guards

- **"Trained a VLM" overclaim.** A LoRA-tuned projector on a frozen encoder + frozen-ish LLM is *connector/adapter training*, not from-scratch native multimodal pretraining. **Guard:** state the architecture precisely everywhere ("frozen SigLIP + trained MLP projector + LoRA Qwen-3B"); never imply pretraining-from-scratch. Keep the nano-LM (`pretraining/nano`) clearly labeled as a toy.
- **Synthetic-eval ≠ real-perception.** `render.py` scenes are rectangles, not photographs; `encoder_probe`'s default `hashing` backend measures captions, not pixels. **Guard:** keep the existing self-labeling (`isRealEncoder`, `perceptionNote`); add natural-image evals (POPE on COCO, RefCOCO) before any perception claim — synthetic traps test grounding *logic*, real images test *perception*; report them separately.
- **Data contamination / leakage.** LLaVA/Cauldron overlap with eval sets. **Guard:** extend the family-disjoint split (`visual_dataset.py`) and perceptual-hash dedup (`pipeline/multimodal/phash.py`) across train/eval image sets; entity-disjoint where possible.
- **Reward hacking in M4.** A model can learn to over-abstain to dodge the −1 wrong penalty. **Guard:** the `correct > abstain > wrong` ordering is *bounded* and abstention is *costed* (−0.25); report coverage alongside risk so over-abstention is visible.
- **Judge circularity.** **Guard:** keep `judge.py`/`verifiers.py` code-disjoint (already true); the verifier never sees the judge's logic.
- **GPU run never lands.** **Guard:** failure-ledger "Open" until a real CUDA run produces gated numbers; no projected/extrapolated metrics in RESULTS.md.
- **Computer-use safety.** A grounded click is still a *hypothesis*. **Guard:** the existing fail-closed `verify_action` gate + HITL/BLP gates stay in front of every mutating action; report withhold/escalation rate as a first-class metric, not a footnote.

---

## 8. Effort

| Milestone | Eng effort | GPU | Gating dependency |
|---|---|---|---|
| M0 encoder live | 1–2 days | T0 | deps install |
| **M1 LLaVA projector + VIT (START)** | **1.5–2.5 wks** | T0→T1 | data download, two-stage loop, eval wiring |
| M2 grounding/abstention eval | 1–1.5 wks | T2 | M1 VLM backend; POPE/RefCOCO data |
| M3 grounded computer-use | 1.5–2 wks | T2 | M1 VLM; SoM + screen parse |
| M4 RLVR/DPO (stretch) | 2–3 wks | T3 | M1–M2; live GRPO |

**Critical path to the single highest-signal deliverable** (a trained VLM scored by the existing honest harness): **M0 + M1 ≈ 2–3 weeks of eng + ~$80–250 GPU.** Everything after that is differentiation (the grounding/abstention moat and computer-use), each independently shippable.

---

## Appendix — file manifest (new vs reused)

**New (`multimodal_train/`):** `encoders.py`, `model.py`, `connectors.py`, `data.py`, `train_projector.py`, `train_vit.py`, `checkpoint.py`, `requirements-multimodal.txt`.
**New (`multimodal_bench/`):** `pope_eval.py`, `grounding_eval.py`, `screen_parse.py`, `gui_policy.py`, `screenspot_eval.py`.
**Extended:** `multimodal_bench/model.py` (`vlm:<ckpt>` backend), `calibration.py` (grounding-derived confidence), `verifiers.py` (IoU), `gui_agent.py` (producer+gate), `render.py`/`synthesize.py` (UI scenes), `tools/runpod_train.py` (two-stage schedule), `tools/run_visual_rlvr.py` (live GRPO, M4).
**Reused as-is:** `verifiers.py`, `judge.py`, `runner.py` (aggregate/validated-checks), `agent/calibration.py`, `tools/train_lora.py` (LoRA config), `agent/model.py` (backend pattern), `pipeline/multimodal/phash.py` (dedup).
