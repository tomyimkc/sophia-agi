# 03 — Mechanistic Interpretability for Sophia-AGI

**Sparse dictionary features over the local model's residual stream, in service of honesty.**

A research-grade plan to add a *credible* mechanistic-interpretability (mech-interp)
capability to `sophia-agi`, at the bar of Anthropic's Interpretability team
(Research Scientist/Engineer). The unifying thesis: the repo already *behaves*
honestly (provenance gates, abstention, fail-closed steering) — interpretability
asks **what computation inside the weights produces or violates that honesty**, and
gives us *internal* (not just behavioral) instruments for deception, hallucination,
and abstention. The narrative coherence is the asset: every milestone lands on a
feature the repo already cares about.

> **Standing honesty constraint (inherited from the repo ethos).** Every claim in
> this workstream is *falsifiable and fail-closed*. A null result ("no
> interpretable honesty feature found at L_k, K=16k, after N hours") is a
> publishable, accepted outcome — exactly as `SSA = 0/N` is in the steering
> experiment (`docs/09-Agent/Steering-Experiment.md`). We do **not** ship the word
> "circuit" or "the deception feature" without the causal evidence defined in §7.

---

## 1. Thesis & references

### 1.1 The scientific problem

A 7B transformer represents far more concepts than it has neurons. The
**superposition hypothesis** (Elhage et al., *Toy Models of Superposition*,
Anthropic 2022) holds that features are stored as **near-orthogonal directions in
activation space**, more numerous than dimensions, so individual neurons are
**polysemantic** (fire for unrelated concepts). Direct neuron inspection therefore
fails. The program below recovers an *overcomplete, sparse, more-monosemantic
basis* and then does **causal** work in that basis.

### 1.2 Method families (cited by name)

**Sparse autoencoders (SAEs) / dictionary learning.**
- Bricken et al., *Towards Monosemanticity: Decomposing Language Models with
  Dictionary Learning* (Anthropic, 2023) — the founding SAE-on-residual-stream
  recipe: a wide ReLU autoencoder with an L1 sparsity penalty learns
  monosemantic features; introduces feature interpretability + activation
  dashboards, dead-feature and feature-density diagnostics.
- Templeton et al., *Scaling Monosemanticity* (Anthropic, 2024) — scales SAEs to
  Claude 3 Sonnet; finds abstract, multilingual, multimodal features including
  safety-relevant ones (deception, sycophancy, bias, dangerous content); shows
  **feature steering** (clamping a feature up/down changes behavior) and the
  "Golden Gate Bridge" clamp as the canonical causal demo.
- **Architecture variants** to de-risk training: **Gated SAEs** (Rajamanoharan
  et al., DeepMind 2024) and **TopK / JumpReLU SAEs** (Gao et al., *Scaling and
  Evaluating Sparse Autoencoders*, OpenAI 2024; Rajamanoharan et al., JumpReLU,
  DeepMind 2024) — fix the L1 *shrinkage* bias and give a directly-controlled L0.
  **Anthropic's "April Update" / crosscoders** (Lindsey et al., 2024) and
  **transcoders** (below) are the current frontier.

**Transcoders.** Predict an MLP's *output* from its *input* through a sparse
hidden layer (Jacob/Dunefsky et al., *Transcoders Find Interpretable LLM Feature
Circuits*, 2024; Anthropic's *attribution-graph* / "circuit tracing" line, 2025).
Transcoders replace a dense MLP with an interpretable sparse map and make
*input-independent* circuit analysis tractable — the basis for attribution graphs.

**Causal localization.**
- **Activation patching / causal tracing**: Meng et al., *Locating and Editing
  Factual Associations in GPT* (**ROME**, 2022) — corrupt-then-restore tracing
  localizes factual recall to mid-layer MLPs; **causal mediation analysis** (Vig
  et al., 2020). **Path patching / attribution patching** (Wang et al.,
  *IOI circuit*, 2022; Nanda, attribution patching, 2023) scales patching to many
  components cheaply.
- **Logit lens** (nostalgebraist, 2020) and **tuned lens** (Belrose et al., 2023)
  — decode intermediate residual streams through the unembedding to watch a
  prediction form layer-by-layer.

**Attention & known circuits.**
- **Induction heads**: Elhage et al., *A Mathematical Framework for Transformer
  Circuits* (Anthropic, 2021) and Olsson et al., *In-context Learning and
  Induction Heads* (Anthropic, 2022) — `[A][B]…[A]→[B]` copy heads; the
  best-understood real circuit and a perfect *validation target* for our tooling.

**Probing classifiers.** Alain & Bengio (2016) linear probes; **honesty/truth
directions** — Burns et al., *Discovering Latent Knowledge* (CCS, 2022);
Marks & Tegmark, *The Geometry of Truth* (2023); Azaria & Mitchell, *The
Internal State of an LLM Knows When It's Lying* (2023); Zou et al.,
*Representation Engineering* (RepE, 2023). **Caveat we will honor:** a probe shows
*correlation/decodability*, not that the model *uses* the direction — only
intervention does (Belinkov, *Probing Classifiers: Promises & Pitfalls*, 2022).

**Steering / feature clamping.** Turner et al., *Activation Addition* (ActAdd,
2023); Rimsky et al., *Contrastive Activation Addition* (CAA, 2023) — already the
math in `agent/steering/vectors.py` (`diff_of_means` = CAA Eq. 1). SAE feature
clamping (Scaling Monosemanticity) is the higher-precision successor: steer a
*single interpretable feature*, not a diffuse contrast vector.

**SAE evaluation metrics.** L0 (mean active features/token), reconstruction loss
(normalized MSE / fraction of variance explained), **cross-entropy loss recovered**
(substitute reconstruction into the forward pass; how much CE is preserved vs the
ablation floor), **dead-feature %**, feature-density histograms, and
**feature interpretability** (auto-interp: Bills et al., OpenAI 2023 — an LLM names
a feature from top activations, a second LLM predicts activations from the name;
score = correlation). Anthropic/EleutherAI **SAEBench** (2024) standardizes these.

### 1.3 The honesty tie-in (the repo's narrative)

The repo's themes map directly onto interpretability targets:
- **Hallucination / confabulation** ↔ entity-recognition & "I-know-this-entity"
  features (Scaling Monosemanticity describes "known vs unknown entity" features
  that gate refusal/confabulation — the mechanistic root of hallucination).
- **Deception / sandbagging / verifier-tampering** ↔ the labelled directions in
  `eval/deception/deception_v1.jsonl` (`tamper`, `launder`, `sandbag`).
- **Abstention / "fail-closed"** ↔ uncertainty/refusal features; the repo already
  *rewards* abstention (`safe_uncertain` label) — we look for its internal cause.
- **Faithfulness** ↔ `benchmark/agent_faithfulness.json` grounded-vs-unfaithful
  trajectories.

This is the credible, non-overclaiming framing: **we are not "solving deception."
We are building instruments to measure and causally test honesty-relevant
representations in a 7B model we control end-to-end.**

---

## 2. Current repo state (honest)

**There is essentially no mechanistic interpretability today.** Verified by
reading the tree and grepping `sae|activation|circuit|probe|steering|
monosemantic|logit lens|induction|transcoder`:

- The `*_probe` hits are **unrelated**: `agent/continual_qa_answer.py`,
  `selfextend/verifier_synthesis.py`, `tests/test_continual_qa_answer.py` use
  "probe" to mean a *QA query*, not a probing classifier. No SAE, no dictionary,
  no patching, no lens, no auto-interp anywhere.
- **What *does* exist — and is genuinely useful scaffolding:**
  - `agent/steering/` — a clean, CI-tested **residual-stream hook layer**:
    `hooks.py` (`make_steering_hook`, `attach_hooks`, `capture_residual` via
    `register_forward_hook` on `model.model.layers[L]`, handles transformers 4.x
    tuple / 5.x bare-tensor outputs, fp32-master→fp16 MPS-safe casts);
    `vectors.py` (pure-stdlib `diff_of_means`/`normalize`/`cosine` = CAA);
    `compose.py` (orthogonalization), `stats.py`, `anti_gaming.py`,
    `pif_harness.py`. **This is exactly the capture/intervention primitive an SAE
    pipeline needs** — reuse it, do not rebuild it.
  - `tools/run_steering.py` — model loader with a Phi-3.5-mini fallback chain;
    offline `--model mock --dry-run` path; **template for our offline-first,
    CI-green pattern.**
  - Honesty datasets ready to mine: `eval/deception/deception_v1.jsonl`,
    `eval/conscience/honeypots.v1.json`, `benchmark/agent_faithfulness.json`,
    `eval/continual_qa`, `eval/fact_check`, `eval/belief_revision`.
- **Models** (`models/manifest.json`, `training/local_sophia_7b/manifest.json`,
  `models/ollama/Modelfile`): base **Qwen/Qwen2.5-7B-Instruct** (the
  `local_sophia_7b` LoRA target) and **Qwen/Qwen2.5-3B-Instruct** (shipped LoRA,
  Ollama); steering uses **Phi-3.5-mini**. Qwen2.5 is a standard
  Llama-style dense decoder — well-supported by TransformerLens/SAELens.
- **Compute:** RunPod via MCP (`.mcp.json` → `@runpod/mcp-server`). Confirmed
  available: H200/H100-80GB/A100-80GB (high/medium stock), RTX 4090/5090-32GB,
  L40S/A6000-48GB. Plenty for 7B-residual SAEs.
- **Ethos to preserve:** offline/mock CI path, deterministic pure-stdlib cores,
  pre-registered falsifiable claims, fail-closed verdicts, "NOT AGI / claim
  boundary" honesty in every manifest (`CONTRACT.md`, `agi-proof/`).

**Gap to close:** activation harvesting at scale, a real SAE trainer + checkpoints,
a feature dictionary + dashboards + auto-interp, causal patching, and SAE-feature
steering — none of which exist yet.

---

## 3. Top-tier target end-state

A self-contained `interp/` package + `tools/` entrypoints that deliver:

1. **Activation harvesting** at any `(layer, hook-point)` of Qwen2.5-7B over a
   corpus, sharded to disk — built on the existing hook layer.
2. **A trained residual-stream SAE** (TopK or JumpReLU, dict size 16k–65k) on a
   mid-layer of Qwen2.5-7B, with **honest eval metrics** (L0, FVU/CE-recovered,
   dead-%, density histogram) checked against published baselines.
3. **A feature dictionary**: per-feature top-activating exemplars, logit-lens
   projections (top promoted/suppressed tokens), density, and **auto-interp
   labels + auto-interp scores** with confidence intervals — browsable dashboards.
4. **A steering demo on an honesty/deception feature**: identify candidate
   honesty/deception/abstention features (by labelled-probe correlation + auto-interp),
   then **clamp** one and show a *pre-registered, CI-bounded* behavioral change
   on held-out deception/faithfulness prompts — graduating CAA's diffuse vector to
   a single interpretable feature.
5. **Activation patching on a hallucination case**: a corrupt/clean
   causal-tracing harness; localize where a confabulated vs grounded answer
   diverges (which layers/heads/MLPs, and which SAE features mediate it), reported
   as effect sizes with nulls, **never as "the hallucination circuit."**
6. **Validation harness** on a *known* circuit (induction heads) so reviewers
   trust the tooling before trusting the honesty claims.
7. **RunPod-native, offline-CI-green, fully reproducible** (seeds, manifests,
   public reports under `agi-proof/interp/`), matching repo provenance discipline.

---

## 4. Phased plan — milestones

Conventions: new top-level package `interp/`; CLI under `tools/`; reports under
`agi-proof/interp/`; configs under `configs/interp/`. Mirror the steering pattern:
**pure-stdlib/offline core is CI-tested; the GPU path is skip-guarded and runs on
RunPod.** Reuse `agent/steering/hooks.py` rather than reimplementing capture.

### Libraries

- **TransformerLens** (`transformer_lens`) — `HookedTransformer` loads Qwen2.5,
  exposes named hook points, `run_with_cache`, ablation/patching utilities. The
  spine for patching, logit lens, induction-head analysis.
- **SAELens** (`sae_lens`) — `SAETrainingRunner` / `ActivationsStore` /
  `SAEConfig`; standard TopK/Gated/JumpReLU SAEs, checkpointing, eval, and a
  loadable `SAE` for inference/steering. Primary SAE path.
- **nnsight** (`nnsight`) — remote/traced access to arbitrary internals on the
  HF model; backstop where TransformerLens lacks a Qwen hook or for large-model
  remote execution. Use for cross-validating patching results.
- **SAEBench / sae_dashboard** — standardized eval + feature dashboards.
- Plus existing stack: `torch`, `transformers>=4.46`, `accelerate`, `datasets`,
  `safetensors`, `einops`. Add `requirements-interp.txt` (kept out of CI default,
  like `requirements-steering.txt`).
- **From-scratch fallback:** a ~150-line `interp/sae/model.py` (TopK SAE:
  encoder `W_enc`, decoder `W_dec` with unit-norm columns, tied-bias
  pre-encoder subtract, TopK activation, dead-feature resampling per Bricken
  2023). Guarantees we are not blocked on a library API and that we *understand*
  the objective. SAELens stays the production path.

### Milestone M0 — Harness + offline core (no GPU)  ✦ start here

- `interp/__init__.py`, `interp/hookpoints.py` — thin adapter over
  `agent.steering.hooks` resolving Qwen2.5 hook points (`blocks.{L}.hook_resid_post`
  ↔ `model.model.layers[L]`); maps TransformerLens names ↔ HF module paths.
- `interp/sae/model.py` — from-scratch TopK SAE (forward/encode/decode, unit-norm
  decoder, L0/FVU computed in-module). **Pure-torch, tiny-tensor unit-testable.**
- `interp/sae/metrics.py` — **pure-stdlib/numpy** L0, FVU, CE-recovered,
  dead-feature %, density histogram, with bootstrap CIs (mirror
  `agent/steering/stats.py`).
- `tools/run_interp.py --mode mock --dry-run` — deterministic mock activations
  (seeded), trains a toy SAE for a few steps, emits a report. **CI-green, no
  torch-on-GPU, no downloads** — the `--model mock` analogue.
- `tests/test_interp_sae.py`, `tests/test_interp_metrics.py` — assert SAE
  reconstructs a planted low-rank signal, decoder norms stay ~1, TopK gives exact
  L0=K, metrics match closed form on synthetic data. **Skip-guard the torch path.**
- **Exit:** `pytest -k interp` green offline; reviewer can read the metric defs.

### Milestone M1 — Activation harvesting (small GPU)

- `interp/harvest.py` + `tools/harvest_activations.py` — load Qwen2.5-7B (4-bit ok
  on a 24–48GB GPU; bf16 on A100/H100), stream a corpus
  (`training/corpus.jsonl`, deception/faithfulness sets, a generic slice), capture
  `hook_resid_post` at a chosen mid-layer (start L≈14–18 of 28), shard to
  `safetensors` with a manifest (token counts, seed, dtype, layer, corpus hash).
- Target **~50–200M activation tokens** for a first 16k-dict SAE (scale later).
- **Exit:** sharded activations on a RunPod network volume + manifest; a histogram
  of activation norms sanity-checked; reproducible from manifest.

### Milestone M2 — Train the first SAE + honest metrics

- `tools/train_sae.py` — SAELens `SAETrainingRunner` (primary) **or** our
  from-scratch trainer; TopK SAE, dict 16k (×~2k d_model), on the M1 layer.
- Track during training: L0, FVU/CE-recovered, dead-%, density; resample dead
  features (Bricken 2023). Checkpoints + `agi-proof/interp/sae-{layer}-{run}.json`.
- **Honest acceptance gate (pre-registered, §6):** L0 in a sane band (~20–80),
  CE-recovered ≥ ~0.9 of the substitution-loss budget, dead-% below threshold —
  else report the SAE as **not-yet-usable** and iterate. A bad SAE is a reported
  fact, not a hidden one.
- **Exit:** a checkpointed SAE whose metrics are in-distribution vs published 7B
  residual SAEs, or an honest "did not meet bar" report with next steps.

### Milestone M3 — Feature dictionary + auto-interp

- `interp/dictionary.py` + `tools/build_dictionary.py` — for each live feature:
  top-K activating exemplars (text + token), activation density, **logit-lens**
  projection of `W_dec[:,f]` through the unembedding (top promoted/suppressed
  tokens), and a `sae_dashboard` HTML card.
- `interp/autointerp.py` — auto-interp (Bills 2023): a judge LLM (local Qwen/Ollama
  *and* an external family for cross-validation, matching the repo's multi-judge
  discipline) labels a feature from exemplars; a second pass predicts activations
  from the label; **score = correlation, with CIs and a multi-family κ check.**
- **Exit:** `agi-proof/interp/dictionary-{layer}.json` + dashboards; a reported
  **% of features that are interpretable** (auto-interp score above threshold),
  honestly including the uninterpretable remainder.

### Milestone M4 — Honesty/deception feature → steering demo

- `interp/honesty_features.py` — score every SAE feature for honesty relevance by
  (a) **correlation with labelled directions** from
  `eval/deception/deception_v1.jsonl` (tamper/launder/sandbag/safe_uncertain) and
  `benchmark/agent_faithfulness.json`; (b) auto-interp labels mentioning
  deception/uncertainty/abstention/refusal; (c) logit-lens token signatures.
  Output a *ranked candidate list with effect-size CIs* — **candidates, not "the
  feature."**
- `tools/steer_feature.py` — **clamp** a chosen feature (set its activation to
  ±value via the existing hook layer; SAE-feature steering > diffuse CAA vector),
  generate on held-out deception/faithfulness prompts, score with the repo's
  deterministic + multi-family judges.
- **Pre-registered claim (OPEN until run):** clamping candidate feature `f↑`
  shifts behavior toward {abstention / honest-uncertainty / deception-flag} with a
  paired effect size whose 95% CI excludes 0, **and** is capability-preserving
  (no general-quality regression on a retention set) — mirroring the steering
  experiment's SSA gate. **A null is a legitimate, reported result.**
- **Exit:** `agi-proof/interp/steering-feature.public-report.json` with the
  pre-registered claim marked PASS/NULL and CIs.

### Milestone M5 — Activation patching on a hallucination case

- `interp/patching.py` + `tools/patch_hallucination.py` — TransformerLens
  `run_with_cache` + a **corrupt/clean** pair (a prompt the model confabulates vs
  a grounded variant, drawn from fact_check/continual_qa). Restore activations
  component-by-component (attention out, MLP out, resid by layer) and via
  **attribution patching** for cheap full sweeps; cross-check critical results
  with nnsight. Then **patch SAE features** to ask *which features mediate* the
  confabulation→grounded flip.
- **Exit:** `agi-proof/interp/patching-hallucination.public-report.json` — a
  *localization map* with effect sizes and nulls, framed as "components/features
  whose restoration most reduces the confabulation logit," explicitly **NOT** "the
  hallucination circuit."

### Milestone M6 — Known-circuit validation + writeup

- `tools/validate_induction.py` — reproduce **induction-head** detection
  (per-head prefix-matching/induction scores on random repeated sequences;
  Olsson 2022) on Qwen2.5-7B to prove the patching/attention tooling is correct on
  ground truth *before* anyone leans on the honesty claims.
- `docs/09-Agent/Interpretability-Experiment.md` — the writeup in the repo's
  pre-registration style (falsifiable offline claim + OPEN live claims), plus a
  `transcoder` stretch note (M7) for attribution graphs.
- **Exit:** induction validation green; doc merged; capability advertised honestly.

---

## 5. Compute / budget tiers

All on RunPod (MCP). Qwen2.5-7B in bf16 ≈ 15GB weights; 4-bit ≈ 5GB. SAE training
is **activation-bound** (stream cached activations; the SAE itself is small).

- **Tier 0 — Free / CI (no GPU).** M0 offline mock core; metric unit tests; auto-interp
  dry-run with canned exemplars. **$0.** Runs in CI on every push.
- **Tier 1 — Single 24–32GB (RTX 4090/5090/L4), ~$0.3–0.7/hr.** 4-bit 7B harvest
  (M1, ~50–100M tokens) + one 16k SAE (M2) + dictionary/auto-interp (M3) +
  steering demo (M4). Days-scale, the bulk of the science. **Est. ~$30–80 total**
  including iteration.
- **Tier 2 — A100/H100-80GB, ~$2–4/hr.** bf16 7B, 200M–1B-token harvests, 32–65k
  dicts, JumpReLU/Gated sweeps, patching sweeps (M5) and induction validation
  (M6). **Est. ~$150–400** for a thorough pass with seed replication.
- **Tier 3 — H100/H200 or multi-GPU, stretch.** Multi-layer SAE suite,
  transcoders + attribution graphs (M7), larger dicts. Open-ended; scope per
  result. Keep network volumes for cached activations to avoid re-harvest cost.

Cost-control discipline (repo ethos): cache activations once on a network volume;
prefer attribution patching over exhaustive patching; stop/destroy pods after runs
(`mcp__runpod__stop-pod`/`delete-pod`); every run writes a manifest so reruns are
reproducible, not redone.

---

## 6. Honest metrics (pre-registered)

Reported with **bootstrap 95% CIs** and seed replication; thresholds registered
*before* runs (like `agi-proof/preregistered-thresholds.md`).

**SAE quality (M2):**
- **L0** (mean active features/token) — target band ~20–80; report the value, not a
  pass/fail alone.
- **Reconstruction:** FVU (fraction of variance unexplained) / normalized MSE.
- **CE-loss recovered** — fraction of the cross-entropy gap (clean vs
  mean-ablation) recovered by substituting SAE reconstruction; **the metric that
  matters most** (reconstruction can look good while CE collapses). Target ≥ ~0.9.
- **Dead-feature %** — fraction never firing over the eval set; target below a
  registered ceiling (e.g. <10–20% post-resampling).
- **Feature-density histogram** — flag the "ultra-high-density" (likely
  uninterpretable) and dead tails.

**Feature interpretability (M3):**
- **% features interpretable** = fraction with auto-interp score above a
  pre-registered threshold, cross-checked across ≥2 judge families (κ ≥ 0.40, the
  repo's bar). Report the *whole distribution*, including uninterpretable features.

**Steering (M4):**
- **Steering effect** = paired behavioral shift on held-out honesty prompts (95% CI
  excludes 0 to claim an effect), scored by deterministic + multi-family judges.
- **Capability preservation** = no significant regression on a general retention
  set (reuse the steering experiment's two-channel cross-validation idea).

**Patching (M5):** per-component / per-feature **restoration effect size**
(Δ logit / Δ CE) with CIs; report the full sweep including null components.

**Tooling validity (M6):** induction-head scores reproduce the known pattern
(specific heads, high induction score) — a *ground-truth* check on the harness.

---

## 7. Risks & overclaim guards

The single biggest risk is **narrative overreach** — the field's credibility tax.
Guards, enforced as review gates:

1. **No "we found the deception circuit."** Permitted language: "feature `f`
   *correlates with* deception labels and *when clamped causally shifts* behavior
   by X (95% CI …)." A circuit claim requires a *validated, sufficient-and-necessary
   subgraph* — out of scope; say so.
2. **Correlation ≠ use.** A probe/SAE-feature being decodable does **not** mean the
   model uses it (Belinkov 2022). Only **intervention** (clamping/patching with an
   effect-size CI) upgrades a correlational finding — and even then to "causally
   implicated," not "the cause."
3. **SAEs are lossy + can hallucinate features.** Report FVU/CE-recovered honestly;
   note that SAE features are a *basis we imposed*, not ground-truth model
   ontology; cross-check load-bearing features with patching on the *original*
   (non-SAE) activations.
4. **Auto-interp is an LLM judging an LLM.** Multi-family judges + κ; report
   prediction-score CIs; never treat a single label as the feature's meaning.
5. **Cherry-picking.** Pre-register layer, dict size, thresholds, and the target
   feature-selection procedure *before* runs; report the full ranked candidate
   list and all nulls. Null results ship.
6. **Generalization.** Findings are about **Qwen2.5-7B**, not "LLMs"; honesty
   features found here may not transfer. State scope explicitly.
7. **Fail-closed.** If an SAE misses the §6 bar, the dictionary/steering built on
   it is labelled provisional; downstream claims inherit the weakest upstream CI.
8. **Dual-use.** A "deception" feature that can be clamped *up* is dual-use; keep
   steering demos defensive (toward abstention/honesty) and documented under the
   repo's safety norms.

---

## 8. Effort

Roughly (one researcher-engineer; calendar overlaps with GPU runs):

- **M0** offline harness + metric tests — **2–3 days.** (Highest leverage; reuses
  `agent/steering/hooks.py`.)
- **M1** harvesting — **1–2 days** + GPU time.
- **M2** first SAE + honest metrics — **3–5 days** (most iteration risk).
- **M3** dictionary + auto-interp — **3–4 days.**
- **M4** honesty-feature steering demo — **3–5 days** (the headline result).
- **M5** hallucination patching — **4–6 days.**
- **M6** induction validation + writeup — **2–3 days.**

**MVP (M0→M2):** ~1.5–2.5 weeks to a credible, eval'd SAE on the local model.
**Headline (through M4):** ~4–6 weeks to "interpretable honesty feature, clamped,
with CIs." Patching + validation + transcoder stretch extend from there.

**Definition of done for "credible capability":** SAE meeting the §6 bar **with**
auto-interp'd dictionary **and** at least one causally-validated (clamp **or**
patch) honesty-relevant feature reported with CIs and nulls — all reproducible,
offline-CI-green at the core, and free of the overclaim language in §7.
