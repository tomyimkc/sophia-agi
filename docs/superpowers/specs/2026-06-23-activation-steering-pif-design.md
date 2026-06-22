# MBTI Vector Agents — Spec B: Level-3 Activation-Steering Engine + Behavioral PIF

**Date:** 2026-06-23
**Status:** Approved → ready for implementation plan
**Program:** "MBTI Vector Agents" — Spec **B** of 4 (stacks on Spec A; C/D in §12)
**Branch:** `feat/activation-steering-pif` (off `feat/personality-measurement-gate` / PR #64)

## Problem

Spec A proved we can *declare* a personality target, *induce* it with a Level-1
persona prompt, and *measure* the result under a no-overclaim gate. But a persona
prompt is the weakest possible induction — it changes what the model is *told*,
not what it *is*. Spec B adds the **Level-3** induction channel — **activation
steering**: extract a direction in the residual stream for each Big-Five axis and
add it back during generation via `register_forward_hook`.

The entire reason B exists is one falsifiable question:

> **Does activation steering produce a strictly larger, behavior-corroborated,
> capability-preserving OCEAN shift than Spec A's Level-1 persona-prompt baseline,
> on the same inventory and seeds?**

If steering does *not* beat a prompt, **`SSA = 0/N` is a publishable honest
result**, not a failure to hide. This repo refuses to overclaim, so B is built to
make "steering beats prompting" *hard to assert* and easy to falsify.

## The central constraints (every component is shaped by these)

1. **Two-tier execution, mirroring `tools/run_rlvr.py`.** A **deterministic CI
   core** (a toy `nn.Module` + synthetic data; pure-Python — numpy, stdlib,
   already-installed CPU torch) proves the steering *machinery* is arithmetically
   correct with **no GPU, no network, no model download**. **Opt-in real runs**
   (`--model phi3.5`) execute on this M3/MPS machine and produce *illustrative*
   empirical numbers behind K≥20 seeds with reported variance — **never a CI
   assertion**. Only the pure `score_items` scorer is truly deterministic; MPS
   kernels and Ollama GPU float ops drift run-to-run even at temp 0.

2. **MBTI veneer-invariance is inherited from Spec A, unbroken.** `mbti_to_ocean`
   is consumed *upstream* to choose which axis vectors/signs to apply. **No
   composed-vector, gate, verifier, judge, or effect-size path reads the MBTI
   string.** A unit test asserts the composition path is identical with the MBTI
   label present or absent.

3. **Interference is the headline, not a footnote.** OCEAN directions are not
   orthogonal; naive vector summation bleeds across traits. We orthogonalize
   (soft-projection), but orthogonalization *reduces*, it does **not eliminate**,
   behavioral cross-trait interference. **A low pass-count is the honest
   Geometric-Limitations finding, not a bug to bury.** The **T/F→Agreeableness
   axis (Spec A's weakest, sex-confounded mapping) is pre-registered as
   expected-entangled → ABSTAIN.**

4. **Family-disjointness: the judges are never the subject.** Subject = Phi
   (Phi family); judges = Qwen + Llama (two distinct families). The run **aborts
   /abstains** if `{subject} ∩ {judges} ≠ ∅` — protecting non-circularity and
   catching the fallback hazard in §10.

## Locked decisions (owner)

- **Subject = local `microsoft/Phi-3.5-mini-instruct`** (ungated, in-process
  `transformers`), with an explicit **fallback chain** (§10) of ungated,
  distinct-architecture instruct models on MPS/load failure.
- **Judges = local Ollama `qwen2.5:3b` + `llama3.2:3b`**; **optional** remote
  OpenRouter family as a stronger third judge, key **only in gitignored `.env`**.
- **Full behavioral half of PIF** in scope.
- **Pre-registered SSA thresholds (locked before any run):** N=8 personas (all
  five OCEAN poles), K=20 seeds, steered residualized `d > 0.5`, superiority
  `Δd` point ≥ **+0.3** with bootstrap 95% CI lower bound > 0, off-target
  `|d| < 0.2`, inter-judge `κ ≥ 0.40`, capability `ε = 5%` relative drop +
  coherence floor 75.
- **Defer the GPU GRPO loop:** build the contamination-free extract/measure split
  + the offline reward-wiring invariants only; the trained steering policy is
  out of scope (B's claim is *measurement* superiority, not a trained policy).
- **Build the engine + execute ONE real demo run** (gated Phi-3.5 download +
  one steered, Ollama-judged run producing illustrative SSA numbers vs Level-1).
  The demo is a **reduced-scope, end-to-end illustration** (e.g. 2–3 axes, a
  handful of personas/seeds) that proves the whole loop runs on this M3 and emits
  a real Δd table; the full pre-registered N=8/K=20 headline run is a **separate,
  later execution** — the demo run never makes a headline SSA claim.

## Components (each small, single-purpose; offline-testable core)

| Module (new) | Responsibility | Interface | Reuse / deps |
|---|---|---|---|
| `agent/steering/vectors.py` | CAA difference-of-means extraction on contrastive high/low-trait prompt pairs at layer L; **normalized** direction. Ships a **deterministic numpy mock extractor** (seeded, no torch). Serializes to `training/steering/vectors/<model-slug>/<DIM>.json` with provenance `{layer, model, prompt-pair hash, transformers_version, normalized:true}`. | `extract_persona_vector(model, pairs, layer, *, normalize=True) -> Vector`; `mock_vector(dim, seed) -> Vector` | torch/transformers **lazy** (real path only); numpy (mock). Reuses Spec A `data/personality_items.json` IPIP items as poles. |
| `agent/steering/hooks.py` | `register_forward_hook` apply of `alpha·v̂` at target layer(s); `attach_hooks(...)` context manager guaranteeing `handle.remove()`. **`SteeredClient`** duck-types Spec A's `generate(system,user)->result(.text,.ok)` so one code path measures prompt-steered AND activation-steered runs. Offline shim = deterministic no-op. | `attach_hooks(model, vec, alpha, layers)`; `SteeredClient.generate(...)` | torch lazy; duck-types `agent/model.py:ModelResult`; in-process load mirrors `tools/eval_local_model.py`. |
| `agent/steering/compose.py` | Combine per-axis vectors → one steering tensor: orthogonalize (**soft-projection default**; Gram–Schmidt + Löwdin available for ablation), normalize, per-axis alpha, clamp. **Pure numpy → fully CI-testable.** Takes OCEAN target **signs only** — never an MBTI string. | `compose_vectors(vecs, alphas, *, scheme="soft_proj") -> (vec, manifest)` | numpy only. |
| `agent/personality_behavioral.py` | The **behavioral** PIF channel (sibling to Spec A's self-report `personality_measure.py`). Open-ended (non-Likert) elicitation; runs the steered client; scores enacted trait via `personality_faithful` + the Ollama judge panel; emits per-axis behavioral d + coherence + κ. | `score_behavioral(client, dim, *, judges) -> {trait_d, coherence, kappa, verdict}` | reuses `personality_faithful`, `target_markers` from `tests/benchmark-personality.json`, judges via `agent/model.py` ollama/openrouter presets. |
| `tools/run_steering.py` | CLI **mirroring `tools/run_rlvr.py`**: `--model mock`(default)/`--dry-run` → `_offline_invariants()` (no torch); `--model phi3.5` → real MPS extraction + steered measurement + judged battery. Writes `agi-proof/benchmark-results/steering.public-report.json` (`claimStatus "Open — capability claim requires a gated run"`); registers a `agi-proof/failure-ledger.md` entry. | `python tools/run_steering.py [--model …] [--device mps] [--dry-run]` | reuses `provenance_bench/aggregate.py:_ci`, `consensus.py:cohen_kappa`, `KAPPA_FLOOR`. |
| `provenance_bench/steering_dataset.py` | Entity-disjoint extract/measure split so a vector is never evaluated on its own fitting prompts. | `build_steering_dataset() -> {train_*, eval_*, *_sealed, entity_intersection}` | mirrors `provenance_bench/rl_dataset.py` (`split_cases`, `sealed_hash`, `entity_intersection`). |

**Critical transport note.** `agent/model.py` **already speaks Ollama and
OpenRouter** (both `kind='openai'`, single urllib transport; ollama at
`http://localhost:11434/v1`, openrouter at `https://openrouter.ai/api/v1` with
`OPENROUTER_API_KEY`; localhost whitelisted under airgap). **No new HTTP client.**
But an HTTP client *cannot inject activation vectors* — so the real Level-3
channel needs the **in-process `transformers` model** (`hooks.py`).
`agent/model.py` is reused only for the prompt-only Level-1 baseline, the judges,
and the mock provider in offline tests.

## Steering mechanics (Phi-3.5, MPS)

**Verified model facts:** 32 hidden layers, `hidden_size = 3072`, full MHA (no
GQA), `intermediate_size = 8192`, LongRoPE. `transformers` exposes layers as
`model.model.layers[L]`; the decoder layer's forward returns the residual stream
as `output[0]` (4.x tuple) or a bare tensor (5.x — installed env is **5.5.3**).

**fp32-on-MPS handling (verified).** Load fp16 (**not bf16** — Phi-3.5 config
says bfloat16 but MPS bf16 is chip/macOS/PyTorch-dependent; force fp16/fp32).
Keep the steering vector master copy in **fp32**; cast to `h.dtype/h.device`
*inside* the hook. **Never let float64 touch MPS** (numpy defaults to float64 →
wrap `torch.tensor(arr, dtype=torch.float32)`). Set
`PYTORCH_ENABLE_MPS_FALLBACK=1` **before** `import torch` (cheap insurance, not a
guarantee — it does not cover every unimplemented op). Use
`attn_implementation='eager'` for reproducible activations (MPS SDPA has
correctness edge cases). The hook fires every forward pass → auto-reapplies at
every autoregressive decode step under `model.generate`.

```python
def make_steering_hook(vec_f32, alpha, device):
    def hook(module, inputs, output):
        if isinstance(output, tuple):                    # transformers 4.x
            hs = output[0]
            v = vec_f32.to(device=hs.device, dtype=hs.dtype)
            return (hs + alpha * v,) + tuple(output[1:])  # MUST return tuple
        hs = output                                      # transformers 5.x: bare tensor
        v = vec_f32.to(device=hs.device, dtype=hs.dtype)
        return hs + alpha * v                            # MUST return tensor
    return hook
# attach_hooks(): register_forward_hook(...) → keep RemovableHandle → handle.remove() in finally
```

> **Verified correction (layer):** "~2/3 depth is canonical" was **refuted**.
> Treat the layer as **swept, not fixed** — offline sensitivity sweep over
> `L ∈ {16..24}`, pick per-axis; start L=21.

## CAA extraction + composition + orthogonalization

**Extraction (CAA Eq. 1, verified):** `v_MD = (1/|D|) Σ [a_L(p_pos) − a_L(p_neg)]`
over paired positive/negative trait prompts built from the Spec A IPIP items
(e.g. E+ "I am the life of the party" vs reverse-keyed E− "I keep in the
background"). Add at **all post-prompt generated positions**, scaled by signed
scalar `alpha` (positive → high pole, negative → low pole).

> **Verified correction (normalization):** the original research said "leave the
> vector unnormalized; raw L2 sets the magnitude" — **refuted/inverted.** Both
> CAA and Persona Vectors **normalize the direction** and scale a unit/fixed-norm
> vector by `alpha`. **Decision: normalize each axis vector, steer
> `h ← h + alpha·v̂`, tune `alpha` per layer; record `normalized:true`.**

**Composition + orthogonalization.** The five raw OCEAN directions are not
orthogonal (e.g. on LLaMA-3-8B, steering Openness cross-bleeds ≈−3.5 onto
Neuroticism). Schemes (arXiv:2602.15847):
- **C4 soft projection** `d_i ← d_i − β⟨d_i,d̂_j⟩d̂_j` (β=0.5, τ=0.5) — best
  trade-off (0.85–1.00 signal retention) — **our default**.
- C2 greedy Gram–Schmidt — strict orthonormality but **order-dependent**;
  offered, not default.
- C5 Löwdin — perfect orthogonality but fluency loss; ablation only.

> **Honest limitation (verbatim into the report):** hard orthonormalization
> removes *linear* overlap but **does NOT eliminate behavioral cross-trait
> interference** — even at `max|cos|<1e-8`, behavioral bleed persists. Treat
> orthogonalization as interference *reduction*; **validate each axis
> behaviorally after composition; keep alphas modest.** Emit the **5×5 cosine
> Gram matrix pre-flight** before any composition. The T/F→Agreeableness axis is
> pre-registered **expected-entangled → ABSTAIN**.

## Behavioral PIF channel

**Battery (Persona-Vectors discipline).** Per OCEAN axis, ~20 **held-out,
open-ended** prompts that pull on the trait's behavioral surface but **never name
the trait** (e.g. E: "You arrive at a party where you know no one. Describe your
next hour."). Prompts **byte-identical across conditions**; the persona is
realized only by steering. ≥10 rollouts per (axis × condition × prompt), varying
generation seed `base+i`; judging at temp 0.

**Rubric (judge blind to condition, one trait per call).** Strict JSON
`{trait_score:0-100, coherence:0-100, reason}`, both poles anchored. **Discard /
flag `coherence < 75`** so garbled output can't inflate trait scores.

**Local deterministic Ollama judging.** `options.temperature=0`,
`options.seed=42`, `stream:false`, pinned tags (`qwen2.5:3b`, `llama3.2:3b`),
fixed `num_ctx`/`num_predict`, `format:"json"`; record Ollama version + model
digest.

> **Verified caveat:** temp 0 + seed *reduces* but does **not** eliminate
> run-to-run drift (non-associative GPU float ops). Variance is quantified across
> K≥20 replicates; only the pure scorer is the deterministic core.

**Stats.** Cohen's d (steered vs neutral, pooled-SD two-group); **bootstrap 95%
CI** via `provenance_bench/aggregate.py:_ci` (seed fixed, `n_boot≥2000`; unit of
analysis = **per-prompt-mean** primary, per-response robustness check).
Inter-judge agreement via `provenance_bench/consensus.py:cohen_kappa`, **floor
`KAPPA_FLOOR=0.40`**; bin 0–100 → ordered bins; report raw % agreement +
Spearman alongside (κ is paradoxically low under skewed marginals).

> **Verified nuance:** Persona Vectors validated its judge against humans via
> **Pearson r>0.87**, not κ — don't imply they used weighted-κ. Weighted-κ-vs-human
> on a calibration subset is *our* proposed additional check (deferred to C).

**Optional OpenRouter third judge.** OpenAI-compatible, pinned slug+snapshot,
`temperature=0`, key from gitignored `.env`. **Off by default; trigger only when
local κ<0.40 on an axis.** Cache transcripts as golden fixtures so reported
numbers re-derive offline. On network/key failure → **degrade-to-ABSTAIN, never
silently degrade to local-only**.

## Data flow

```
request (OCEAN axis + intensity; or MBTI→ocean signs via personality_map; MBTI string dropped here)
   ├─ extract per-axis v̂  (vectors.py; CAA diff-of-means, normalized)        ← entity-disjoint extract split
   ├─ compose + orthogonalize (compose.py; soft-projection)                   → composed vector + manifest
   ├─ Level-3 arm: attach_hooks(model, vec, alpha, L) → SteeredClient          (NO persona prompt)
   └─ Level-1 arm: persona-as-system-prompt via agent/model.py                 (NO activation hook)
       (both arms: same IPIP bank, same K≥20 seeds, same neutral baseline, same scorer, same snapshot)
   ├─ SELF-REPORT channel:  measure_ocean(steered_client, bank) → score_items (pure, no model)
   └─ BEHAVIORAL channel:   score_behavioral(steered_client, axis, judges)     ← Ollama panel, κ
   ▼  residualize target vs off-target 4-vector → Cohen's d (+bootstrap CI) per channel
   ▼  paired Δd = d_steer − d_baseline  (paired by persona×axis×seed)
   ▼  verdict: enacted iff superiority ∧ floor ∧ orthogonality ∧ behavior-corroboration ∧ capability-preserved
              else ABSTAIN (reason)   — three-way, fail-closed, mirrors personality_faithful
```

## Headline metric — SSA (Steering Superiority over the Activation-free baseline)

Pre-registered, **count-out-of-N**, monotone-down under stricter measurement:

> *"For N≥8 personas spanning all five OCEAN poles, Level-3 activation steering
> produces a residualized requested-axis OCEAN shift strictly LARGER than the
> Spec A Level-1 persona-prompt baseline on the same inventory and seeds, while
> remaining coherent, capability-preserving, and behavior-corroborated."*

A persona×axis cell counts toward SSA **only if all hold** (any failure →
ABSTAIN, lowering the count):

1. **Superiority (headline):** paired bootstrap 95% CI of `Δd = d_steer −
   d_baseline` (paired by persona+axis, `_ci`, seed-fixed, `n_boot≥2000`)
   excludes zero, **lower bound > 0, point estimate ≥ +0.3**, survives Holm/BH.
2. **Absolute floor:** steered residualized `d > 0.5`, replicated across K≥20 seeds.
3. **Orthogonality:** every off-target axis `|residualized d| < 0.2`.
4. **Behavior corroboration:** ≥2 judge families distinct from the subject agree
   the trait moved, inter-judge `κ ≥ 0.40`. Self-report shift + null/opposite
   behavior → ABSTAIN.
5. **Capability preservation:** held-neutral slice (GSM8K-style + judge-coherence)
   steered-vs-unsteered, relative drop `< ε = 5%` + coherence floor 75. Report
   the **(shift, retention) PAIR**, never shift alone.
6. **Non-mock:** real subject (Phi-3.5 or a recorded fallback), not `mock`.

Reported as **count-out-of-N + binomial 95% CI + per-axis abstention rate + the
(Level-3 vs Level-1) Δd table**. **Never** a single aggregate score, **never**
"MBTI type achieved", **never** human-norm percentiles. **`SSA = 0/N` is a
legitimate honest result.**

**Required ablations** (defend against "steering is just a fancier prompt"):
(i) α dose-response (flat/non-monotone = red flag); (ii) **random-direction
control** of matched norm must NOT reproduce the shift; (iii) wrong-axis control;
(iv) composition without one axis cannibalizing another.

**Abstain conditions (fail-closed):** steer ≤ baseline; capability/coherence drop
> ε; κ<0.40 or a judge overlaps the subject family; off-target `|d|≥0.2` or the
random-direction control reproduces the shift; self-report shift without behavior;
K<20 / n=1; mock/unset subject; **any MBTI-veneer leak into a gate/judge/effect
input or "type achieved" framing**; any human-norm percentile.

## Testing discipline

**CI TIER — `tests/test_personality_steering.py` + `tests/test_steering.py`**
(plain `main()->int`, **NO pytest, NO GPU/network/download**; torch 2.4.0 CPU is
already installed). Seeds `torch.manual_seed(0)`, builds a toy decoder
(`d_model=16, n_layers=3`), asserts a dict of named booleans. Unit-tested offline:

1. `hookAddsAlphaV` — `hidden[L]_steered == clean + alpha*v` (allclose);
   `hookSurgicalNoLeak` — other layers byte-identical; `hookRemovalRestores`.
2. `diffOfMeansRecoversDirection` — plant unit `u`, assert `cos(v̂,u) > 0.98` at
   n=512, sign correct.
3. `gramSchmidtOrthogonal` / `compositionLinear` — `|dot|<1e-6`, norms as spec'd,
   injecting two composed vectors reproduces the sum at the hook.
4. `cohenDMatchesAnalytic` + `residualizationRemovesHalo` + bootstrap-CI
   excludes/includes-zero (seeded `_ci`).
5. `kappaIdentityIsOne` / `kappaNegationIsMinusOne` / `kappaConfusionMatchesClosedForm`
   (reuse `consensus.cohen_kappa`).
6. `abstainOnCIZero` / `abstainOnHalo` / `abstainOnLowKappa` /
   `abstainOnCapabilityDrop` / `abstainOnSteerNotBeatingBaseline` — each synthetic
   cell trips its condition → ABSTAIN with the matching reason string.
7. **`veneerInvariant`** — composition verdict identical with/without an MBTI label.

Plus `python tools/run_steering.py --model mock --dry-run` runs the **same**
invariants through the **shipping** extract/hook/score functions (not a test-only
reimpl) — exactly run_rlvr's mock/dry-run tier. Both wired into
`.github/workflows/ci.yml` next to `test_personality.py`.

**REAL TIER (never CI; opens a failure-ledger entry until a gated run).**
`python tools/run_steering.py --model phi3.5` does real Phi-3.5 MPS extraction +
steering + Ollama-judged battery, guarded by a **load-and-smoke probe** (load → 8-
token greedy → hidden-state shape assert at L → fp32 hook-delta dtype assert →
**no-silent-CPU-fallback assert**) before any steering run. The **multi-GB
download is gated** behind a preflight (free-disk + `ollama list` + `huggingface`
presence; prints "pull N GB first" rather than failing mid-run). The owner's
chosen **one real demo run** is executed here to produce illustrative SSA numbers.

## Repo integration (file-by-file)

New files (all under the `feat/activation-steering-pif` worktree):

- `agent/steering/__init__.py` — package marker; re-exports the three steering
  APIs; **torch lazy-imported inside functions** (importing the package must not
  require torch).
- `agent/steering/vectors.py`, `agent/steering/hooks.py`, `agent/steering/compose.py`
- `agent/personality_behavioral.py` — behavioral/judge channel.
- `provenance_bench/steering_dataset.py` — entity-disjoint split (mirrors `rl_dataset.py`).
- `tools/run_steering.py` — CLI mirroring `tools/run_rlvr.py` (anchors:
  `_offline_invariants` L73, `--model mock` default L238, `claimStatus "Open — …"`
  L216, "REWARD WIRING VERIFIED" print pattern L269).
- `tests/test_steering.py` + `tests/test_personality_steering.py`.
- `requirements-steering.txt` — header mirrors `requirements-rl.txt` ("in-process
  only — offline `--model mock` needs NONE of these; pure Python core = numpy +
  stdlib"). Contents: `torch>=2.3`, `transformers>=4.46.2` *(raise the floor — the
  `<5.0` pin in requirements-lora is too loose for Phi-3.5 LongRoPE; installed env
  is 5.5.3)*, `accelerate>=1.14.0`, `numpy>=1.26`, `safetensors>=0.4`,
  `bitsandbytes>=0.43; platform_system != "Darwin"`. **No vLLM** (incompatible
  with in-process forward hooks).
- `docs/09-Agent/Steering-Experiment.md` — companion to `RLVR-Experiment.md`.

**Spec A surfaces reused verbatim (import, do not fork):**
1. `agent/personality_measure.py` — `score_items` (pure deterministic
   effect-size oracle), `measure_ocean` (the `SteeredClient` is passed in),
   `load_bank` + `data/personality_items.json`.
2. `agent/verifiers.py:personality_faithful` (registered in `VERIFIERS`; alias
   `personality_discipline`) — the behavioral-PIF reward seam, mirroring how the
   RLVR reward wraps `provenance_faithful`. **Veneer-invariance inherited as a
   hard constraint.**
3. `tests/benchmark-personality.json` (`mustExpressTarget`/`mustLabelMyth`) — the
   enactment + myth targets a vector must move.

**Benchmark / leaderboard:** the steering eval emits the **same artifact shape**
into `benchmark/model_runs/local-<label>-<domain>.json` (+ `.report.json`) that
`tools/eval_local_model.py` produces, so `update_leaderboards.py` /
`rescore_model_runs.py` pick it up for free.

**Left alone:** `personality_map.py:mbti_to_ocean` is consumed *upstream* only;
no gate/verifier/effect-size path reads the MBTI string.

## Risks & mitigations

| # | Risk | Sev | Mitigation |
|---|---|---|---|
| 1 | **Model-load / MPS runtime failure** (Phi-3.5 LongRoPE coupling; SDPA/bf16 NaNs; silent CPU fallback; `trust_remote_code` supply chain) | HIGH | load-and-smoke probe; **fallback chain** (below); pin `transformers` floor; `eager` attn; force fp16/fp32; assert no silent CPU fallback; record full env manifest so the headline is never attributed to a silently-swapped subject. |
| 2 | **Capability/coherence collapse (caricature)** | HIGH | report the **(shift, retention) pair**; capability slice + coherence floor 75; **Pareto-frontier α selection pre-registered** (not tuned to the eval); degeneracy detector (n-gram repetition, TTR, length blowup) hard-fails to ABSTAIN. |
| 3 | **Axis interference / non-orthogonality** | HIGH | Gram/cosine pre-flight; off-target null-band; soft-projection; report raw-additive too; T/F→A pre-registered ABSTAIN; **low pass-count is the honest headline**. |
| 4 | **Judge reliability** (two 3B judges below κ=0.40) | HIGH | κ is a gate; pre-registered escalation: κ<0.40 → add a distinct OpenRouter judge; still <0.40 on an axis → that axis ABSTAINS. Randomized-order, rubric-anchored judging. |
| 5 | **Non-determinism of real runs vs deterministic CI** | MED | two-tier split; CI asserts only math on golden fixtures; real runs illustrative behind K≥20 with reported variance + full env manifest. |
| 6 | **Hosted judge reintroduces network/non-determinism/circularity** | MED | hard family-disjointness assertion; pin OpenRouter slug+snapshot; Tier-2 only; **degrade-to-ABSTAIN** on failure; cache transcripts as golden fixtures. |
| 7 | **Download size / disk / first-run latency** | MED | never download in CI; Tier-2 preflight checks disk + presence; separate first-run pull latency from steady-state in the manifest. |
| 8 | **Reward / measurement hacking** | HIGH | independent self-report + behavioral channels must converge; trait-keyword-stuffing detector; residualize off halo; **all thresholds pre-registered**; PIF count-out-of-N (framing cannot inflate). Held-out sealing is Spec C. |

**Fallback chain (record which loaded; each ungated, dense/non-MoE, hook-friendly,
distinct family):** `microsoft/Phi-3.5-mini-instruct` [PRIMARY, Phi] →
`HuggingFaceTB/SmolLM2-1.7B-Instruct` [Llama-arch, cleanest eager-attn on MPS] →
`ibm-granite/granite-3.1-2b-instruct` [Granite] → `stabilityai/stablelm-2-1_6b-chat`
[StableLM]. **Cross-constraint:** if SmolLM2 (Llama-arch) becomes the subject,
**swap the Llama judge out** (use Qwen + a non-Llama family) so `{subject} ∩
{judges} = ∅`. With the Phi primary the separation is already clean.

**Security note.** The OpenRouter key lives **only in a gitignored `.env`**
(`OPENROUTER_API_KEY`, the env name `agent/model.py` already expects). Never in
the spec, report, manifest, or CI. Remote judge is opt-in; missing key → that
channel ABSTAINS (no error, no silent local fallback).

## Open decisions — resolved at recommended defaults

1. **Steering layer:** swept `L∈{16..24}` offline, per-axis; start L=21.
2. **Normalization:** normalize each axis vector, per-layer α; `normalized:true`.
3. **Composition:** C4 soft projection default; always also report raw-additive.
4. **Capability slice + ε:** small GSM8K subset + short-QA coherence; ε=5%
   relative + coherence floor 75 (pre-registered).
5. **Third-judge escalation:** OpenRouter off by default; trigger only at
   κ<0.40 on an axis; one distinct frontier family (not Qwen/Llama/Phi).
6. **Unit of analysis:** per-prompt-mean primary, per-response robustness check.
7. **N/K:** N=8, K=20 (pre-registered minimums).
8. **RL/GRPO:** build the contamination-free split + offline reward-wiring
   invariants now; defer the GPU GRPO loop.

## Explicitly deferred to Spec C/D

- **Held-out anti-gaming family + sealed hidden-eval** (a held-out inventory AND
  behavioral battery invisible to whatever produced the vector/coefficient; shift-
  on-seen-but-absent-on-held-out → ABSTAIN). Spec B stubs the interface; the
  sealed machinery is **Spec C**. Until then B's behavioral PIF is
  **illustrative-grade, honestly labelled, not headline-grade.**
- **A trained steering policy** (GRPO/RL that optimizes the vector/coefficient).
- **Human-rater calibration of the judges** (weighted-κ-vs-human on a gold set).
- **Cross-model generalization** (vectors transferred across subjects; the
  fallback chain as first-class subjects rather than degradation fallbacks).
- **Full FastMCP packaging + capability-retention as a product gate** — Spec D.

## Residual uncertainty (honesty ledger)

1. Normalization was the original research's inverted claim — corrected here
   (normalize + α-scale); confirm against the live Persona-Vectors / CAA code
   during implementation.
2. Layer 21 is a *starting point*, not canonical — the sweep decides.
3. "Deterministic" real runs are best-effort on MPS/Ollama; only `score_items`
   is bitwise-deterministic.
4. Two 3B judges may not clear κ≥0.40 on some axes → those axes ABSTAIN (this is
   the honest behavior, not a failure to fix by relaxing the floor).
5. Phi-3.5 on MPS is the primary risk; the fallback chain + load-smoke probe
   contain it, and the env manifest prevents silent subject-swap from corrupting
   the headline.

## Sources (primary)

- Rimsky/Panickssery et al. (2023), *Contrastive Activation Addition*,
  arXiv:2312.06681.
- Chen, Arditi, Sleight, Evans, Lindsey (2025), *Persona Vectors*,
  arXiv:2507.21509.
- *Geometric Limitations of steering* (2026), arXiv:2602.15847 (composition /
  interference).
- PyTorch MPS notes; HF `register_forward_hook` documentation.
- Reuses Spec A: `docs/superpowers/specs/2026-06-22-personality-measurement-gate-design.md`.
