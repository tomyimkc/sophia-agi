# Building an LLM From Scratch — Takeaways for Sophia

> Brainstorm note. Reconciles Sebastian Raschka's *Build a Large Language Model
> (From Scratch)* (`rasbt/LLMs-from-scratch`) with Sophia's existing charter and
> stack. **Not a capability claim.** Every concrete model proposal below inherits
> the no-overclaim gate (multi-judge + CIs before any number headlines) and the
> charter rule *"don't out-train frontier labs — innovate at the trust layer"*
> ([VISION.md](../../VISION.md)).

## TL;DR

1. **rasbt teaches the *mechanism*, not the *scale*.** Its payoff is that you can
   read and modify every line of a GPT — tokenizer, attention, block, pretrain
   loop, SFT, instruction-tuning. That is exactly the literacy Sophia needs to
   stop *describing* training in pure-Python toys and *own* a real one.
2. **Sophia should not build a from-scratch LLM to compete on capability.** The
   charter forbids it and the math is hopeless. Build one to do something a
   frontier model *can't*: be **provenance-native and gate-shaped from token zero**.
3. **You already have ~80% of the plumbing.** `pretraining/nano/` (real
   hand-backpropped LM), `tools/train_lora.py|train_dpo.py|train_orpo.py`,
   `tools/runpod_train.py` (gate-disciplined LoRA on rented GPU), RLVR with a
   verifier reward, council distillation, a 528-row corpus + a 1,478-row
   `local_sophia_7b` SFT/DPO pack, and a live RunPod MCP with H100/A100/4090 in
   stock. The missing rung is a **single real from-scratch PyTorch model** that
   ties the pure-Python research to the GPU fine-tuning work.

---

## Part A — What rasbt actually gives you (mapped to Sophia artifacts)

| rasbt chapter | What it builds | Sophia artifact it upgrades |
|---|---|---|
| Ch2 Text & tokenization | BPE, dataloaders, sliding windows | `pretraining/nano/data.py` (today: order-*k* Markov toy) → a real tokenizer over the bilingual corpus |
| Ch3 Attention | self-attention → causal multi-head | `kernels/flash_attention.py` (numpy/Triton ref) gets a real consumer |
| Ch4 GPT model | full GPT-124M in PyTorch | **the missing piece** — `pretraining/` has only a 1-hidden-layer `NanoLM` |
| Ch5 Pretraining | training loop, loss curves, loading GPT-2 weights | `pretraining/scaling/`, `optimizer_probe/` get a real model to measure |
| Ch6 Classification finetune | head-swap finetuning | maps to the gate's `accept/abstain/block` classifier head |
| Ch7 Instruction finetune | SFT on instruction data | `tools/train_lora.py` + `training/corpus.jsonl` already do this on a 7B base |
| Bonus: distillation, RL, KV-cache, MoE | efficiency + alignment | `tools/distill_*`, `tools/run_rlvr.py`, `serving/`, `moe/` |

**Takeaway:** rasbt is the bridge between the two halves you already have — the
honest pure-Python `pretraining/` research and the real-GPU LoRA/RLVR tooling.
Right now nothing connects them: the research model can't run on a GPU, and the
GPU tooling only ever *fine-tunes someone else's base*. A rasbt-style GPT is the
first model that is **yours from random init**.

---

## Part B — The strategic takeaway (read this before writing code)

The charter line is blunt: *"Assemble and orchestrate; innovate at the trust
layer. Don't try to out-train frontier labs."* A from-scratch LLM only earns its
place if it **serves the trust mission instead of fighting it**. Three honest
framings that pass that test:

- **Pedagogical sovereignty.** Owning a model end-to-end (init → tokenizer →
  pretrain → SFT → serve → gate) means every claim in `pretraining/` stops being
  a toy analogy and becomes a measured property of a model you fully control.
  This is *defensible* because the deliverable is understanding + reproducibility,
  not a leaderboard number.
- **A provenance-native testbed.** Frontier models bolt provenance on *after*
  training. Sophia can ask the opposite question — *what changes if a model is
  trained to attribute and abstain from the first token?* — and that is a genuine
  research contribution no lab is incentivised to run. (See Part D.)
- **The smallest model that the gate makes trustworthy.** Sophia's whole thesis is
  that a weak model + a strong gate beats a strong model alone on *honesty*. A
  from-scratch ~10–124M model is the most extreme, most honest version of that
  experiment: how small can the model get before the gate can no longer rescue it?

**Anti-goals (write these down so the scope can't drift):** no general-capability
claim, no "we trained an LLM" marketing, no benchmark headline without the
no-overclaim gate, and `canClaimAGI: false` stamped on every artifact — exactly
like `pretraining/agent/` and `autopilot/` already do.

---

## Part C — Three feasible tiers (pick by budget & intent)

All costs are RunPod community-cloud order-of-magnitude (4090 ≈ $0.3–0.5/hr,
A100-80GB ≈ $1.5/hr, H100 ≈ $2–3/hr) and assume the existing
`tools/runpod_train.py` lifecycle (create pod → train → eval through the gate →
copy artifacts back → **always delete pod**).

### Tier 0 — "rasbt-faithful": GPT-from-scratch, laptop/single-GPU (days, ~$0–10)
The honest entry point and the one that most directly answers your question.
- Port the rasbt GPT-124M into `pretraining/gpt/` (PyTorch, optional dep — keep
  the pure-Python `nano/` as the zero-dep reference).
- Train a BPE tokenizer on `training/corpus.jsonl` + the OKF wiki + math-code
  curriculum.
- Pretrain on a 4090 (or even CPU for a 10M-param config) and reproduce the
  `scaling/` law `L(D)=E+A·D^-p` on a **real** model instead of the Markov toy.
- **Deliverable:** a `*-latest.json` report showing the same scaling methodology
  holds on a real GPT — directly upgrades `pretraining/scaling/` and `architecture/`.

### Tier 1 — "born-gated Sophia-nano" (the differentiated bet) (~$20–100)
A small (10–60M) model **pretrained with provenance structure baked in** — the
thing no frontier lab will build for you. Details in Part D. Train on 1×A100 or
2–4×4090; the corpus is small, so this is data-bound, not compute-bound.

### Tier 2 — "pragmatic": keep fine-tuning a real base (already 80% built)
If the *goal* is "a usable Sophia model" rather than "from random init," you are
already here and should just finish the existing plans:
- `training/local_sophia_7b/` — Qwen2.5-7B QLoRA pack (1,478 rows SFT+DPO).
- [Sophia-Wisdom-4B-Training-Plan.md](Sophia-Wisdom-4B-Training-Plan.md),
  [Local-Sophia-Training.md](../11-Platform/Local-Sophia-Training.md).
- `tools/runpod_train.py --yes` runs the full gate-disciplined LoRA+eval ladder.
- **This is the cheapest path to a model people can actually run**; "from
  scratch" is the *research/literacy* play, not the *product* play. Be explicit
  with yourself about which one you want — they don't compete, they layer.

---

## Part D — Creative ideas (the brainstorm proper)

Ordered roughly by novelty-to-Sophia, not by ease. Each is feasible on the
current stack; the differentiated ones cluster around *provenance-native training*.

### 1. Provenance-native pretraining tokens
Add special tokens to the tokenizer — `<src=…>`, `<conf=hi|lo>`, `<abstain>`,
`<doNotMergeWith>` — and structure the corpus so the model **emits an attribution
trail inline**, the way the corpus's `metadata.textIds`/`traditions` already
encode it. The model learns *"a claim co-occurs with its source token"* as a
first-class pattern instead of a post-hoc filter. **Falsifiable test:** does a
born-with-`<src>` model hallucinate fewer attributions at equal perplexity than
the same architecture trained on plain text? That is a real, publishable ablation.

### 2. Verifier-in-the-loss (gate-shaped pretraining)
You already have RLVR (`tools/run_rlvr.py`, `provenance_bench/rl_reward.py`) — a
verifier that returns reward. Extend it *down into pretraining*: an auxiliary loss
that penalises the model when its greedy continuation would be **blocked by
`agent/gate.py`**. The gate becomes a differentiable-ish training signal, not just
an inference filter. Start as a reranking/DPO objective (cheap, no custom CUDA),
graduate to a token-level penalty only if it pays off.

### 3. Abstention / calibration head (rasbt Ch6, repurposed)
rasbt's classification-finetuning chapter swaps the LM head for a classifier.
Train a **second head** that predicts `accept | hedge | abstain` — i.e. make
`agent/graded_decision.py` a *learned* head instead of a threshold rule. The
VISION.md "Self-model & calibration" pillar is marked ⚠️ *partial* precisely
because confidence is a weak correctness predictor today; a trained head, measured
with the existing ECE/risk-coverage tooling (`agent/calibration.py`), is the
honest fix.

### 4. Council → student distillation, but starting from your own base
`tools/distill_council_traces.py` already distills the multi-seat council into a
student. Today the student is a borrowed base. Distil into your **Tier-0
from-scratch GPT** instead — the cleanest possible test of *"can a tiny model born
inside Sophia absorb the council's discipline?"* ([Council-Distillation.md](../11-Platform/Council-Distillation.md)).

### 5. Known-floor honesty, scaled up
`pretraining/nano/` is special because the source's irreducible loss `E` is known
in closed form, so every fit is *checked, not trusted*. Preserve that discipline
on the real GPT: hold out a synthetic sub-corpus with a computable entropy floor
and report real-model fits against it. This keeps the from-scratch work inside the
no-overclaim charter instead of drifting into vibes-based loss curves.

### 6. Bilingual-first tokenizer as a research artifact
The corpus is EN+中文 with deliberate 儒家/道家 lineage separation. A from-scratch
BPE tokenizer trained on *this* mixture (vs. an off-the-shelf English tokenizer)
is itself a small contribution: measure token efficiency and whether lineage terms
(老子/孔子) stay in distinct subword neighbourhoods. Feeds `data_mixing/` (配比).

### 7. MoE / efficiency, but only with the trust governor
`moe/` (top-k routing + INT8/FP8) and `kernels/flash_attention.py` exist but have
no real model to accelerate. A from-scratch GPT gives them one. Charter rule
([Governed-Scaling.md](../11-Platform/Governed-Scaling.md)): adopt an efficiency
primitive **only with an equivalence/error-bound proof bolted on** — so any MoE or
quant win ships with a measured accuracy-delta, never as raw speed.

### 8. "Born-gated" model card with a failure ledger
Ship the model the way Sophia ships everything: a HF model card
(`models/hf-model-card/`) whose headline section is *what it cannot do*, mirroring
`agi-proof/failure-ledger.md`. The model's honesty about its own limits becomes
part of the artifact — a differentiator, not an afterthought.

### 9. Autopilot-driven pretraining sweeps
`pretraining/autopilot/` already proposes→runs→reads-back nano configs and can
emit a **gated, dry-run-by-default** RunPod plan. Point it at the real GPT: let it
hill-climb LR / data-mixture / N-vs-D (Chinchilla-flavoured) on actual hardware,
with the existing cost-ceiling guard so it *never auto-spends*.

---

## Part E — Concrete first move — **scaffolded** in `pretraining/gpt/`

Tier 0 is now in the tree (see [`pretraining/gpt/README.md`](../../pretraining/gpt/README.md)).
It converts the whole `pretraining/` story from analogy to fact and unlocks every
idea in Part D:

1. ✅ `pretraining/gpt/model.py` — readable rasbt-faithful GPT (PyTorch, optional
   dep; `nano/` stays the zero-dependency reference and CI path). Causal attention
   via `scaled_dot_product_attention` — no x86-only flash-attn wheel, runs on
   CUDA/MPS/CPU.
2. ✅ `pretraining/gpt/tokenizer.py` — dependency-free **byte-level** codec that
   **reserves the `<src>`/`<abstain>`/`<doNotMergeWith>` special tokens now**
   (ids ≥256), so the born-gated vocab is stable before idea #1 lands. (BPE merges
   are a later, vocab-compatible hook.)
3. ✅ `pretraining/gpt/train.py` — device-agnostic loop logging the same
   `epoch_loss` / `grad_norms` shape as `nano` and `optimizer_probe`; writes
   `gpt-train-latest.json` stamped `canClaimAGI: false`.
4. ✅ `tests/test_gpt_pretraining.py` — tokenizer/data/cluster tested
   dependency-free (CI); torch model + a real descending step under `importorskip`.
5. ✅ `pretraining/gpt/born_gated.py` + `--born-gated` — idea #1 made runnable:
   `data/attributions.json` → inline `<src>`/`<conf_*>`/`<doNotAttributeTo>` text
   (low-confidence by default, fail-closed) trained as first-class tokens.
6. ✅ `pretraining/gpt/scaling.py` — reproduces the `scaling/` law on the real GPT
   with a pre-registered extrapolation gate.
7. ✅ Abstention head (idea #3, `gpt/abstain.py` + `model.py` head, calibrated via
   `agent/calibration.py`), verifier-in-the-loss (idea #2, `gpt/verifier_loss.py`),
   born-gated ablation (`gpt/ablation.py` + `gpt/provenance_eval.py`), tokenizer
   fairness (idea #6, `gpt/tokenizer_analysis.py`), and a failure-ledger-first
   model card (idea #8, `agi-proof/model-cards/sophia-gpt-nano.md`).
8. **Next (needs the cluster/GPU to *run*, not to build):** a multi-seed real
   pretrain to interpret the ablation sign; council distillation into this base
   (idea #4); MoE/quant under the trust governor (idea #7).

> **Status:** the from-scratch track is **fully scaffolded and CI-tested on its
> dependency-free surface**. Every torch path compiles and is tested under
> `importorskip`; the remaining work is *running* them at scale on the Spark/M3
> cluster — which produces measured numbers, not more code.

### Engineering discipline (Karpathy skills, applied)

Studied [`multica-ai/andrej-karpathy-skills`](https://github.com/multica-ai/andrej-karpathy-skills)
— four anti-pitfall coding guidelines (think-before-coding, simplicity-first,
surgical-changes, goal-driven-verified-execution). They map cleanly onto Sophia's
charter (fail-closed, no-overclaim, "every change more checkable"), so they are
adapted into a portable skill: [`skills/portable/sophia-karpathy-engineering/SKILL.md`](../../skills/portable/sophia-karpathy-engineering/SKILL.md).
The overlay adds Sophia's reflexes: abstain-and-ask on uncertainty, prefer
dependency-free/deterministic, never weaken a gate to pass a diff, and "done"
means *a check was run and observed* (no claimed-green-without-running).

Everything through step 4 runs free on CPU/laptop, exactly as rasbt intends — and
every piece lands inside an existing folder, test, and gate.

### Cluster mapping — DGX Spark + Mac Studio M3 Ultra

`pretraining/gpt/cluster.py` encodes the charter's node roles so the same script
runs everywhere (`--prefer auto`):

| Node | tier | device / dtype | Use |
|---|---|---|---|
| **DGX Spark** (GB10, aarch64, 128 GB unified) | `spark` | CUDA / **bf16** | pretrain + iteration; **never MLX**, bf16 not 4-bit (bitsandbytes aarch64 pain) |
| **Mac Studio M3 Ultra** (96 GB unified) | `m3` | MPS / fp16 | pretrain/eval on Apple; hand LoRA + serving to MLX tooling (`tools/eval_mlx_model.py`, `training/mlx_adapters/`) |
| **CPU / CI** | `cpu` | fp32 | tokenizer + smoke train |

Both nodes hold the whole small model in unified memory, so Tier 0/1 is
**data-bound, not compute-bound** on this cluster. The honest boundary
([DGX-Spark.md](../11-Platform/DGX-Spark.md)) still holds: Spark/M3 runs are the
**iteration tier** (`headline_ok: false`); registered headline numbers stay on
x86 RunPod, with `tools/spark_vs_runpod_ab.py` keeping that rule data-backed.

---

*Boundary: this document is a research brainstorm. No model here exists yet; no
number is claimed. The from-scratch model is a literacy + provenance-research
play, explicitly **not** a frontier-capability or AGI claim.*
