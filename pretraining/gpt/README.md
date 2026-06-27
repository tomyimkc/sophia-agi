# `pretraining/gpt/` — Tier-0 from-scratch GPT

> The "rasbt-faithful" model from
> [`docs/06-Roadmap/From-Scratch-LLM-Brainstorm.md`](../../docs/06-Roadmap/From-Scratch-LLM-Brainstorm.md).
> A small, fully-owned, **provenance-vocab-reserved** GPT that bridges the
> zero-dep `pretraining/nano` research model and the real-GPU fine-tuning tools.
> **Not a capability claim** — every run stamps `canClaimAGI: false`, and headline
> numbers stay on x86 RunPod (a Spark/M3 is the iteration tier — see
> [`docs/11-Platform/DGX-Spark.md`](../../docs/11-Platform/DGX-Spark.md)).

## Why this exists

`pretraining/nano` proves *honest methodology* on a 1-hidden-layer model with a
known irreducible-loss floor. The GPU tooling (`tools/train_lora.py`,
`tools/runpod_train.py`, RLVR) only ever *fine-tunes someone else's base*. This
package is the missing rung: a **real GPT trained from random init**, so the
scaling-law / optimizer / MoE studies can re-run on an actual transformer, and so
born-gated training (provenance tokens from token zero) has a model to live in.

## Layout

| File | Deps | Role |
|---|---|---|
| `tokenizer.py` | none | byte-level codec; **reserves** `<src>`, `</src>`, `<abstain>`, `<conf_hi/lo>`, `<doNotMergeWith>`, … at ids ≥256 so the born-gated vocab is stable now |
| `data.py` | none | `training/corpus.jsonl` → eot-separated token stream (+ synthetic fallback) |
| `cluster.py` | torch (lazy) | device/tier resolver: **Spark** (CUDA/bf16, never MLX) · **M3** (MPS) · CPU; never raises |
| `model.py` | torch | readable decoder-only GPT (SDPA causal attention — no x86-only flash-attn wheel) |
| `train.py` | torch | device-agnostic loop; logs `epoch_loss`/`grad_norms` like `nano`; writes `gpt-train-latest.json` |

The tokenizer/data/cluster layer is **dependency-free and CI-tested**
(`tests/test_gpt_pretraining.py`); the torch model + a real training step are
tested with `pytest.importorskip("torch")` and run on the cluster.

## Quick start

```bash
# dependency-free checks (any machine, CI)
python -m pytest tests/test_gpt_pretraining.py -q

# smoke train (seconds, CPU)
python -m pretraining.gpt.train --quick

# real iteration run on the cluster
python -m pretraining.gpt.train --prefer cuda --steps 5000 --report   # DGX Spark (bf16)
python -m pretraining.gpt.train --prefer mps  --steps 5000 --report   # Mac Studio M3
```

## Cluster mapping (DGX Spark + M3 Ultra)

| Node | `--prefer` | dtype | Use |
|---|---|---|---|
| **DGX Spark** (GB10, aarch64, 128 GB) | `cuda` | bf16 | pretrain + iteration; `resolve_tier` tags it `spark`, never MLX |
| **Mac Studio M3 Ultra** (96 GB) | `mps` | fp16 | pretrain/eval on Apple; hand LoRA/serve to MLX tooling (`tools/eval_mlx_model.py`) |
| **CPU / CI** | `cpu` | fp32 | tokenizer + smoke |

Both nodes hold the whole small model in unified memory — this tier is
data-bound, not compute-bound. Headline evidence still goes to x86 RunPod
(`tools/runpod_train.py`); `tools/spark_vs_runpod_ab.py` keeps that rule
data-backed.

## Roadmap (the brainstorm ideas, in order)

1. ✅ **Scaling law on the real GPT** — `pretraining/gpt/scaling.py` trains at a
   geometric token-budget schedule, fits `L(D)=E+A·D^-p` via
   `pretraining/scaling/fit.py`, and runs a pre-registered extrapolation gate
   (floor = uniform upper bound). `python -m pretraining.gpt.scaling --quick`.
2. ✅ **Born-gated pretraining (idea #1)** — `pretraining/gpt/born_gated.py` turns
   `data/attributions.json` into inline `<src>`/`<conf_*>`/`<doNotAttributeTo>`
   text; `python -m pretraining.gpt.train --born-gated`. Next: ablate
   attribution-hallucination vs. a plain-text twin at equal perplexity.
3. ✅ **Abstention head (idea #3)** — `model.py` optional 3-way `accept|hedge|abstain`
   head; `pretraining/gpt/abstain.py` trains it on the provenance signal and
   scores it with `agent/calibration.py` (ECE / risk-coverage).
4. ✅ **Verifier-in-the-loss (idea #2)** — `pretraining/gpt/verifier_loss.py`:
   provenance penalty → DPO preference loss (+ `sequence_reward` hook for the
   existing RLVR stack).
5. ✅ **Born-gated ablation (idea #1 experiment)** — `pretraining/gpt/ablation.py`
   trains born-gated vs plain arms and scores forbidden-attribution rate
   (`pretraining/gpt/provenance_eval.py`).
6. ✅ **Tokenizer fairness (idea #6)** — `pretraining/gpt/tokenizer_analysis.py`
   (EN/中文 byte-tax + lineage-term separation; finding: CJK ≈ 3× byte tax, 0
   lineage collisions).
7. ✅ **Born-gated model card (idea #8)** — [`agi-proof/model-cards/sophia-gpt-nano.md`](../../agi-proof/model-cards/sophia-gpt-nano.md), failure-ledger-first.
8. **Next (needs the cluster/GPU):** multi-seed real pretrain → interpret the
   ablation sign; distill the council into this base (idea #4,
   `tools/distill_council_traces.py`); MoE/quant with the trust governor (idea #7).
